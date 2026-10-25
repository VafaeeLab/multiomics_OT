#!/usr/bin/env python3
"""
plot_sorted_foscttm.py — SORTED per-cell line plots + MULTIPANEL grids, for FOSCTTM & nFOSCTTM.

Benchmarks diffusion_v1, diffusion_v2, diffusion_v3 (=diffusion_v10) against
SCOT / UnionCom / MMD-MA on ONE honest ruler: per-cell metrics are RECOMPUTED from each
method's pairwise_matrix_common.npy.
  * FOSCTTM  : MID-RANK tie handling
        foscttm_i = ( #{j: D[i,j]<D[i,i]} + 0.5*#{j!=i: D[i,j]==D[i,i]} ) / (n-1)
    (removes the strict-< float32-tie artifact that flattered near-uniform -gamma).
  * nFOSCTTM : mean-over-violators distance-weighted penalty
        nfoscttm_i = mean_{j: D[i,j]<D[i,i]} max(0,(D[i,i]-D[i,j])/D[i,i])
Both averaged over row & col. Lower = better for both.

Outputs:
  <outdir>/sorted/sorted_<metric>_<dataset>.png(.pdf)        per-dataset sorted line plot
  <outdir>/multipanel/multipanel_<metric>.png(.pdf)          all-datasets grid (one panel/dataset)
  <outdir>/foscttm_nfoscttm_summary.csv                      mean metrics per method,dataset
where <metric> in {foscttm_sym, nfoscttm_sym}.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"diffusion_v1": "diffusion_v1", "diffusion_v2": "diffusion_v2",
          "diffusion_v3": "diffusion_v10", "scot": "SCOT", "unioncom": "UnionCom", "mmdma": "MMD-MA"}
# maximally-distinct palette (Trubetskoy) + solid (diffusion) vs dashed (baselines)
COLORS = {"diffusion_v1": "#E6194B", "diffusion_v2": "#911EB4", "diffusion_v3": "#F58231",
          "scot": "#4363D8", "unioncom": "#3CB44B", "mmdma": "#000000"}
STYLES = {"diffusion_v1": "-", "diffusion_v2": "-", "diffusion_v3": "-",
          "scot": "--", "unioncom": "--", "mmdma": "--"}
ORDER = ["diffusion_v1", "diffusion_v2", "diffusion_v3", "scot", "unioncom", "mmdma"]
METRICS = [("foscttm_sym", "FOSCTTM (symmetric)"), ("nfoscttm_sym", "nFOSCTTM (symmetric)")]
EPS = 1e-12


def run_dir(model, dataset, args):
    if model == "diffusion_v1": return Path(args.diffusion_v1_dir) / dataset
    if model == "diffusion_v2": return Path(args.diffusion_v2_dir) / dataset
    if model == "diffusion_v3": return Path(args.diffusion_v3_dir) / dataset
    return Path(args.other_results_base) / model / dataset


def metrics_from_matrix(D, chunk=2048):
    """Return dict foscttm_sym (mid-rank), nfoscttm_sym (penalty). D square, diag=true match."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    mn = np.nanmin(D[np.isfinite(D)]) if np.isfinite(D).any() else 0.0
    if mn != 0.0:  # shift so negatives (-gamma) don't blow up the penalty denominator
        D = D - mn
    diag = np.diag(D).copy()
    denom = np.maximum(diag, EPS)
    fos_r = np.empty(n); fos_c = np.empty(n); nf_r = np.empty(n); nf_c = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n); blk = D[s:e]; dv = diag[s:e][:, None]
        less = (blk < dv).sum(1); eq = (blk == dv).sum(1) - 1
        fos_r[s:e] = (less + 0.5 * eq) / max(1, n - 1)
        pen = np.maximum(0.0, (dv - blk) / denom[s:e][:, None]); pen[blk >= dv] = 0.0
        nviol = less
        nf_r[s:e] = np.where(nviol > 0, pen.sum(1) / np.maximum(nviol, 1.0), 0.0)
    for s in range(0, n, chunk):
        e = min(s + chunk, n); blk = D[:, s:e]; dv = diag[s:e][None, :]
        less = (blk < dv).sum(0); eq = (blk == dv).sum(0) - 1
        fos_c[s:e] = (less + 0.5 * eq) / max(1, n - 1)
        pen = np.maximum(0.0, (dv - blk) / denom[s:e][None, :]); pen[blk >= dv] = 0.0
        nviol = less
        nf_c[s:e] = np.where(nviol > 0, pen.sum(0) / np.maximum(nviol, 1.0), 0.0)
    return {"foscttm_sym": 0.5 * (fos_r + fos_c), "nfoscttm_sym": 0.5 * (nf_r + nf_c)}


def load_metrics(model, dataset, args):
    d = run_dir(model, dataset, args)
    for fn in ("pairwise_matrix_common.npy", "pairwise_matrix_common_normalized_0_1.npy"):
        p = d / fn
        if p.exists():
            try:
                D = np.load(p)
                if D.ndim == 2 and D.shape[0] == D.shape[1]:
                    return metrics_from_matrix(D)
            except Exception as ex:
                print(f"  [warn] {model}/{dataset}: {ex}")
    csv = d / "per_cell_nfoscttm_metrics.csv"
    if csv.exists():
        df = pd.read_csv(csv); out = {}
        for m, aliases in [("foscttm_sym", ["foscttm_sym", "foscttm_symmetric"]),
                            ("nfoscttm_sym", ["nfoscttm_sym", "nfoscttm_symmetric"])]:
            for a in aliases:
                if a in df.columns: out[m] = df[a].to_numpy(float); break
        if out: return out
    return None


def infer_datasets(args):
    base = Path(args.diffusion_v1_dir); ds = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and ((d / "pairwise_matrix_common.npy").exists() or (d / "per_cell_nfoscttm_metrics.csv").exists()):
                ds.append(d.name)
    return ds


def draw(ax, series, metric):
    for m in ORDER:
        v = series.get(m, {}).get(metric)
        if v is None or len(v) == 0: continue
        sv = np.sort(np.asarray(v, float)); x = np.arange(1, len(sv) + 1)
        ax.plot(x, sv, color=COLORS[m], ls=STYLES.get(m, "-"), lw=1.8, label=f"{LABELS[m]} ({np.mean(v):.3f})")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("number of cells"); ax.set_ylabel(metric + " (lower=better)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diffusion_v1_dir", required=True)
    ap.add_argument("--diffusion_v2_dir", required=True)
    ap.add_argument("--diffusion_v3_dir", required=True)
    ap.add_argument("--other_results_base", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--write_pdf", action="store_true")
    args = ap.parse_args()

    datasets = args.datasets or infer_datasets(args)
    if not datasets:
        raise SystemExit("No datasets found under --diffusion_v1_dir")
    out = Path(args.outdir)
    for _short in ("foscttm", "nfoscttm"):
        (out / _short / "sorted").mkdir(parents=True, exist_ok=True)
        (out / _short / "multipanel").mkdir(parents=True, exist_ok=True)
    all_series, summ = {}, []
    for ds in datasets:
        series = {}
        for m in ORDER:
            mt = load_metrics(m, ds, args)
            if mt is not None:
                series[m] = mt
                row = {"dataset": ds, "method": m}
                for k in ("foscttm_sym", "nfoscttm_sym"):
                    if k in mt: row["mean_" + k] = float(np.mean(mt[k]))
                summ.append(row)
        all_series[ds] = series
        for metric, title in METRICS:
            if not any(metric in series.get(m, {}) for m in series): continue
            short = metric.replace("_sym", "")
            fig, ax = plt.subplots(figsize=(7, 5)); draw(ax, series, metric)
            ax.set_title(f"Sorted per-cell {title} — {ds}", fontsize=12); fig.tight_layout()
            fig.savefig(out / short / "sorted" / f"sorted_{metric}_{ds}.png", dpi=300, bbox_inches="tight")
            if args.write_pdf: fig.savefig(out / short / "sorted" / f"sorted_{metric}_{ds}.pdf", bbox_inches="tight")
            plt.close(fig)
        print(f"[ok] {ds}: {sorted(series)}", flush=True)
    # multipanel grids (one subplot per dataset) per metric
    nz = [ds for ds in datasets if all_series.get(ds)]
    for metric, title in METRICS:
        if not nz: break
        ncol = min(4, len(nz)); nrow = math.ceil(len(nz) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.6, nrow * 3.8), squeeze=False)
        for ax, ds in zip(axes.ravel(), nz):
            draw(ax, all_series[ds], metric); ax.set_title(ds, fontsize=9); ax.legend(fontsize=5)
        for ax in axes.ravel()[len(nz):]: ax.axis("off")
        fig.suptitle(f"Sorted per-cell {title} — all datasets", fontsize=14, y=1.005); fig.tight_layout()
        short = metric.replace("_sym", "")
        fig.savefig(out / short / "multipanel" / f"multipanel_{metric}.png", dpi=300, bbox_inches="tight")
        if args.write_pdf: fig.savefig(out / short / "multipanel" / f"multipanel_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(summ).to_csv(out / "foscttm_nfoscttm_summary.csv", index=False)
    print("DONE. sorted/ + multipanel/ in", out)


if __name__ == "__main__":
    main()
