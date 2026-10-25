#!/usr/bin/env python3
"""Ambiguity detection for one (dataset, coupling) pair, all three detectors.

    runs_ambiguity/<regime>/<pair-type>/<dataset>/<coupling>/<detector>/
        per_cell_ambiguity_mod{1,2}.csv     rank, score, is_ambiguous at the knee
        ambiguous_cell_ids_mod{1,2}.csv     the knee selection
        removal_lists/rm{05..95}_mod{1,2}.csv
        elbow_mod{1,2}.png
        summary.json

Detectors, each run in BOTH directions -- mod1 scored from G, mod2 from G.T --
following score_modalities:

    pseudo_label       Leiden clusters of the reference matrix; label-free, and
                       the frozen production default
    label_margin       same margin computation but on TRUE cell types; a ceiling
                       reference, not a usable selector, flagged in summary.json
    local_consistency  on an embedding derived from the coupling itself (SVD of
                       the row-normalised G), not from any integration tool's
                       output, so the flags stay a property of the coupling

top1_mass is deliberately absent: across the 12-pair test it was worse than
removing the same number of cells at random in 5 of 6 modality groups
(mean gap -0.031, 15% win rate).

Nothing is subsampled. The 12-pair test capped the detectors at 6000 cells
because it only needed a ranking, but the removal lists here have to name every
cell, so the full coupling is scored.

usage: run_ambiguity_all.py --dataset DS --coupling {diffusion,cca_ot,maxfuse,ensemble}
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- project root and output roots -------------------------------------------
# On the cluster these were fixed paths. Each one now reads an environment
# variable and falls back to the original value, so the code behaves exactly as
# it did before unless a variable is set. The worked example in 0_example_data
# sets them to OT_Results, Data_integration_results and Label_transfer_results.
P = os.environ.get("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration")
OT_ROOT = os.environ.get("OT_ROOT", P + "/runs")
AMB_ROOT = os.environ.get("AMB_ROOT", P + "/runs_ambiguity")
DI_ROOT = os.environ.get("DI_ROOT", P + "/runs_data_integration")
LT_ROOT = os.environ.get("LT_ROOT", P + "/runs_label_transfer")
# -----------------------------------------------------------------------------
sys.path.insert(0, f"{P}/2_ambiguity_and_elbow")
from ambiguity_detection import (label_margin_ambiguity, pseudo_label_ambiguity,
                                 local_consistency_ambiguity)

PT = {"rna_adt": "rna-adt", "rna_atac": "rna-atac",
      "rna_rna": "rna-rna", "rna_meth": "rna-methylome"}
DETECTORS = ["pseudo_label", "label_margin", "local_consistency"]
PCTS = list(range(5, 100, 5))          # 5..95, the frozen removal-list grid


def load_norm(path):
    return sio.mmread(path).tocsr().T.tocsr()      # features x cells -> cells x features


def knee(scores):
    """Max distance to the chord, clamped to [8%, 50%] as production does."""
    s = np.asarray(scores, float)
    f = np.isfinite(s)
    if f.sum() < 20:
        return 0
    v = np.sort(s[f])[::-1]
    n = len(v)
    if np.allclose(v, v[0]):
        return 0
    x = np.arange(n, dtype=float)
    d = np.abs((v[-1] - v[0]) * x - (n - 1) * v + (n - 1) * v[0]) / \
        (np.hypot(v[-1] - v[0], n - 1) + 1e-12)
    return int(np.clip(int(np.argmax(d)), int(0.08 * n), int(0.50 * n)))


def topk_flags(scores, k):
    a = np.zeros(len(scores), bool)
    if k > 0:
        s = np.where(np.isfinite(scores), scores, -np.inf)
        a[np.argsort(-s)[:k]] = True
    return a


def elbow_png(scores, k, path, title):
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])[::-1]
    if not len(s):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(np.arange(len(s)), s, lw=1.4, color="#1f3b63")
    if k > 0:
        ax.axvline(k, color="#c1121f", ls="--", lw=1.2)
        ax.fill_between(np.arange(k), s[:k], s.min(), color="#c1121f", alpha=.12)
        ax.text(k, s[0], f"  knee={k} ({100*k/len(s):.0f}%)", va="top",
                fontsize=8, color="#c1121f")
    ax.set_xlabel("cells ranked by ambiguity")
    ax.set_ylabel("ambiguity score")
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def write_modality(m, ids, scores, k, outdir):
    """Every per-cell output for one modality.

    Each modality is scored and thresholded independently -- mod1 from G, mod2
    from G.T -- so it carries its own knee, its own barcode lists and its own
    elbow plot, and removal is applied on both sides of the coupling.

        per_cell_ambiguity_mod{m}.csv    rank, score, is_ambiguous at the knee
        ambiguous_cell_ids_mod{m}.csv    barcodes flagged at the knee
        nonambiguous_cell_ids_mod{m}.csv the complement -- the cells to keep
        removal_5_95_mod{m}.csv          one file, one boolean column per
                                         threshold (rm05..rm95)
        removal_lists/rm{p}_mod{m}.csv   the same selections split per
                                         threshold, for scripts that want to
                                         read one level without filtering
    """
    scores = np.asarray(scores, float)
    is_amb = topk_flags(scores, k)
    order = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf))
    rank = np.empty(len(scores), int)
    rank[order] = np.arange(1, len(scores) + 1)
    cid = np.asarray(ids, object).astype(str)
    df = pd.DataFrame({"cell_id": cid, "rank_by_ambiguity": rank,
                       "ambiguity_score": scores, "is_ambiguous": is_amb})
    df.to_csv(f"{outdir}/per_cell_ambiguity_mod{m}.csv", index=False)
    df.loc[is_amb, ["cell_id"]].to_csv(
        f"{outdir}/ambiguous_cell_ids_mod{m}.csv", index=False)
    df.loc[~is_amb, ["cell_id"]].to_csv(
        f"{outdir}/nonambiguous_cell_ids_mod{m}.csv", index=False)

    wide = df[["cell_id", "rank_by_ambiguity", "ambiguity_score"]].copy()
    rd = f"{outdir}/removal_lists"
    os.makedirs(rd, exist_ok=True)
    for p in PCTS:
        kk = int(round(p / 100.0 * len(scores)))
        f = topk_flags(scores, kk)
        wide[f"rm{p:02d}"] = f
        df.loc[f, ["cell_id"]].to_csv(f"{rd}/rm{p:02d}_mod{m}.csv", index=False)
    wide.to_csv(f"{outdir}/removal_5_95_mod{m}.csv", index=False)
    return int(is_amb.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--coupling", required=True,
                    choices=["diffusion", "cca_ot", "maxfuse", "ensemble"])
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    MAP = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))
    v = MAP[a.dataset]
    reg, ptr = v["regime"], v["pair_type"]
    pt = PT[ptr]
    gp = f"{OT_ROOT}/{reg}/{pt}/{a.dataset}/{a.coupling}/gamma.npy"
    if not os.path.exists(gp):
        raise SystemExit(f"no coupling: {gp}")

    base = f"{AMB_ROOT}/{reg}/{pt}/{a.dataset}/{a.coupling}"
    G = np.asarray(np.load(gp, mmap_mode="r"), np.float64)
    n1, n2 = G.shape
    print(f"[{a.dataset} / {a.coupling}] {reg}/{pt}  gamma={G.shape}", flush=True)

    d = f"{P}/Sample_dataset/{a.dataset}"
    X1 = load_norm(f"{d}/mod1_normalized.mtx")
    X2 = load_norm(f"{d}/mod2_normalized.mtx")
    obs = pd.read_csv(f"{P}/Sample_dataset/{a.dataset}"
                      "/obs_normalized.csv")
    o1 = obs[obs.modality == "mod1"]
    o2 = obs[obs.modality == "mod2"]
    ids1 = o1.cell_id.astype(str).to_numpy()
    ids2 = o2.cell_id.astype(str).to_numpy()
    y1 = o1.cell_type.astype(str).to_numpy()
    y2 = o2.cell_type.astype(str).to_numpy()
    if (X1.shape[0], X2.shape[0]) != (n1, n2) or (len(y1), len(y2)) != (n1, n2):
        raise SystemExit(f"shape mismatch: G{G.shape} X1{X1.shape} X2{X2.shape} "
                         f"y{len(y1)},{len(y2)}")

    # embedding for local_consistency, derived from the coupling itself
    Z = None

    for det in DETECTORS:
        out = f"{base}/{det}"
        if os.path.exists(f"{out}/summary.json") and not a.force:
            print(f"  SKIP {det} (already done)", flush=True); continue
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        if det == "pseudo_label":
            s1 = pseudo_label_ambiguity(G, X2, a.res)
            s2 = pseudo_label_ambiguity(G.T, X1, a.res)
        elif det == "label_margin":
            s1 = label_margin_ambiguity(G, y2)
            s2 = label_margin_ambiguity(G.T, y1)
        else:
            if Z is None:
                C = G / (G.sum(1, keepdims=True) + 1e-12)
                k = max(2, min(30, min(C.shape) - 1))
                U, S, Vt = sp.linalg.svds(sp.csr_matrix(C), k=k)
                Z = np.vstack([U * S, Vt.T * S])
            lc = local_consistency_ambiguity(Z, np.arange(n1),
                                             np.arange(n1, n1 + n2))
            s1, s2 = lc[:n1], lc[n1:]

        k1, k2 = knee(s1), knee(s2)
        na1 = write_modality(1, ids1, s1, k1, out)
        na2 = write_modality(2, ids2, s2, k2, out)
        elbow_png(s1, k1, f"{out}/elbow_mod1.png",
                  f"{a.dataset} · {a.coupling} · mod1 · {det}")
        elbow_png(s2, k2, f"{out}/elbow_mod2.png",
                  f"{a.dataset} · {a.coupling} · mod2 · {det}")
        json.dump({"dataset": a.dataset, "regime": reg, "pair_type": pt,
                   "coupling": a.coupling, "detector": det,
                   "uses_true_labels": det == "label_margin",
                   "n_mod1": n1, "n_mod2": n2,
                   "knee_mod1": k1, "knee_mod2": k2,
                   "pct_knee_mod1": round(100.0 * k1 / n1, 2),
                   "pct_knee_mod2": round(100.0 * k2 / n2, 2),
                   "n_ambiguous_mod1": na1, "n_ambiguous_mod2": na2,
                   "mean_score_mod1": float(np.nanmean(s1)),
                   "mean_score_mod2": float(np.nanmean(s2)),
                   "removal_pcts": PCTS,
                   "seconds": round(time.time() - t0, 1)},
                  open(f"{out}/summary.json", "w"), indent=2)
        print(f"  {det:<18} knee mod1={k1} ({100*k1/n1:.0f}%) "
              f"mod2={k2} ({100*k2/n2:.0f}%)  {time.time()-t0:.0f}s", flush=True)

    print(f"AMB_DONE {a.dataset} {a.coupling}", flush=True)


if __name__ == "__main__":
    main()
