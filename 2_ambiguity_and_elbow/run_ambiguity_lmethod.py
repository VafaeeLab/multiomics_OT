#!/usr/bin/env python3
"""L-method / exp(10a) ambiguity, three new configurations, into runs_ambiguity/.

Written ALONGSIDE the original chord-rule folders, never over them, giving six
per (dataset, coupling):

    runs_ambiguity/<regime>/<pair-type>/<dataset>/<coupling>/
        local_consistency/           ORIGINAL chord rule   (untouched)
        pseudo_label/                ORIGINAL chord rule   (untouched)
        label_margin/                ORIGINAL chord rule   (untouched)
        pseudo_label_pseudo_label/   mod1 pseudo,  mod2 pseudo
        pseudo_label_label_margin/   mod1 pseudo,  mod2 margin
        label_margin_pseudo_label/   mod1 margin,  mod2 pseudo

New folders are named <mod1 score>_<mod2 score>.

The two mixed folders are the realistic one-sided-labels case. A margin score
for one modality consumes the OTHER modality's true labels, so scoring mod1 by
margin requires mod2 to be labelled, and mod2 must then fall back to
pseudo_label, which uses only mod1's Leiden clusters. So
label_margin_pseudo_label is what can actually be run when only mod2 carries
labels, and pseudo_label_label_margin when only mod1 does. Margin on BOTH sides
is deliberately absent: it needs labels everywhere, which is the situation
where ambiguity detection has least to offer, and the original label_margin
folder already covers it as a ceiling reference.

Only four distinct scores exist across these configurations, so each is
computed once and shared. pseudo_label runs Leiden on the full normalised
matrix and dominates the runtime; computing it per folder would triple the job
for no reason.

Threshold: L-method on exp(10*a), no clamp. The transform is strictly
increasing, so within any one modality the cell RANKING is identical to the
original method - only the cut moves. See elbow_lmethod.py.

local_consistency is not produced here; ~71% of its scores are exactly zero and
exp() collapses its threshold to 2-3%, which loses to random removal. It keeps
its original folder and rule.

usage: run_ambiguity_lmethod.py --dataset DS --coupling {diffusion,cca_ot,maxfuse,ensemble}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
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

from ambiguity_detection import (label_margin_ambiguity,      # noqa: E402
                                 pseudo_label_ambiguity)
from run_ambiguity_all import PT, PCTS, load_norm, topk_flags  # noqa: E402
import elbow_lmethod as EL                                     # noqa: E402


def write_modality_lm(m, ids, scores, k, outdir):
    """Per-cell outputs for one modality. Returns (n_flagged, wide frame).

    Deliberately does NOT create a removal_lists/ directory. The old layout
    wrote 19 thresholds x 2 modalities as separate CSVs - 38 files per folder,
    which across 1872 folders was ~71k inodes and the single largest cause of
    the project going over its inode quota. The same information is one frame
    with one boolean column per threshold, returned here and written once per
    folder as a workbook.
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
    for p in PCTS:
        wide[f"rm{p:02d}"] = topk_flags(scores, int(round(p / 100.0 *
                                                          len(scores))))
    # most ambiguous first, so reading down a column walks the removal order
    wide = wide.sort_values("rank_by_ambiguity").reset_index(drop=True)
    return int(is_amb.sum()), wide


def write_removals_xlsx(outdir, frames):
    """One workbook per folder: a sheet per modality, rm05..rm95 as columns.

    Replaces the removal_lists/ directory entirely. Rows run most ambiguous to
    least, so rm05 marks the first 5% of the sheet, rm10 the first 10%, and so
    on - the removal order is readable straight down the sheet.
    """
    path = f"{outdir}/removals_5_95.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for m, wide in sorted(frames.items()):
            wide.to_excel(xl, sheet_name=f"mod{m}", index=False)
    return path

# folder -> (mod1 score, mod2 score); the folder name is <mod1>_<mod2>
CONFIGS = {
    "pseudo_label_pseudo_label": ("pseudo", "pseudo"),
    "pseudo_label_margin_label": ("pseudo", "margin"),
    "margin_label_pseudo_label": ("margin", "pseudo"),
}
# a margin score for one modality reads the OTHER modality's true labels
NEEDS_LABELS = {"pseudo": None, "margin": "other"}


def elbow_png(scores, k, path, title, kexp):
    """Sorted curve on the transformed axis, with the chosen cut marked.

    Plotted on exp(k*a) because that is the curve the threshold was read from;
    drawing the raw curve would show a cut sitting at no visible feature.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return
    v = np.sort(EL.transform(s, kexp))[::-1]
    x = 100.0 * np.arange(len(v)) / len(v)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, v, lw=1.6, color="#1f4e79")
    ax.plot([x[0], x[-1]], [v[0], v[-1]], ls=":", lw=1.1, color="#999")
    if k:
        pct = 100.0 * k / len(v)
        ax.axvline(pct, color="#c00", lw=1.5)
        ax.annotate(f"{pct:.1f}% flagged", xy=(pct, v[0]), xytext=(6, -6),
                    textcoords="offset points", color="#c00", fontsize=9,
                    va="top")
    ax.axvline(50, color="#444", lw=1.0, ls="-.", alpha=.5)
    ax.set_xlabel("% of cells (most ambiguous first)")
    ax.set_ylabel(f"exp({kexp:g}·score)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--coupling", required=True,
                    choices=["diffusion", "cca_ot", "maxfuse", "ensemble"])
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--k", type=float, default=EL.K_DEFAULT)
    ap.add_argument("--out", default=f"{AMB_ROOT}")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    want = [c for c in a.configs.split(",") if c.strip()]
    bad = [c for c in want if c not in CONFIGS]
    if bad:
        raise SystemExit(f"unknown config(s): {bad}; choose from {list(CONFIGS)}")

    MAP = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))
    v = MAP[a.dataset]
    reg, ptr = v["regime"], v["pair_type"]
    pt = PT[ptr]
    gp = f"{OT_ROOT}/{reg}/{pt}/{a.dataset}/{a.coupling}/gamma.npy"
    if not os.path.exists(gp):
        raise SystemExit(f"no coupling: {gp}")

    base = f"{a.out}/{reg}/{pt}/{a.dataset}/{a.coupling}"
    todo = [c for c in want
            if a.force or not os.path.exists(f"{base}/{c}/summary.json")]
    if not todo:
        print(f"[{a.dataset} / {a.coupling}] all configs present, nothing to do",
              flush=True)
        print(f"AMB_LM_DONE {a.dataset} {a.coupling}", flush=True)
        return

    # guard: these folders must never be the originals
    for c in todo:
        if c in ("pseudo_label", "label_margin", "local_consistency"):
            raise SystemExit(f"refusing to write original folder {c}")

    G = np.asarray(np.load(gp, mmap_mode="r"), np.float64)
    n1, n2 = G.shape
    print(f"[{a.dataset} / {a.coupling}] {reg}/{pt}  gamma={G.shape}  "
          f"rule=lmethod_exp{a.k:g}  configs={','.join(todo)}", flush=True)

    d = f"{P}/data/{a.dataset}"
    obs = pd.read_csv(f"{P}/data/{a.dataset}"
                      "/obs_normalized.csv")
    o1 = obs[obs.modality == "mod1"]
    o2 = obs[obs.modality == "mod2"]
    ids1 = o1.cell_id.astype(str).to_numpy()
    ids2 = o2.cell_id.astype(str).to_numpy()
    y1 = o1.cell_type.astype(str).to_numpy()
    y2 = o2.cell_type.astype(str).to_numpy()
    if (len(y1), len(y2)) != (n1, n2):
        raise SystemExit(f"shape mismatch: G{G.shape} y{len(y1)},{len(y2)}")

    need = {s for c in todo for s in CONFIGS[c]}
    X1 = X2 = None
    if "pseudo" in need:
        X1 = load_norm(f"{d}/mod1_normalized.mtx")
        X2 = load_norm(f"{d}/mod2_normalized.mtx")
        if (X1.shape[0], X2.shape[0]) != (n1, n2):
            raise SystemExit(f"shape mismatch: G{G.shape} X1{X1.shape} "
                             f"X2{X2.shape}")

    # each of the four scores computed at most once, then shared
    S = {}
    t0 = time.time()
    if "pseudo" in need:
        S[("pseudo", 1)] = pseudo_label_ambiguity(G, X2, a.res)
        S[("pseudo", 2)] = pseudo_label_ambiguity(G.T, X1, a.res)
        print(f"  pseudo_label scores  {time.time()-t0:.0f}s", flush=True)
    if "margin" in need:
        t1 = time.time()
        S[("margin", 1)] = label_margin_ambiguity(G, y2)
        S[("margin", 2)] = label_margin_ambiguity(G.T, y1)
        print(f"  label_margin scores  {time.time()-t1:.0f}s", flush=True)

    for cfg in todo:
        src1, src2 = CONFIGS[cfg]
        out = f"{base}/{cfg}"
        os.makedirs(out, exist_ok=True)
        t1 = time.time()
        s1, s2 = S[(src1, 1)], S[(src2, 2)]
        tag = f"{cfg} · {a.dataset} · {reg}/{pt} · {a.coupling}"
        k1 = EL.knee(s1, a.k, label=f"{tag} · mod1({src1})")
        k2 = EL.knee(s2, a.k, label=f"{tag} · mod2({src2})")
        cap1 = k1 == int(EL.MAX_FRAC * n1)
        cap2 = k2 == int(EL.MAX_FRAC * n2)
        na1, w1 = write_modality_lm(1, ids1, s1, k1, out)
        na2, w2 = write_modality_lm(2, ids2, s2, k2, out)
        write_removals_xlsx(out, {1: w1, 2: w2})
        elbow_png(s1, k1, f"{out}/elbow_mod1.png",
                  f"{a.dataset} · {a.coupling} · mod1 · {src1}", a.k)
        elbow_png(s2, k2, f"{out}/elbow_mod2.png",
                  f"{a.dataset} · {a.coupling} · mod2 · {src2}", a.k)
        d1, d2 = EL.diagnostics(s1, a.k), EL.diagnostics(s2, a.k)
        json.dump({"dataset": a.dataset, "regime": reg, "pair_type": pt,
                   "coupling": a.coupling, "detector": cfg,
                   "rule": f"lmethod_exp{a.k:g}", "k": a.k,
                   # capped_mod* records where the 50% policy limit decided the
                   # threshold instead of the curve - those runs are not
                   # comparable with the rest and must be readable as such
                   "max_frac": EL.MAX_FRAC,
                   "capped_mod1": bool(cap1), "capped_mod2": bool(cap2),
                   "score_mod1": src1, "score_mod2": src2,
                   # a margin score for one modality reads the OTHER
                   # modality's true labels; pseudo reads none
                   "labels_used_mod1": ("mod2_true" if src1 == "margin"
                                        else "none"),
                   "labels_used_mod2": ("mod1_true" if src2 == "margin"
                                        else "none"),
                   "uses_true_labels": "margin" in (src1, src2),
                   "one_sided_labels": src1 != src2,
                   "n_mod1": n1, "n_mod2": n2,
                   "knee_mod1": k1, "knee_mod2": k2,
                   "pct_knee_mod1": round(100.0 * k1 / n1, 2),
                   "pct_knee_mod2": round(100.0 * k2 / n2, 2),
                   "n_ambiguous_mod1": na1, "n_ambiguous_mod2": na2,
                   "mean_score_mod1": float(np.nanmean(s1)),
                   "mean_score_mod2": float(np.nanmean(s2)),
                   "diagnostics_mod1": d1, "diagnostics_mod2": d2,
                   "removal_pcts": PCTS,
                   "seconds": round(time.time() - t1, 1)},
                  open(f"{out}/summary.json", "w"), indent=2)
        print(f"  {cfg:<22} mod1={k1} ({100*k1/n1:.1f}%, {src1})  "
              f"mod2={k2} ({100*k2/n2:.1f}%, {src2})", flush=True)

    print(f"AMB_LM_DONE {a.dataset} {a.coupling}", flush=True)


if __name__ == "__main__":
    main()
