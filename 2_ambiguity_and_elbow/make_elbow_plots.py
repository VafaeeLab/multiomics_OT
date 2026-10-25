#!/usr/bin/env python3
"""One elbow figure per modality pair, from a single coupling.

The ambiguity step writes a per-cell score for 78 dataset pairs, four
couplings, three detector configurations and two modalities, which is far more
curves than a reader needs. This script draws one figure for each of the six
modality pairs instead, using one coupling and one representative dataset.

    co-assay RNA-ADT, RNA-ATAC, RNA-methylome
    diagonal RNA-ATAC, RNA-methylome, RNA-RNA

The coupling is Ensemble-OT by default, because that is the coupling the paper
selects. Pass --coupling to use another one.

The representative dataset is the median-sized pair of its group, measured by
the total number of cells in both modalities. Taking the median rather than
the largest or the smallest avoids choosing a pair whose curve is unusual only
because of its size. Pass --dataset to choose a different one, in which case
only that pair's figure is drawn.

Each figure has two panels, one per modality, so a pair is shown whole.

The curve plotted is exp(10*a), not the raw score, because that is the curve
the threshold was actually taken from. Plotting the raw score would put the cut
at a point with no visible feature and invite the reader to distrust a correct
number - the raw curve is close to a straight line, which is the whole reason
the transform exists.

Design: one hue for the data, one accent for the decision, everything else
recessive. The removed region is filled rather than outlined so the eye reads
"this much was discarded" without decoding a legend.

    usage: make_elbow_plots.py [--coupling ensemble] [--detector NAME]
                               [--dataset DS] [--root DIR] [--out DIR]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
ROOT = f"{AMB_ROOT}"
OUT = f"{ROOT}/elbow_plots"
K = 10.0
INK = "#12263f"
CURVE = "#1f4e79"
FILL = "#c0392b"
MUTE = "#8a94a6"
CPLL = {"maxfuse": "MaxFuse", "ensemble": "Ensemble-OT", "cca_ot": "CCA-OT",
        "diffusion": "Diffusion-OT"}
DETL = {"pseudo_label_pseudo_label": "pseudo-label (both modalities)",
        "pseudo_label_margin_label": "pseudo-label / margin-label",
        "margin_label_pseudo_label": "margin-label / pseudo-label",
        "pseudo_label": "pseudo-label", "label_margin": "margin-label",
        "local_consistency": "local consistency"}
PAIRLAB = {"rna-adt": "RNA–ADT", "rna-atac": "RNA–ATAC",
           "rna-methylome": "RNA–methylome", "rna-rna": "RNA–RNA"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": "#c8cdd6", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def thousands(x, _pos):
    return f"{int(x):,}"


def draw(ax, scores, k):
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = len(s)
    v = np.sort(np.clip(s, 0, 1))[::-1]
    y = np.exp(K * (v - 1.0))
    x = np.arange(1, n + 1)

    ax.plot(x, y, color=CURVE, lw=1.9, solid_capstyle="round", zorder=3)
    if k and k < n:
        # filled region = the cells discarded; reads without a legend
        ax.fill_between(x[:k], y[:k], y[k - 1], color=FILL, alpha=0.16,
                        lw=0, zorder=2)
        ax.axvline(k, color=FILL, lw=1.4, zorder=4)
        ax.annotate(f"{k:,} cells", xy=(k, 0.97), xycoords=("data",
                    "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", color=FILL, fontsize=8.5,
                    fontweight="semibold", va="top", zorder=5)
    cap = int(0.5 * n)
    ax.axvline(cap, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    # both labels are pinned to fixed axes-fraction heights, not to the curve:
    # anchoring to the data put grey text on the line where the cap fell on a
    # shallow tail, and the two labels collided whenever the elbow sat near
    # the cap. Fixed heights keep them clear of the data and of each other.
    ax.annotate("50% cap", xy=(cap, 0.86), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", color=MUTE,
                fontsize=7.5, va="top", ha="left", zorder=5)

    ax.set_xlim(0, n)
    ax.set_ylim(0, max(y[0] * 1.06, 1e-9))
    ax.set_xlabel("number of cells")
    ax.set_ylabel("transformed ambiguity   exp(10·a)", color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#eef1f5", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    return n


def count_rows(path):
    """Cells in a per-cell file, without parsing it."""
    try:
        with open(path) as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    cpl = arg("--coupling", "ensemble")
    det = arg("--detector", "pseudo_label_pseudo_label")
    want = arg("--dataset")
    root = arg("--root", ROOT)
    out = arg("--out", f"{root}/elbow_plots")

    # every dataset that has this coupling and this detector
    found = {}
    for sj in sorted(glob.glob(f"{root}/*/*/*/{cpl}/{det}/summary.json")):
        d = os.path.dirname(sj)
        reg, pt, ds = d.split("/")[-5:-2]
        n = sum(count_rows(f"{d}/per_cell_ambiguity_mod{m}.csv")
                for m in (1, 2))
        if n:
            found.setdefault((reg, pt), []).append((n, ds, d, sj))
    if not found:
        raise SystemExit(f"no results under {root}/*/*/*/{cpl}/{det}/ - "
                         f"run the ambiguity step for coupling '{cpl}' first")

    # one representative per modality pair: the median-sized dataset
    chosen = []
    for key in sorted(found):
        group = sorted(found[key])
        pick = next((g for g in group if g[1] == want), None) \
            if want else group[len(group) // 2]
        if pick:
            chosen.append((key, pick, len(group)))
    if want and not chosen:
        raise SystemExit(f"dataset '{want}' has no results for {cpl}/{det}")

    os.makedirs(out, exist_ok=True)
    made = 0
    for (reg, pt), (_n, ds, d, sj), ngroup in chosen:
        meta = json.load(open(sj))
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
        drew = []
        for m, ax in zip((1, 2), axes):
            f = f"{d}/per_cell_ambiguity_mod{m}.csv"
            sc = pd.read_csv(f, usecols=["ambiguity_score"])[
                "ambiguity_score"].dropna().to_numpy(float)
            k = int(meta.get(f"knee_mod{m}") or 0)
            n = draw(ax, sc, k)
            capped = bool(meta.get(f"capped_mod{m}"))
            pct = 100.0 * k / n if n else 0.0
            ax.set_title(f"modality {m}   ·   {k:,} of {n:,} cells flagged "
                         f"({pct:.1f}%)"
                         + ("   [cut at the 50% cap]" if capped else ""),
                         fontsize=9, color=INK, pad=6)
            drew.append(n)
        fig.suptitle(
            f"{PAIRLAB.get(pt, pt)}  ·  {reg}  ·  {ds}\n"
            f"{CPLL.get(cpl, cpl)}  ·  {DETL.get(det, det)}  ·  "
            f"elbow on exp(10·a), cells ranked most → least ambiguous",
            fontsize=10.5, color=INK, y=1.07)
        fig.tight_layout()
        name = f"{out}/elbow__{reg}__{pt}__{cpl}.png"
        fig.savefig(name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"  {reg:<9} {pt:<15} {ds:<24} "
              f"{drew[0]:>7,} + {drew[1]:>7,} cells   "
              f"(median of {ngroup})", flush=True)
    print(f"  wrote {made} figures -> {out}")


if __name__ == "__main__":
    main()
