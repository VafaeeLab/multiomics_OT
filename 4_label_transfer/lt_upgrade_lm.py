#!/usr/bin/env python3
"""LT-BlendGate upgrade at lambda=0.5 with query-side ambiguity removal.

One job per (tool, dataset); all four couplings are handled inside, since they
share the baseline predictions and the label bookkeeping.

Four arms per (coupling, set):

    baseline     no fusion, no removal - the reference the upgrade must beat
    upgraded     LT-BlendGate at lambda=0.5, every cell kept
    amb_removed  upgraded, the QUERY's ambiguous cells dropped
    rand_removed upgraded, the SAME NUMBER of query cells dropped at random,
                 averaged over 3 draws

TWO SETS, one per choice of reference. The detector configuration differs
between them, because the query's ambiguity is scored by the margin rule
against the REFERENCE modality's true cell types - the annotation label
transfer already grants. The query's own labels are never used.

  set A  reference = mod1, query = mod2 -> pseudo_label_margin_label
         that folder scores mod2 by the margin rule on mod1's labels. Its mod1
         score is the label-free one and is not used here.

  set B  reference = mod2, query = mod1 -> margin_label_pseudo_label
         mirror image: mod1 is scored by margin on mod2's labels.

Reading the folder names: a margin score for one modality consumes the OTHER
modality's labels, so the folder whose query half is the margin one is named
for the opposite modality. An earlier version of this script had the two
mapped the other way round, which scored the query label-free; that run is
archived under runs_label_transfer/archive/labelfree_query_score_20260908/.

Only the QUERY modality's ambiguous cells are removed. Label-transfer metrics
are computed on query cells alone, so masking reference cells after the fact
would change nothing - and masking them BEFORE would change the predictions,
which is a different experiment (a cleaned reference) rather than removal.

ORDER MATTERS: fuse first, then drop. The upgrade is computed once from the
FULL coupling and the FULL reference, and removal is applied only afterwards,
to the query cells being scored. Rebuilding the vote matrix from a reduced
reference would change the fused predictions themselves, so the removal arms
would no longer be measuring removal.

AnchorAE-SVM is excluded from the tool roster; it does not perform well enough
to be informative and its near-flat bars compress every figure it appears in.

The random arm is the one that matters. Removal always raises metrics because
discarding cells makes the remaining problem easier, so beating `upgraded`
proves nothing on its own; only beating a count-matched random draw shows the
detector picked the RIGHT cells.

Ambiguous barcodes come from the per-set folder named above:
    runs_ambiguity/<reg>/<pt>/<ds>/<coupling>/<detector>/
        ambiguous_cell_ids_<query modality>.csv
the elbow selection under the L-method/exp(10a) rule with the 50% cap. The
older removal_lists/removal_lists.xlsx path no longer exists - that directory
was removed during the inode cleanup - so the lists are read directly.

Metrics: accuracy, balanced_accuracy, macro_f1, weighted_f1.

usage: lt_upgrade_lm.py <tool> <dataset>
"""
import glob
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfuse

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
# the baseline predictions this arm builds on, and where the four-arm result
# goes. Both are overridable so the corrected tree can be used end to end:
# the custom tools' co-assay baselines changed, and every arm derived from them
# has to be recomputed from the corrected predictions rather than the old ones.
LT = os.environ.get("LT_ROOT", f"{LT_ROOT}")
OUT = os.environ.get("LT_UPGRADE_OUT", f"{LT}/upgrade_lm")
# direction -> (reference modality, ambiguity folder whose QUERY score is the
# margin one computed from the REFERENCE modality's true labels.
#
# Label transfer already grants the reference's annotation, so using it to score
# the query's ambiguity reveals nothing the task withholds, and true cell types
# give a sharper margin than Leiden clusters do. The query's OWN labels are
# never touched.
#
# Reading the folder names: a margin score for a modality consumes the OTHER
# modality's labels, so the folder whose query half is "margin_label" is the one
# named for the opposite modality. mod1->mod2 therefore reads
# pseudo_label_margin_label (mod2 scored by margin, on mod1's labels), and
# mod2->mod1 reads margin_label_pseudo_label.
DIR2DET = {"mod1_to_mod2": ("mod1", "pseudo_label_margin_label"),
           "mod2_to_mod1": ("mod2", "margin_label_pseudo_label")}
SETNAME = {"mod1_to_mod2": "A_ref_mod1", "mod2_to_mod1": "B_ref_mod2"}
COUPLINGS = ["maxfuse", "ensemble", "cca_ot", "diffusion"]
LAMBDA = 0.5
RAND_DRAWS = 3
RANK_NORMALISE = {"scmap"}
LT_REGIME = {"coassay": "coassay", "diagonal": "closedset"}
PTL = {"rna_adt": "rna-adt", "rna_atac": "rna-atac",
       "rna_meth": "rna-methylome", "rna_rna": "rna-rna"}
MET = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]


def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p),
                balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro", zero_division=0),
                weighted_f1=f1_score(y, p, average="weighted", zero_division=0))


def rank_norm(s):
    s = pd.to_numeric(s, errors="coerce")
    ok = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if ok.sum():
        out[ok] = s[ok].rank(method="average") / (ok.sum() + 1.0)
    return out.to_numpy()


def seed(*parts):
    """Deterministic across jobs; Python's hash() is salted per process."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def main():
    tool, ds = sys.argv[1], sys.argv[2]
    m = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))[ds]
    reg, ptr = m["regime"], m["pair_type"]
    ptl = PTL[ptr]
    d = f"{P}/Sample_dataset/{ds}"
    b1 = [l.strip() for l in open(f"{d}/mod1_barcodes_normalized.tsv") if l.strip()]
    b2 = [l.strip() for l in open(f"{d}/mod2_barcodes_normalized.tsv") if l.strip()]
    obs = pd.read_csv(f"{d}/obs.csv", dtype=str)
    lab = {(r.modality, r.cell_id): r.cell_type for r in obs.itertuples()}

    pf = f"{LT}/{tool}/{LT_REGIME[reg]}/baseline/{ds}/predictions.csv"
    if not os.path.exists(pf):
        raise SystemExit(f"no baseline predictions: {pf}")
    pred = pd.read_csv(pf, dtype={"cell_id": str, "target_cell_id": str})
    idc = "cell_id" if "cell_id" in pred.columns else "target_cell_id"
    pred = pred.rename(columns={idc: "cell_id"})
    pred["direction"] = pred["direction"].str.replace("->", "_to_", regex=False)
    if "prediction_score" not in pred.columns:
        pred["prediction_score"] = np.nan

    pos = {"mod1": {c: i for i, c in enumerate(b1)},
           "mod2": {c: i for i, c in enumerate(b2)}}
    bars = {"mod1": b1, "mod2": b2}
    rows = []

    for coupling in COUPLINGS:
        g = glob.glob(f"{OT_ROOT}/{reg}/*/{ds}/{coupling}/gamma.npy")
        if not g:
            print(f"  [skip] no {coupling} coupling", flush=True)
            continue
        G = np.load(g[0])
        if G.shape != (len(b1), len(b2)):
            print(f"  [skip] {coupling} shape {G.shape}", flush=True)
            continue

        # each direction reads its OWN detector folder: the query's ambiguity
        # comes from the margin score built on the reference modality's labels
        amb = {}
        for dirn, (_refmod, det) in DIR2DET.items():
            qm = "mod2" if dirn == "mod1_to_mod2" else "mod1"
            f = (f"{AMB_ROOT}/{reg}/{ptl}/{ds}/{coupling}/{det}"
                 f"/ambiguous_cell_ids_{qm}.csv")
            if os.path.exists(f):
                try:
                    amb[dirn] = set(
                        pd.read_csv(f, dtype=str)["cell_id"].astype(str))
                except Exception as e:
                    print(f"  [warn] {f}: {e}", flush=True)

        for direction, dd in pred.groupby("direction"):
            if direction not in DIR2DET:
                continue
            refmod, det = DIR2DET[direction]
            qmod = "mod2" if direction == "mod1_to_mod2" else "mod1"
            rmod = "mod1" if qmod == "mod2" else "mod2"
            dd = dd[dd.cell_id.isin(pos[qmod])].copy()
            if not len(dd):
                continue
            qids = dd.cell_id.tolist()
            qidx = np.array([pos[qmod][c] for c in qids])
            qset = set(qids)
            rids = [c for c in bars[rmod] if (rmod, c) in lab
                    and (reg != "coassay" or c not in qset)]
            if not rids:
                continue
            ridx = np.array([pos[rmod][c] for c in rids])
            y_ref = np.array([lab[(rmod, c)] for c in rids])
            y_true = dd.true_label.astype(str).to_numpy()
            base_pred = dd.predicted_label.astype(str).to_numpy()
            cls = sorted(set(y_ref) | set(base_pred))
            ci = {c: i for i, c in enumerate(cls)}
            sc = (rank_norm(dd.prediction_score) if tool in RANK_NORMALISE
                  else pd.to_numeric(dd.prediction_score,
                                     errors="coerce").to_numpy())
            P_base = gfuse.soft_base(
                np.array([ci.get(p, -1) for p in base_pred]), sc, len(cls))
            Gc = G if rmod == "mod1" else G.T
            S = Gc[np.ix_(ridx, qidx)]
            V = np.zeros((len(qidx), len(cls)))
            for j, c in enumerate(cls):
                sel = y_ref == c
                if sel.any():
                    V[:, j] = S[sel].sum(0)
            V = gfuse.sharpen(V)
            # fused ONCE, from the full reference: removal must not alter it
            fused = np.array(cls)[
                gfuse.gate_fuse(P_base, V, lam_max=LAMBDA).argmax(1)]

            def add(arm, keep, p, pct):
                if keep.sum() < 10 or len(np.unique(y_true[keep])) < 2:
                    return None
                rows.append(dict(
                    tool=tool, dataset=ds, regime=reg, pair_type=ptl,
                    lt_set=SETNAME[direction], direction=direction,
                    reference_modality=refmod, query_modality=qmod,
                    coupling=coupling, detector=det, arm=arm, lam=LAMBDA,
                    pct_removed=round(pct, 2), n=int(keep.sum()),
                    **metrics(y_true[keep], p[keep])))

            allk = np.ones(len(qids), bool)
            add("baseline", allk, base_pred, 0.0)
            add("upgraded", allk, fused, 0.0)

            a = amb.get(direction)
            if a is None:
                continue
            keep = np.array([c not in a for c in qids])
            ndrop = int((~keep).sum())
            if ndrop < 1 or ndrop >= len(qids) - 10:
                continue
            pct = 100.0 * ndrop / len(qids)
            add("amb_removed", keep, fused, pct)

            # count-matched random removal on the same fused predictions
            acc = []
            for t in range(RAND_DRAWS):
                rs = np.random.RandomState(seed(ds, coupling, qmod, t))
                rk = np.ones(len(qids), bool)
                rk[rs.choice(len(qids), ndrop, replace=False)] = False
                if rk.sum() < 10 or len(np.unique(y_true[rk])) < 2:
                    continue
                acc.append(metrics(y_true[rk], fused[rk]))
            if acc:
                rows.append(dict(
                    tool=tool, dataset=ds, regime=reg, pair_type=ptl,
                    lt_set=SETNAME[direction], direction=direction,
                    reference_modality=refmod, query_modality=qmod,
                    coupling=coupling, detector=det, arm="rand_removed",
                    lam=LAMBDA, pct_removed=round(pct, 2),
                    n=int(len(qids) - ndrop),
                    **{k: float(np.mean([x[k] for x in acc])) for k in MET}))

    if not rows:
        raise SystemExit(f"nothing scored for {tool}/{ds}")
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{tool}__{ds}.csv", index=False)
    print(f"LTUP_DONE {tool} {ds} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
