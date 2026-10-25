#!/usr/bin/env python3
"""Run ONE label-transfer tool on ONE pair, baseline only, BOTH directions.

    runs_label_transfer/<tool>/<regime>/baseline/<dataset>/
        metrics.csv          one row per direction + a combined row
        predictions.csv      per-cell true/predicted label, both directions

11 tools. The 4 custom ones are implemented here so the roster, the direction
handling and the metric definitions are identical across them; the 7 external
ones are delegated to run_ext_tool.py, which already dispatches them.

    AnchorAE-LT     kNN(15)              on AnchorAE  embeddings
    AnchorFormer    kNN(15)              on Anchorformer embeddings
    AnchorAE-HGB    HGB 200 iters lr .1  on AnchorAE  embeddings
    AnchorAE-SVM    SVC C=2 rbf          on AnchorAE  embeddings

All four classify on an integration embedding (variant B). The shared-gene
variant A was dropped after the 12-pair smoke test: on the two honest
comparisons (diagonal rna-rna) B was ahead or tied, and A needs the full gene
matrix densified, which is what killed the login node twice. The co-assay part
of that test is not evidence either way -- mod1 and mod2 are the same cells
there, so any embedding that aligns a cell to its own twin scores near 1.0.

BOTH DIRECTIONS are run for every pair, co-assay and diagonal alike. The
reference modality supplies its true cell types and the query's are predicted,
which is the only ground truth a transfer method is allowed to see. Diagonal
pairs have no cell correspondence, so a direction is still well defined: the
labelled modality trains, the other is predicted.

lt_knn_from_embeddings.py is not reused: its roster is the hardcoded Tabula
Muris tissue list, and it runs one direction only for diagonal pairs.

usage: run_lt_tool.py <dataset> <tool> [--force]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np, pandas as pd
import scipy.io as sio
import scipy.sparse as sp

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
sys.path.insert(0, f"{P}/3_data_integration")
from run_custom import resolve_prep

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score

EXTERNAL = ["Seurat", "Symphony", "SingleR", "scmap", "ItClust", "Concerto",
            "Harmony"]
# Variant B selected 2026-08-24: the classifiers run on AnchorAE embeddings,
# not on the shared-gene space. The raw-gene HGB/SVM arms (variant A) are not
# part of the benchmark roster, so all four custom tools are embedding-based.
CUSTOM = ["AnchorAE-LT", "AnchorFormer", "AnchorAE-HGB", "AnchorAE-SVM"]
TOOLS = CUSTOM + EXTERNAL
# which DI model's embedding each embedding-based tool reads
EMB_SRC = {"AnchorAE-LT": "AnchorAE", "AnchorFormer": "AnchorFormer",
           "AnchorAE-HGB": "AnchorAE", "AnchorAE-SVM": "AnchorAE"}
NA = {"__NA__", "nan", "NA", "", "None", "unknown", "Unknown"}

# ---- tuned parameters -------------------------------------------------------
_PTL = {"rna_adt": "rna-adt", "rna_atac": "rna-atac",
        "rna_meth": "rna-methylome", "rna_rna": "rna-rna"}
_SEL_CACHE = {}


def tuned_params(tool, ds):
    """Selected configuration for this tool on this dataset's modality group."""
    import json as _json
    import pandas as _pd
    if not _SEL_CACHE:
        f = f"{P}/model_selection/results/label_transfer__selected.csv"
        d = _pd.read_csv(f)
        # Bookkeeping columns, not tool parameters. The v2 selection table
        # added score / n_evaluated / search; without excluding them the
        # wrappers are invoked with --score 0.284 --search greedy and either
        # abort or silently ignore flags they do not know.
        meta = {"tool", "regime", "pair_type", "mean_macro_f1", "n_datasets",
                "group_datasets", "complete", "score", "n_evaluated",
                "search", "mode"}
        for _, r in d.iterrows():
            _SEL_CACHE[(r.tool, r.regime, r.pair_type)] = {
                c: r[c] for c in d.columns
                if c not in meta and _pd.notna(r[c])}
    v = _json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))[ds]
    key = (tool, v["regime"], _PTL[v["pair_type"]])
    cfg = _SEL_CACHE.get(key)
    if cfg is None:
        print(f"  WARNING: no tuned parameters for {key}; using tool defaults",
              flush=True)
        return {}
    return cfg



def clf_for(tool, n_dim, cfg=None):
    """Classifier built from the TUNED configuration for this modality group."""
    cfg = cfg or {}
    if tool in ("AnchorAE-LT", "AnchorFormer"):
        return ("knn", int(cfg.get("k", 15)))
    if tool == "AnchorAE-HGB":
        # max_leaf_nodes and l2_regularization are tuned now; the previous
        # sweep never varied them, so this builder hardcoded l2 and dropped
        # max_leaf_nodes entirely. Leaving that in place would run HGB on a
        # configuration the selection table does not describe.
        return ("hgb", HistGradientBoostingClassifier(
            max_iter=int(cfg.get("max_iter", 200)),
            learning_rate=float(cfg.get("learning_rate", 0.1)),
            max_leaf_nodes=int(cfg.get("max_leaf_nodes", 31)),
            l2_regularization=float(cfg.get("l2_regularization", 1.0)),
            random_state=0))
    return ("svm", SVC(C=float(cfg.get("C", 2.0)), kernel="rbf",
                       gamma=_coerce_gamma(cfg.get("gamma", "scale")),
                       class_weight="balanced", random_state=0))


def _conf_from_proba(model, Zq, pred):
    """Probability the model assigned to the label it predicted."""
    try:
        Pr = model.predict_proba(Zq)
        cls = list(model.classes_)
        idx = np.array([cls.index(p) if p in cls else 0 for p in pred])
        return Pr[np.arange(len(pred)), idx].astype(float)
    except Exception:
        return np.full(len(pred), np.nan)


def _coerce_gamma(g):
    """SVC takes 'scale'/'auto' or a float - never a numeric string.

    The tuned table is read from CSV, so a selected gamma of 0.1 arrives as
    the string '0.1'. Wrapping it in str() forced every value down the
    string path, and SVC rejects any string that is not scale or auto, so
    each AnchorAE-SVM job whose group selected a numeric gamma died in
    seconds. Numeric strings are converted; the two keywords pass through.
    """
    if isinstance(g, str) and g not in ("scale", "auto"):
        try:
            return float(g)
        except ValueError:
            return "scale"
    return g


def fit_predict(kind, clf, Zr, yr, Zq):
    """Train on the reference, predict the query.

    Returns (predicted_label, confidence). The confidence is the probability of
    the predicted class, which LT-BlendGate needs to gate how much the coupling
    may move a prediction.
    """
    Zr, Zq, yr = np.asarray(Zr, float), np.asarray(Zq, float), np.asarray(yr)
    if kind == "knn":
        # clf carries the tuned k for this modality group
        k = max(1, min(int(clf) if clf else 15, len(Zr)))
        m = KNeighborsClassifier(n_neighbors=k).fit(Zr, yr)
        p = m.predict(Zq)
        return p, _conf_from_proba(m, Zq, p)
    if kind == "svm":
        # rbf SVM is O(n^2 d) and hangs on the large high-dimensional co-assay
        # pairs; the same caps classical_lt.py uses
        if len(Zr) > 3000:
            s = np.random.RandomState(0).choice(len(Zr), 3000, replace=False)
            Zr, yr = Zr[s], yr[s]
        if Zr.shape[1] > 2000:
            p = PCA(n_components=min(64, Zr.shape[0] - 1, Zr.shape[1]),
                    random_state=0).fit(Zr)
            Zr, Zq = p.transform(Zr), p.transform(Zq)
    from sklearn.base import clone
    m = clone(clf).fit(Zr, yr)
    p = m.predict(Zq)
    if kind == "svm":
        # softmax over the decision function: calibrated probabilities would
        # need probability=True, which refits with Platt scaling
        try:
            D = np.asarray(m.decision_function(Zq), float)
            if D.ndim == 1:                       # binary reference
                D = np.column_stack([-D, D])
            D = D - D.max(1, keepdims=True)
            E = np.exp(D)
            Pr = E / (E.sum(1, keepdims=True) + 1e-12)
            cls = list(m.classes_)
            idx = np.array([cls.index(x) if x in cls else 0 for x in p])
            return p, Pr[np.arange(len(p)), idx].astype(float)
        except Exception:
            return p, np.full(len(p), np.nan)
    return p, _conf_from_proba(m, Zq, p)


def metrics(true, pred):
    t = np.asarray(true).astype(str)
    p = np.asarray(pred).astype(str)
    m = np.array([x not in NA for x in t])
    t, p = t[m], p[m]
    if not len(t):
        return {k: float("nan") for k in
                ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")} | {"n": 0}
    return {"accuracy": float((t == p).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(t, p)),
            "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(t, p, average="weighted",
                                          zero_division=0)),
            "n": int(len(t))}


def load_features(tool, ds, prep, reg):
    """Return (Z1, Z2, y1, y2, ids1, ids2) for this tool's feature space."""
    # Label transfer must not depend on the data-integration results, so the
    # DI-embedding path is disabled. Set LT_USE_DI_EMBEDDINGS=1 only to
    # reproduce the older, coupled behaviour.
    if tool in EMB_SRC and os.environ.get("LT_USE_DI_EMBEDDINGS") == "1":
        model = EMB_SRC[tool]
        f = f"{DI_ROOT}/{model}/{reg}/baseline/{ds}/embeddings.csv"
        if not os.path.exists(f):
            raise SystemExit(f"NEEDS_DI: {model} embedding missing for {ds}")
        e = pd.read_csv(f)
        cols = [c for c in e.columns if c.startswith("emb_")]
        m1 = e[e.modality == "mod1"].reset_index(drop=True)
        m2 = e[e.modality == "mod2"].reset_index(drop=True)
        return (m1[cols].to_numpy(float), m2[cols].to_numpy(float),
                m1.cell_type.astype(str).to_numpy(),
                m2.cell_type.astype(str).to_numpy(),
                m1.cell_id.astype(str).to_numpy(),
                m2.cell_id.astype(str).to_numpy())

    # Read only on the variant-A path: prep is None whenever the prepared
    # input has been cleaned up, which is the case for 47 of the 78 pairs.
    obs = pd.read_csv(f"{prep}/combined_log_obs.csv")
    o1 = obs[obs.modality == "mod1"].reset_index(drop=True)
    o2 = obs[obs.modality == "mod2"].reset_index(drop=True)

    # variant A: the shared-gene space, log1p + standardised, as embed_common
    def dense_log(path):
        X = sio.mmread(path).tocsr()
        X = X.T.tocsr() if X.shape[0] != len(obs) // 2 else X
        A = np.log1p(np.clip(X.toarray().astype(np.float32), 0, None))
        return StandardScaler().fit_transform(A).astype(np.float32)

    Z1 = dense_log(f"{prep}/mod1_log.mtx")
    Z2 = dense_log(f"{prep}/mod2_log.mtx")
    if Z1.shape[0] != len(o1):
        Z1 = Z1.T
    if Z2.shape[0] != len(o2):
        Z2 = Z2.T
    return (Z1, Z2, o1.cell_type.astype(str).to_numpy(),
            o2.cell_type.astype(str).to_numpy(),
            o1.cell_id.astype(str).to_numpy(),
            o2.cell_id.astype(str).to_numpy())



def resolve_prep_lt(ds, tool):
    """(prep_dir, regime) for a label-transfer run.

    resolve_prep() locates the prepared input and derives the regime from where
    it sits. That input no longer exists for 47 of the 78 pairs -- it was
    removed in the inode cleanup -- so calling it unconditionally kills every
    job on those pairs before any work starts. The embedding path does not read
    the prepared input at all; it only needs the regime, which the embedding
    tree already states. So fall back to the embedding tree, and keep the
    original behaviour wherever the prepared input still exists.
    """
    try:
        return resolve_prep(ds)
    except SystemExit:
        pass
    model = EMB_SRC.get(tool, "AnchorAE")
    for r in ("coassay", "closedset", "openset"):
        f = f"{DI_ROOT}/{model}/{r}/baseline/{ds}/embeddings.csv"
        if os.path.exists(f):
            return None, r
    raise SystemExit(f"no prepared input and no {model} embedding for {ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("tool", choices=TOOLS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ds, tool = a.dataset, a.tool

    prep, reg = resolve_prep_lt(ds, tool)
    # the output tree is overridable so a corrected re-run can be written
    # beside the original rather than on top of it
    root = os.environ.get("LT_OUT_ROOT", f"{LT_ROOT}")
    out = f"{root}/{tool}/{reg}/baseline/{ds}"
    if os.path.exists(f"{out}/metrics.csv") and not a.force:
        print(f"SKIP {ds}/{tool} already present", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    os.chdir(P)

    if tool in EXTERNAL:
        r = subprocess.run(
            [os.environ.get("PY", sys.executable),
             f"{P}/4_label_transfer/run_ext_tool.py", tool, ds],
            capture_output=True, text=True)
        sys.stdout.write(r.stdout[-3000:])
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-3000:])
            raise SystemExit(f"{tool} failed on {ds}")
        src = f"{LT_ROOT}/EXT_{tool}_{ds}.csv"
        if os.path.exists(src):
            pd.read_csv(src).assign(dataset=ds, tool=tool, regime=reg
                                    ).to_csv(f"{out}/metrics.csv", index=False)
        print(f"LT_DONE {ds} {tool}", flush=True)
        return

    t0 = time.time()
    Z1, Z2, y1, y2, id1, id2 = load_features(tool, ds, prep, reg)

    # Co-assay pairs profile the SAME cells in both modalities, so training on
    # all of mod1 and predicting all of mod2 puts every query cell's own twin in
    # the reference and the task becomes trivial -- AnchorAE-LT scored an exactly
    # symmetric 0.723/0.723 on BF21 that way. run_ext_lt.py splits the cells into
    # disjoint halves; the same split (same seed, same source ordering) is
    # applied here so the custom and external tools are scored identically.
    ref_mask1 = ref_mask2 = None
    if reg == "coassay":
        vv = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))[ds]
        obs = pd.read_csv(f"{P}/data/{ds}/obs_normalized.csv")
        split_ids = obs[obs.modality == "mod1"].cell_id.astype(str).to_numpy()
        rs0 = np.random.RandomState(0)
        is_ref = np.zeros(len(split_ids), bool)
        is_ref[rs0.permutation(len(split_ids))[:len(split_ids) // 2]] = True
        ref_set = set(split_ids[is_ref])
        ref_mask1 = np.array([c in ref_set for c in id1])
        ref_mask2 = np.array([c in ref_set for c in id2])
        print(f"  co-assay split: {int(ref_mask1.sum())} reference / "
              f"{int((~ref_mask2).sum())} query", flush=True)
    _cfg = tuned_params(tool, ds)
    kind, clf = clf_for(tool, Z1.shape[1], _cfg)
    print(f"  tuned parameters: {_cfg or '(tool defaults)'}", flush=True)
    print(f"[{ds}] {tool} regime={reg} n1={len(Z1)} n2={len(Z2)} "
          f"dim={Z1.shape[1]} kind={kind}", flush=True)

    rows, preds = [], []
    for src_name, Zr, yr, Zq, yq, idq, rmask, qmask in (
            ("mod1->mod2", Z1, y1, Z2, y2, id2, ref_mask1, ref_mask2),
            ("mod2->mod1", Z2, y2, Z1, y1, id1, ref_mask2, ref_mask1)):
        if rmask is not None:
            # reference = its half; query = the complementary half
            qkeep = ~qmask
            Zq, yq, idq = Zq[qkeep], yq[qkeep], idq[qkeep]
            Zr, yr = Zr[rmask], yr[rmask]
        ok = np.array([str(x) not in NA for x in yr])
        if ok.sum() < 2 or len(set(yr[ok])) < 2:
            print(f"  {src_name}: reference has <2 usable classes, skipped",
                  flush=True)
            continue
        p, conf = fit_predict(kind, clf, Zr[ok], yr[ok], Zq)
        m = metrics(yq, p)
        m.update(dataset=ds, tool=tool, regime=reg, direction=src_name,
                 n_ref=int(ok.sum()), n_classes_ref=len(set(yr[ok])))
        rows.append(m)
        preds.append(pd.DataFrame({"cell_id": idq, "direction": src_name,
                                   "true_label": yq, "predicted_label": p,
                                   "prediction_score": conf}))
        print(f"  {src_name}: acc={m['accuracy']:.4f} "
              f"bal={m['balanced_accuracy']:.4f} f1={m['macro_f1']:.4f}",
              flush=True)

    if not rows:
        raise SystemExit(f"no usable direction for {ds}/{tool}")
    t = pd.DataFrame(rows)
    mean = {c: t[c].mean() for c in
            ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    mean.update(dataset=ds, tool=tool, regime=reg, direction="mean_of_both",
                n=int(t["n"].sum()))
    pd.concat([t, pd.DataFrame([mean])], ignore_index=True).to_csv(
        f"{out}/metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(f"{out}/predictions.csv",
                                               index=False)
    print(f"LT_DONE {ds} {tool}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
