#!/usr/bin/env python3
"""ItClust label-transfer wrapper for the final benchmark.

The script prepares AnnData objects from the two modalities and calls the public
ItClust tutorial API when the ItClust package is installed:
    clf = ic.transfer_learning_clf(); clf.fit(adata_train, adata_test)

Because historical ItClust releases expose predictions under slightly different
attribute names, the wrapper searches common attributes and adata_test.obs fields.
If the package/API is unavailable, a method_status.json is written and the global
benchmark continues.
"""
from __future__ import annotations

# --- Force ItClust onto legacy Keras 2 (tf-keras) without changing env Keras 3 ---
import os as _os
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
try:
    import sys as _sys
    import tf_keras as _tfk
    _sys.modules.setdefault("keras", _tfk)
except Exception:
    pass

import argparse, importlib, json, os, sys, traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from label_transfer_core import load_matrix, preprocess_matrix, read_labels, align_label_array


def write_status(out_dir: Path, dataset_id: str, method: str, status: str, message: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "method_status.json", "w") as fh:
        json.dump({"dataset_id": dataset_id, "method": method, "status": status, "message": message}, fh, indent=2)
    pd.DataFrame([{"dataset_id": dataset_id, "method": method, "status": status, "message": message}]).to_csv(out_dir / "method_status.csv", index=False)


def load_common(args):
    X1, ids1, feats1 = load_matrix(args.x1, args.barcodes1, args.features1, args.orientation1)
    X2, ids2, feats2 = load_matrix(args.x2, args.barcodes2, args.features2, args.orientation2)
    # ItClust.fit() runs its own normalize_per_cell+log1p+scale pipeline (and drops
    # low-count cells); feed raw counts so its preprocessing is not applied twice.
    f1 = pd.Index(feats1.astype(str)); f2 = pd.Index(feats2.astype(str))
    # harmonized names can collapse to duplicates (e.g. GENE.1 -> GENE); keep first occurrence
    # so the feature Index is unique and get_indexer() is valid.
    m1 = ~f1.duplicated(); m2 = ~f2.duplicated()
    if (~m1).any(): X1 = X1[:, np.where(m1)[0]]; f1 = f1[m1]; feats1 = f1.to_numpy()
    if (~m2).any(): X2 = X2[:, np.where(m2)[0]]; f2 = f2[m2]; feats2 = f2.to_numpy()
    shared = f1.intersection(f2)
    if len(shared) >= args.min_shared_features:
        idx1 = f1.get_indexer(shared); idx2 = f2.get_indexer(shared)
        return X1[:, idx1].tocsr(), X2[:, idx2].tocsr(), ids1, ids2, shared.astype(str).to_numpy(), "feature_id"
    if X1.shape[1] == X2.shape[1] and X1.shape[1] >= args.min_shared_features:
        feats = np.array([f"feature_{i:06d}" for i in range(X1.shape[1])], dtype=str)
        return X1, X2, ids1, ids2, feats, "feature_index"
    raise ValueError(f"No comparable feature space for ItClust: shared_features={len(shared)}, n_features1={X1.shape[1]}, n_features2={X2.shape[1]}")


def import_itclust():
    errors = []
    for name in ["ItClust", "itclust", "ItClust_package", "itclust_package"]:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import ItClust. Tried: " + "; ".join(errors))


def make_adata(X, ids, feats, labels):
    try:
        import anndata as ad
    except Exception:
        import scanpy as sc  # type: ignore
        ad = None
    if 'ad' in locals() and ad is not None:
        adata = ad.AnnData(X=X.copy())
    else:
        adata = sc.AnnData(X=X.copy())
    adata.obs_names = ids.astype(str)
    adata.var_names = feats.astype(str)
    adata.obs["cell_type"] = pd.Series(labels, index=adata.obs_names).astype(str)
    adata.obs["labels"] = adata.obs["cell_type"]
    # ItClust's fit() reads reference labels from obs["celltype"] (no underscore).
    adata.obs["celltype"] = adata.obs["cell_type"]
    return adata


def extract_predictions(clf, adata_test, n: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    attr_names = ["y_pred", "y_pred_", "pred", "predicted", "predicted_labels", "labels_pred", "y_trans", "y_test_pred"]
    for nm in attr_names:
        if hasattr(clf, nm):
            val = getattr(clf, nm)
            try:
                arr = np.asarray(val)
                if arr.shape[0] == n:
                    return arr.astype(str), None
            except Exception:
                pass
    for col in ["predicted_label", "predicted_labels", "cell_type_pred", "ItClust", "label", "labels", "y_pred"]:
        if col in adata_test.obs.columns and len(adata_test.obs[col]) == n:
            return adata_test.obs[col].astype(str).to_numpy(), None
    # Some releases return a tuple/list from fit.
    if hasattr(clf, "history"):
        pass
    return None, None


def _predict_labels(clf, n):
    """Defensive: ItClust's documented API returns predictions from predict().
    Try it without side effects; coerce common return shapes to a length-n array."""
    if not hasattr(clf, "predict"):
        return None
    res = None
    try:
        res = clf.predict(write=False)
    except TypeError:
        try:
            res = clf.predict()
        except Exception:
            return None
    except Exception:
        return None
    candidates = list(res) if isinstance(res, (tuple, list)) else [res]
    for c in candidates:
        if c is None:
            continue
        try:
            if isinstance(c, pd.DataFrame):
                for col in ["cell_type", "celltype", "cell_type_pred", "pred", "prediction", "labels", "label", "assigned"]:
                    if col in c.columns and len(c[col]) == n:
                        return c[col].astype(str).to_numpy()
                for col in c.columns:
                    if c[col].dtype == object and len(c[col]) == n:
                        return c[col].astype(str).to_numpy()
                continue
            if isinstance(c, pd.Series) and len(c) == n:
                return c.astype(str).to_numpy()
            arr = np.asarray(c)
            if arr.ndim == 1 and arr.shape[0] == n:
                return arr.astype(str)
        except Exception:
            continue
    return None


def run_direction(ic, X_ref, X_query, ids_ref, ids_query, feats, y_ref, y_query, direction, args):
    import numpy as _np
    # Reference: keep labelled cells that also have >=1 count in the shared feature space.
    keep = _np.array([x is not None and str(x).lower() not in {"nan", "none", ""} for x in y_ref])
    ref_nz = _np.asarray(X_ref.sum(axis=1)).ravel() > 0
    keep = keep & ref_nz
    adata_train = make_adata(X_ref[keep], ids_ref[keep], feats, y_ref[keep])
    # Query: drop zero-count cells BEFORE ItClust so its internal prefilter_cells(min_genes=1)
    # removes nothing further -> ItClust preserves cell count & order -> safe positional align.
    q_nz = _np.asarray(X_query.sum(axis=1)).ravel() > 0
    Xq = X_query[q_nz]
    idq = _np.asarray(ids_query)[q_nz]
    yq = _np.asarray(y_query)[q_nz]
    adata_test = make_adata(Xq, idq, feats, yq)
    if hasattr(ic, "transfer_learning_clf"):
        clf = ic.transfer_learning_clf()
    elif hasattr(ic, "ItClust") and hasattr(ic.ItClust, "transfer_learning_clf"):
        clf = ic.ItClust.transfer_learning_clf()
    else:
        raise AttributeError("Imported ItClust module does not expose transfer_learning_clf().")
    clf.fit(adata_train, adata_test)
    # ItClust.predict() -> (pred_df[cell_id, cluster], prob_df, celltype_map{cluster:[celltype, conf]}).
    try:
        pred_df, _prob, celltype_map = clf.predict(write=False)
    except TypeError:
        pred_df, _prob, celltype_map = clf.predict()
    cmap = dict(celltype_map)
    cl2ct = {str(k): (v[0] if isinstance(v, (list, tuple)) else v) for k, v in cmap.items()}
    cl2cf = {str(k): (float(v[1]) if isinstance(v, (list, tuple)) and len(v) > 1 else float("nan")) for k, v in cmap.items()}
    pred_clusters = pred_df["cluster"].astype(str).to_numpy()
    # ItClust preserves order and (after our pre-filter) drops no cells -> positional align to idq.
    if len(pred_clusters) == len(idq):
        clusters_sub = pred_clusters
    else:
        # Rare safety net: ItClust still dropped cells -> id match (strip AnnData concat suffix).
        import re as _re
        stripped = np.array([_re.sub(r"-\d+$", "", x) for x in pred_df["cell_id"].astype(str).to_numpy()])
        ser = pd.Series(pred_clusters, index=stripped)
        ser = ser[~ser.index.duplicated()]
        clusters_sub = ser.reindex(np.asarray(idq).astype(str)).to_numpy()
    sub_pred = np.array([cl2ct.get(str(c), "unassigned") for c in clusters_sub], dtype=object)
    sub_score = np.array([cl2cf.get(str(c), float("nan")) for c in clusters_sub], dtype=float)
    # Scatter back to the FULL query id list; zero-count cells -> unassigned.
    pred = np.full(len(ids_query), "unassigned", dtype=object)
    score = np.full(len(ids_query), float("nan"), dtype=float)
    pred[q_nz] = sub_pred
    score[q_nz] = sub_score
    return pred.astype(str), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", required=True)
    ap.add_argument("--x1", required=True); ap.add_argument("--x2", required=True)
    ap.add_argument("--barcodes1", default=None); ap.add_argument("--barcodes2", default=None)
    ap.add_argument("--features1", default=None); ap.add_argument("--features2", default=None)
    ap.add_argument("--labels1", required=True); ap.add_argument("--labels2", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--orientation1", default="cell_by_feature"); ap.add_argument("--orientation2", default="cell_by_feature")
    ap.add_argument("--label_id_col", default="cell_id"); ap.add_argument("--label_col", default="cell_type")
    ap.add_argument("--min_shared_features", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=20)
    args = ap.parse_args()
    method = "label_transfer_ItClust"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ic = import_itclust()
        X1, X2, ids1, ids2, feats, mode = load_common(args)
        lab1_df = read_labels(args.labels1, args.label_id_col, args.label_col)
        lab2_df = read_labels(args.labels2, args.label_id_col, args.label_col)
        y1, _ = align_label_array(ids1, lab1_df); y2, _ = align_label_array(ids2, lab2_df)
        pred12, score12 = run_direction(ic, X1, X2, ids1, ids2, feats, y1, y2, "mod1_to_mod2", args)
        pred21, score21 = run_direction(ic, X2, X1, ids2, ids1, feats, y2, y1, "mod2_to_mod1", args)
        df = pd.concat([
            pd.DataFrame({"dataset_id": args.dataset_id, "method": method, "direction": "mod1_to_mod2", "target_cell_id": ids2.astype(str), "true_label": y2.astype(str), "predicted_label": pred12.astype(str), "prediction_score": score12, "transfer_status": "accepted"}),
            pd.DataFrame({"dataset_id": args.dataset_id, "method": method, "direction": "mod2_to_mod1", "target_cell_id": ids1.astype(str), "true_label": y1.astype(str), "predicted_label": pred21.astype(str), "prediction_score": score21, "transfer_status": "accepted"}),
        ], ignore_index=True)
        df.to_csv(out_dir / "standardized_predictions_raw.csv", index=False)
        pd.DataFrame([{"feature_alignment": mode, "n_shared_features": len(feats)}]).to_csv(out_dir / "feature_alignment.csv", index=False)
    except Exception as exc:
        msg = f"{exc}\n{traceback.format_exc(limit=3)}"
        status = "skipped_not_applicable" if "No comparable feature space" in str(exc) else "skipped_error"
        write_status(out_dir, args.dataset_id, method, status, msg)

if __name__ == "__main__":
    main()
