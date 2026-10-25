#!/usr/bin/env python3
"""Concerto label-transfer wrapper for the final benchmark.

This wrapper follows the public Concerto reproducibility API when available:
  from concerto_function5_3 import preprocessing_rna, concerto_make_tfrecord,
      concerto_train_ref_query, concerto_test_ref_query, knn_classifier

It writes standardized_predictions_raw.csv when successful.  If Concerto or its
TensorFlow dependencies are not installed, method_status.json is written so the
benchmark can continue and the missing method is visible in the final table.
"""
from __future__ import annotations

# Force Concerto/TF onto legacy Keras 2 (tf_keras); Keras 3 rejects dtype="float" used in
# Concerto bgi model layers. Must be set before tensorflow is imported.
import os as _os
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse, importlib, json, os, tempfile, traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier

from label_transfer_core import load_matrix, preprocess_matrix, read_labels, align_label_array


def write_status(out_dir: Path, dataset_id: str, method: str, status: str, message: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_id": dataset_id, "method": method, "status": status, "message": message}
    with open(out_dir / "method_status.json", "w") as fh: json.dump(payload, fh, indent=2)
    pd.DataFrame([payload]).to_csv(out_dir / "method_status.csv", index=False)


def load_common(args):
    X1, ids1, feats1 = load_matrix(args.x1, args.barcodes1, args.features1, args.orientation1)
    X2, ids2, feats2 = load_matrix(args.x2, args.barcodes2, args.features2, args.orientation2)
    # Concerto.preprocessing_rna performs its own normalize_total+log1p; feed raw counts.
    f1 = pd.Index(feats1.astype(str)); f2 = pd.Index(feats2.astype(str))
    # harmonized names can collapse to duplicates (e.g. GENE.1 -> GENE); keep first occurrence
    # so the feature Index is unique and get_indexer() is valid.
    m1 = ~f1.duplicated(); m2 = ~f2.duplicated()
    if (~m1).any(): X1 = X1[:, np.where(m1)[0]]; f1 = f1[m1]; feats1 = f1.to_numpy()
    if (~m2).any(): X2 = X2[:, np.where(m2)[0]]; f2 = f2[m2]; feats2 = f2.to_numpy()
    shared = f1.intersection(f2)
    if len(shared) >= args.min_shared_features:
        return X1[:, f1.get_indexer(shared)].tocsr(), X2[:, f2.get_indexer(shared)].tocsr(), ids1, ids2, shared.astype(str).to_numpy(), "feature_id"
    if X1.shape[1] == X2.shape[1] and X1.shape[1] >= args.min_shared_features:
        feats = np.array([f"feature_{i:06d}" for i in range(X1.shape[1])], dtype=str)
        return X1, X2, ids1, ids2, feats, "feature_index"
    raise ValueError(f"No comparable feature space for Concerto: shared_features={len(shared)}, n_features1={X1.shape[1]}, n_features2={X2.shape[1]}")


def make_adata(X, ids, feats, labels, tech):
    import anndata as ad
    adata = ad.AnnData(X=X.copy())
    adata.obs_names = ids.astype(str)
    adata.var_names = feats.astype(str)
    adata.obs["celltype"] = pd.Series(labels, index=adata.obs_names).astype(str)
    adata.obs["cell_type"] = adata.obs["celltype"]
    adata.obs["tech"] = tech
    return adata


def import_concerto():
    for name in ["concerto_function5_3", "concerto"]:
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    raise ImportError("Could not import concerto_function5_3 or concerto. Add the Concerto reproducibility repo to PYTHONPATH.")


def fallback_embedding_knn(ref_emb, query_emb, ref_labels, k):
    clf = KNeighborsClassifier(n_neighbors=min(k, len(ref_labels)), weights="distance")
    clf.fit(ref_emb, ref_labels.astype(str))
    pred = clf.predict(query_emb)
    try:
        proba = clf.predict_proba(query_emb)
        score = proba.max(axis=1)
    except Exception:
        score = np.full(len(pred), np.nan)
    return pred.astype(str), score


def run_direction(cf, X_ref, X_query, ids_ref, ids_query, feats, y_ref, y_query, direction, args, tmp):
    keep = np.array([x is not None and str(x).lower() not in {"nan", "none", ""} for x in y_ref])
    adata_ref = make_adata(X_ref[keep], ids_ref[keep], feats, y_ref[keep], "ref")
    adata_query = make_adata(X_query, ids_query, feats, y_query, "query")

    # Concerto API path.  The exact output of knn_classifier varies across repo revisions,
    # so we also support extracting embeddings and using sklearn kNN.
    save_path = Path(tmp) / direction
    save_path.mkdir(parents=True, exist_ok=True)
    if hasattr(cf, "preprocessing_rna"):
        adata_ref = cf.preprocessing_rna(adata_ref, min_features=1, batch_key="tech")
        adata_query = cf.preprocessing_rna(adata_query, min_features=1, batch_key="tech")
    ref_tf_path = cf.concerto_make_tfrecord(adata_ref, tf_path=str(save_path / "tfrecord" / "ref_tf"), batch_col_name="tech")
    query_tf_path = cf.concerto_make_tfrecord(adata_query, tf_path=str(save_path / "tfrecord" / "query_tf"), batch_col_name="tech")
    weight_path = str(save_path / "weights")
    cf.concerto_train_ref_query(ref_tf_path, query_tf_path, weight_path)
    ref_embedding, query_embedding, ref_id, query_id = cf.concerto_test_ref_query(weight_path, ref_tf_path, query_tf_path)
    ref_id = np.asarray(ref_id).astype(str)
    query_id = np.asarray(query_id).astype(str)

    # Predict for the cells Concerto actually embedded (preprocessing_rna may drop some).
    pred_sub = None
    score_sub = None
    try:
        q_neighbor, q_prob = cf.knn_classifier(ref_embedding, query_embedding, adata_ref, ref_id, column_name="celltype", k=args.knn_k)
        cand = np.asarray(q_neighbor).astype(str)
        if len(cand) == len(query_id):
            pred_sub = cand
            sc = np.asarray(q_prob, dtype=float)
            score_sub = sc if sc.shape[0] == len(cand) else np.full(len(cand), np.nan)
    except Exception:
        pred_sub = None
    if pred_sub is None:
        ref_labels_ordered = pd.Series(adata_ref.obs["celltype"].astype(str).values, index=adata_ref.obs_names.astype(str))
        labels = ref_labels_ordered.reindex(pd.Index(ref_id)).fillna(ref_labels_ordered.iloc[0]).to_numpy()
        pred_sub, score_sub = fallback_embedding_knn(np.asarray(ref_embedding), np.asarray(query_embedding), labels, args.knn_k)

    # Align Concerto's (possibly cell-filtered) predictions back to the full query id list.
    ps = pd.Series(np.asarray(pred_sub).astype(str), index=query_id)
    ps = ps[~ps.index.duplicated()]
    ss = pd.Series(np.asarray(score_sub, dtype=float), index=query_id)
    ss = ss[~ss.index.duplicated()]
    idx = pd.Index(np.asarray(ids_query).astype(str))
    pred = ps.reindex(idx).fillna("unassigned").to_numpy().astype(str)
    score = ss.reindex(idx).to_numpy(dtype=float)
    return pred, score


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
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--embedding_dim", type=int, default=128)
    ap.add_argument("--pretrain_epochs", type=int, default=20)
    ap.add_argument("--fine_tune_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=0.0001)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()
    method = "label_transfer_Concerto"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cf = import_concerto()
        X1, X2, ids1, ids2, feats, mode = load_common(args)
        lab1_df = read_labels(args.labels1, args.label_id_col, args.label_col)
        lab2_df = read_labels(args.labels2, args.label_id_col, args.label_col)
        y1, _ = align_label_array(ids1, lab1_df); y2, _ = align_label_array(ids2, lab2_df)
        tmp = tempfile.mkdtemp(prefix="concerto_label_transfer_")
        pred12, score12 = run_direction(cf, X1, X2, ids1, ids2, feats, y1, y2, "mod1_to_mod2", args, tmp)
        pred21, score21 = run_direction(cf, X2, X1, ids2, ids1, feats, y2, y1, "mod2_to_mod1", args, tmp)
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
