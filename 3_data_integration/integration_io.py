#!/usr/bin/env python3
from __future__ import annotations
import gzip, json, os, re, sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy import sparse


def log(msg: str) -> None:
    print(msg, flush=True)


def read_names(path: Optional[str], n: Optional[int] = None, prefix: str = "item") -> np.ndarray:
    if path is None or str(path) in {"", "nan", "None"} or not Path(path).exists():
        if n is None:
            return np.array([], dtype=str)
        return np.array([f"{prefix}_{i}" for i in range(n)], dtype=str)
    p = Path(path)
    if str(p).endswith('.gz'):
        df = pd.read_csv(p, sep='\t', header=None, compression='gzip')
    else:
        df = pd.read_csv(p, sep='\t', header=None)
    if df.shape[1] >= 2:
        vals = df.iloc[:,1].astype(str).to_numpy()
    else:
        vals = df.iloc[:,0].astype(str).to_numpy()
    return vals


def _load_npz(path: Path):
    try:
        return sp.load_npz(path).tocsr()
    except Exception:
        z = np.load(path, allow_pickle=True)
        if {"data","indices","indptr","shape"}.issubset(set(z.files)):
            return sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        if len(z.files) == 1:
            arr = z[z.files[0]]
            return sp.csr_matrix(arr)
        raise


def load_matrix(path: str, orientation: str = "cell_by_feature") -> sp.csr_matrix:
    p = Path(path)
    name = p.name.lower()
    if name.endswith('.mtx') or name.endswith('.mtx.gz'):
        X = sio.mmread(str(p)).tocsr()
    elif name.endswith('.npz'):
        X = _load_npz(p)
    elif name.endswith('.npy'):
        X = sp.csr_matrix(np.load(p, mmap_mode='r', allow_pickle=True))
    elif name.endswith('.csv') or name.endswith('.csv.gz'):
        X = sp.csr_matrix(pd.read_csv(p, index_col=None).values)
    else:
        raise ValueError(f"Unsupported matrix format: {path}")
    if orientation == "feature_by_cell":
        X = X.T.tocsr()
    return X.tocsr()


def load_labels(path: Optional[str], ids: np.ndarray, modality: str) -> pd.DataFrame:
    ids = ids.astype(str)
    if path is None or str(path) in {"", "nan", "None"} or not Path(str(path)).exists():
        return pd.DataFrame({"cell_id": ids, "cell_type": "unknown", "batch": modality})
    df = pd.read_csv(path)
    # flexible column names
    id_col = next((c for c in ["cell_id", "barcode", "cell", "id"] if c in df.columns), df.columns[0])
    ct_col = next((c for c in ["cell_type", "celltype", "label", "annotation"] if c in df.columns), None)
    batch_col = next((c for c in ["batch", "source", "donor", "sample", "orig.ident"] if c in df.columns), None)
    out = pd.DataFrame({"cell_id": ids})
    df2 = df.copy(); df2[id_col] = df2[id_col].astype(str)
    out = out.merge(df2, left_on="cell_id", right_on=id_col, how="left")
    out["cell_type"] = out[ct_col].astype(str) if ct_col else "unknown"
    out["cell_type"] = out["cell_type"].replace({"nan":"unknown", "None":"unknown", "NA":"unknown"}).fillna("unknown")
    out["batch"] = out[batch_col].astype(str) if batch_col else modality
    out["batch"] = out["batch"].replace({"nan":modality, "None":modality, "NA":modality}).fillna(modality)
    return out[["cell_id","cell_type","batch"]]


def make_unique(vals: Iterable[str]) -> np.ndarray:
    seen: Dict[str,int] = {}
    out = []
    for v in vals:
        s = str(v)
        if s not in seen:
            seen[s] = 0; out.append(s)
        else:
            seen[s] += 1; out.append(f"{s}__dup{seen[s]}")
    return np.array(out, dtype=str)


def normalize_total_log1p(X: sp.csr_matrix, scale: float = 1e4) -> sp.csr_matrix:
    X = X.tocsr().astype(np.float32)
    sums = np.asarray(X.sum(axis=1)).ravel()
    sums[sums <= 0] = 1.0
    D = sp.diags(scale / sums)
    Y = D @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr()


def align_features(X1, f1, X2, f2, mode="auto") -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, str]:
    f1 = make_unique(f1); f2 = make_unique(f2)
    idx1 = pd.Index(f1.astype(str)); idx2 = pd.Index(f2.astype(str))
    shared = idx1.intersection(idx2)
    if mode == "auto":
        # Use shared features when names overlap substantially or feature counts match; otherwise use sparse union.
        frac = len(shared) / max(1, min(len(idx1), len(idx2)))
        mode_use = "shared" if (frac >= 0.5 or X1.shape[1] == X2.shape[1]) else "union"
    else:
        mode_use = mode
    if mode_use == "shared":
        if len(shared) > 0:
            return X1[:, idx1.get_indexer(shared)].tocsr(), X2[:, idx2.get_indexer(shared)].tocsr(), shared.astype(str).to_numpy(), "shared_feature_names"
        n = min(X1.shape[1], X2.shape[1])
        feats = np.array([f"feature_{i:06d}" for i in range(n)], dtype=str)
        return X1[:, :n].tocsr(), X2[:, :n].tocsr(), feats, "shared_feature_index"
    # sparse union
    union = idx1.union(idx2).astype(str)
    map_union = {v:i for i,v in enumerate(union)}
    def expand(X, feats):
        cols = np.array([map_union[str(v)] for v in feats], dtype=np.int64)
        X = X.tocoo()
        return sp.csr_matrix((X.data, (X.row, cols[X.col])), shape=(X.shape[0], len(union)))
    return expand(X1, f1), expand(X2, f2), union.to_numpy(dtype=str), "union_feature_names"


def write_mtx_bundle(X: sp.csr_matrix, obs: pd.DataFrame, features: np.ndarray, outdir: Path, prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sio.mmwrite(str(outdir / f"{prefix}.mtx"), X.tocsr())
    obs.to_csv(outdir / f"{prefix}_obs.csv", index=False)
    pd.Series(obs["cell_id"].astype(str)).to_csv(outdir / f"{prefix}_barcodes.tsv", sep='\t', header=False, index=False)
    pd.Series(features.astype(str)).to_csv(outdir / f"{prefix}_features.tsv", sep='\t', header=False, index=False)
    sp.save_npz(outdir / f"{prefix}.npz", X.tocsr())


def load_embedding_csv(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_") or c.startswith("PC_") or c.startswith("X")]
    if not emb_cols:
        emb_cols = [c for c in df.columns if c not in {"cell_id","modality","cell_type","batch","batch_modality","dataset","method","reliable","pseudo_cluster"} and pd.api.types.is_numeric_dtype(df[c])]
    if not emb_cols:
        raise ValueError(f"No embedding columns found in {path}")
    return df, df[emb_cols].to_numpy(dtype=float)


def merge_labels(emb_df: pd.DataFrame, labels1: Optional[str], labels2: Optional[str]) -> pd.DataFrame:
    out = emb_df.copy()
    lab_frames = []
    if labels1 and Path(str(labels1)).exists():
        tmp = pd.read_csv(labels1); tmp["modality"] = "mod1"; lab_frames.append(tmp)
    if labels2 and Path(str(labels2)).exists():
        tmp = pd.read_csv(labels2); tmp["modality"] = "mod2"; lab_frames.append(tmp)
    if lab_frames:
        labs = pd.concat(lab_frames, ignore_index=True)
        id_col = next((c for c in ["cell_id","barcode","cell","id"] if c in labs.columns), labs.columns[0])
        ct_col = next((c for c in ["cell_type","celltype","label","annotation"] if c in labs.columns), None)
        batch_col = next((c for c in ["batch","source","donor","sample","orig.ident"] if c in labs.columns), None)
        keep = [id_col, "modality"]
        if ct_col: keep.append(ct_col)
        if batch_col: keep.append(batch_col)
        labs = labs[keep].copy(); labs[id_col] = labs[id_col].astype(str)
        out["cell_id"] = out["cell_id"].astype(str)
        out = out.merge(labs, left_on=["cell_id","modality"], right_on=[id_col,"modality"], how="left", suffixes=("","_lab"))
        if "cell_type" not in out.columns or out["cell_type"].isna().all():
            out["cell_type"] = out[ct_col] if ct_col else "unknown"
        if "batch" not in out.columns or out["batch"].isna().all():
            out["batch"] = out[batch_col] if batch_col else out["modality"]
    out["cell_type"] = out.get("cell_type", "unknown").astype(str).fillna("unknown")
    out["batch"] = out.get("batch", out["modality"]).astype(str).fillna(out["modality"])
    out["batch_modality"] = out["batch"].astype(str) + "__" + out["modality"].astype(str)
    return out
