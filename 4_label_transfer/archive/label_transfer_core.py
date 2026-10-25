#!/usr/bin/env python3
"""
Shared utilities for ambiguity-aware cross-modality label transfer.

The public entry points are:
  * run_gamma_label_transfer(args, backend)
  * run_embedding_knn_label_transfer(args)

Backends:
  * v8  : direct cross-modal joint-diffusion cost + Sinkhorn
  * v9  : structural entropic Gromov-Wasserstein cost + Sinkhorn
  * v10 : fused direct + structural cost; default alpha is set by wrapper

All gamma matrices are rectangular: rows are modality 1, columns are modality 2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Logging and small helpers
# -----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def mkdir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_int_list(x: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    return tuple(int(v.strip()) for v in str(x).split(",") if str(v).strip())


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return a / np.maximum(b, eps)


def row_median_norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    med = float(np.nanmedian(norms))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return (X / med).astype(np.float32)


def zscore_dense(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z.astype(np.float32)


# -----------------------------------------------------------------------------
# Matrix and metadata loading
# -----------------------------------------------------------------------------


def _read_tsv_names(path: Optional[str | Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t", header=None, dtype=str)
    if df.shape[1] >= 2:
        vals = df.iloc[:, 0].astype(str).to_numpy()
    else:
        vals = df.iloc[:, 0].astype(str).to_numpy()
    return vals


def auto_sidecars(matrix_path: str | Path) -> Tuple[Optional[Path], Optional[Path]]:
    p = Path(matrix_path)
    name = p.name
    if name.endswith(".mtx.gz"):
        stem = name[:-7]
    elif name.endswith(".mtx"):
        stem = name[:-4]
    else:
        stem = p.stem
    candidates_bc = [
        p.with_name(stem + "_barcodes.tsv"),
        p.with_name(stem + "_barcodes.tsv.gz"),
        p.with_name(stem + ".barcodes.tsv"),
        p.with_name(stem + ".barcodes.tsv.gz"),
    ]
    candidates_ft = [
        p.with_name(stem + "_features.tsv"),
        p.with_name(stem + "_features.tsv.gz"),
        p.with_name(stem + ".features.tsv"),
        p.with_name(stem + ".features.tsv.gz"),
    ]
    bc = next((x for x in candidates_bc if x.exists()), None)
    ft = next((x for x in candidates_ft if x.exists()), None)
    return bc, ft


def load_matrix(path: str | Path, barcodes: Optional[str | Path] = None,
                features: Optional[str | Path] = None,
                orientation: str = "cell_by_feature") -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Load a matrix and return (cells x features, cell_ids, feature_ids)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    lower = path.name.lower()
    if lower.endswith(".mtx") or lower.endswith(".mtx.gz"):
        X = sio.mmread(str(path))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()
        auto_bc, auto_ft = auto_sidecars(path)
        if barcodes is None:
            barcodes = auto_bc
        if features is None:
            features = auto_ft
    elif lower.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        keys = set(z.files)
        if {"data", "indices", "indptr", "shape"}.issubset(keys):
            shape = tuple(np.asarray(z["shape"], dtype=np.int64).tolist())
            data = z["data"]
            indices = z["indices"]
            indptr = z["indptr"]
            if len(indptr) == shape[0] + 1:
                X = sp.csr_matrix((data, indices, indptr), shape=shape)
            elif len(indptr) == shape[1] + 1:
                X = sp.csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                raise ValueError(f"Cannot infer CSR/CSC format for {path}")
        elif "X" in keys:
            X = z["X"]
            X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        elif "arr_0" in keys:
            X = z["arr_0"]
            X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        else:
            raise ValueError(f"Unsupported npz keys in {path}: {sorted(keys)}")
        if "row_names" in keys and barcodes is None:
            barcodes = z["row_names"].astype(str)
        if "col_names" in keys and features is None:
            features = z["col_names"].astype(str)
    elif lower.endswith(".npy"):
        X = np.load(path, allow_pickle=True)
        X = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    else:
        sep = "\t" if lower.endswith(".tsv") or lower.endswith(".tsv.gz") else ","
        X = pd.read_csv(path, header=None, sep=sep).to_numpy(dtype=np.float32)
        X = sp.csr_matrix(X)

    if orientation not in {"cell_by_feature", "feature_by_cell"}:
        raise ValueError("orientation must be 'cell_by_feature' or 'feature_by_cell'")
    if orientation == "feature_by_cell":
        X = X.T.tocsr()
        barcodes, features = features, barcodes

    if isinstance(barcodes, np.ndarray):
        cell_ids = barcodes.astype(str)
    else:
        cell_ids = _read_tsv_names(barcodes)
    if isinstance(features, np.ndarray):
        feature_ids = features.astype(str)
    else:
        feature_ids = _read_tsv_names(features)

    if cell_ids is None or len(cell_ids) != X.shape[0]:
        cell_ids = np.array([f"cell_{i}" for i in range(X.shape[0])], dtype=str)
    if feature_ids is None or len(feature_ids) != X.shape[1]:
        feature_ids = np.array([f"feature_{i}" for i in range(X.shape[1])], dtype=str)

    X = X.astype(np.float32).tocsr()
    return X, cell_ids, feature_ids


def preprocess_matrix(X: sp.csr_matrix, do_log1p: bool = True,
                      do_row_l1: bool = True) -> sp.csr_matrix:
    X = X.astype(np.float32).tocsr(copy=True)
    if X.nnz:
        X.data[~np.isfinite(X.data)] = 0.0
        X.data[X.data < 0] = 0.0
    if do_log1p:
        X.data = np.log1p(X.data)
    if do_row_l1:
        row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        inv = np.zeros_like(row_sums)
        good = row_sums > 0
        inv[good] = 1.0 / row_sums[good]
        X = sp.diags(inv) @ X
        X = X.tocsr()
    return X.astype(np.float32)


def svd_embed(X: sp.csr_matrix, n_components: int, seed: int, name: str) -> np.ndarray:
    n_components = int(max(1, n_components))
    n_comp = min(n_components, max(1, X.shape[0] - 1), max(1, X.shape[1] - 1))
    if n_comp <= 0:
        arr = X.toarray().astype(np.float32)
        return zscore_dense(arr)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Z = svd.fit_transform(X).astype(np.float32)
    var = float(np.nansum(getattr(svd, "explained_variance_ratio_", np.array([np.nan]))))
    log(f"[SVD] {name}: {X.shape} -> {Z.shape}; explained_var={var:.3f}")
    return zscore_dense(Z)


def prepare_input_embeddings(
    x1: str | Path,
    x2: str | Path,
    barcodes1: Optional[str | Path],
    barcodes2: Optional[str | Path],
    orientation1: str,
    orientation2: str,
    pca_dim: int,
    seed: int,
    log1p: bool,
    row_l1_norm: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X1, ids1, feats1 = load_matrix(x1, barcodes1, None, orientation1)
    X2, ids2, feats2 = load_matrix(x2, barcodes2, None, orientation2)
    log(f"[LOAD] mod1: {X1.shape[0]} cells x {X1.shape[1]} features")
    log(f"[LOAD] mod2: {X2.shape[0]} cells x {X2.shape[1]} features")
    X1 = preprocess_matrix(X1, do_log1p=log1p, do_row_l1=row_l1_norm)
    X2 = preprocess_matrix(X2, do_log1p=log1p, do_row_l1=row_l1_norm)

    if X1.shape[1] == X2.shape[1]:
        log("[EMBED] shared feature count detected; using combined SVD for a shared coordinate system")
        X_all = sp.vstack([X1, X2], format="csr")
        Z_all = svd_embed(X_all, pca_dim, seed, "combined")
        Z1 = Z_all[:X1.shape[0]]
        Z2 = Z_all[X1.shape[0]:]
    else:
        log("[EMBED] feature counts differ; using per-modality SVD then z-score alignment")
        Z1 = svd_embed(X1, pca_dim, seed, "mod1")
        Z2 = svd_embed(X2, pca_dim, seed + 1, "mod2")
        d = min(Z1.shape[1], Z2.shape[1])
        Z1, Z2 = Z1[:, :d], Z2[:, :d]
    return Z1.astype(np.float32), Z2.astype(np.float32), ids1.astype(str), ids2.astype(str)


def load_integrated_embeddings(path: str | Path, modality1: str = "mod1",
                               modality2: str = "mod2") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "cell_id" not in df.columns or "modality" not in df.columns:
        raise ValueError(f"{path} must contain columns: cell_id, modality")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        # Fall back to all numeric columns except known metadata.
        metadata = {"cell_id", "modality", "reliable", "pseudo_cluster", "batch", "cell_type", "source"}
        emb_cols = [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]
    if not emb_cols:
        raise ValueError(f"No embedding columns found in {path}")
    d1 = df[df["modality"].astype(str) == modality1].copy()
    d2 = df[df["modality"].astype(str) == modality2].copy()
    if d1.empty or d2.empty:
        raise ValueError(f"Could not split embeddings by modality={modality1}/{modality2}; observed {df['modality'].unique()}")
    Z1 = d1[emb_cols].to_numpy(dtype=np.float32)
    Z2 = d2[emb_cols].to_numpy(dtype=np.float32)
    return zscore_dense(Z1), zscore_dense(Z2), d1["cell_id"].astype(str).to_numpy(), d2["cell_id"].astype(str).to_numpy()


# -----------------------------------------------------------------------------
# Diffusion coordinates and OT/GW gamma inference
# -----------------------------------------------------------------------------


def joint_diffusion_coordinates(Z1: np.ndarray, Z2: np.ndarray, knn_k: int,
                                n_components: int, multiscale_times: Sequence[int],
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([Z1, Z2]).astype(np.float32)
    n = X.shape[0]
    if n < 3:
        return row_median_norm(Z1), row_median_norm(Z2)
    k = int(min(max(2, knn_k + 1), n))
    log(f"[DIFFUSION] building joint kNN graph: n={n}, k={k - 1}")
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X, return_distance=True)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    kth = dist[:, -1]
    sigma_global = float(np.nanmedian(kth[kth > 0])) if np.any(kth > 0) else 1.0
    if not np.isfinite(sigma_global) or sigma_global <= 0:
        sigma_global = 1.0
    sigma = np.maximum(kth, sigma_global * 1e-3)
    for i in range(n):
        for d, j in zip(dist[i, 1:], ind[i, 1:]):
            if i == j:
                continue
            denom = sigma[i] * sigma[j] + 1e-12
            w = math.exp(-float(d * d) / denom)
            rows.append(i); cols.append(int(j)); vals.append(w)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    W = W.maximum(W.T)
    W = W + sp.eye(n, dtype=np.float32) * 1e-6
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg <= 0] = 1.0
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    S = (D_inv_sqrt @ W @ D_inv_sqrt).astype(np.float32)
    n_components = int(min(max(1, n_components), n - 2))
    log(f"[DIFFUSION] eigensolve components={n_components}")
    try:
        from scipy.sparse.linalg import eigsh
        evals, evecs = eigsh(S, k=n_components + 1, which="LA", tol=1e-4, maxiter=max(1000, 20 * n))
    except Exception as exc:
        log(f"[DIFFUSION] sparse eigensolve failed ({exc}); falling back to dense eigh")
        Sd = S.toarray()
        evals_all, evecs_all = np.linalg.eigh(Sd)
        order = np.argsort(evals_all)[::-1][:n_components + 1]
        evals, evecs = evals_all[order], evecs_all[:, order]
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float32)
    evecs = np.asarray(evecs[:, order], dtype=np.float32)
    # Drop the top/trivial component.
    evals = np.clip(evals[1:], 0.0, 1.0)
    evecs = evecs[:, 1:]
    blocks = []
    for t in multiscale_times:
        blocks.append((evecs * (evals[None, :] ** float(t))).astype(np.float32))
    Psi = np.concatenate(blocks, axis=1) if blocks else evecs
    Psi[~np.isfinite(Psi)] = 0.0
    return row_median_norm(Psi[:Z1.shape[0]]), row_median_norm(Psi[Z1.shape[0]:])


def sinkhorn_from_cost(cost: np.ndarray, epsilon: float = 0.1, n_iter: int = 500,
                       verbose: bool = False) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be 2D")
    n1, n2 = cost.shape
    if n1 == 0 or n2 == 0:
        raise ValueError("cost matrix has zero-sized dimension")
    cost = cost - np.nanmin(cost)
    med = float(np.nanmedian(cost))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    K = np.exp(-(cost / med) / max(float(epsilon), 1e-6))
    K[~np.isfinite(K)] = 0.0
    a = np.full(n1, 1.0 / n1, dtype=np.float64)
    b = np.full(n2, 1.0 / n2, dtype=np.float64)
    u = np.ones(n1, dtype=np.float64)
    v = np.ones(n2, dtype=np.float64)
    for it in range(int(max(1, n_iter))):
        Kv = K @ v
        Kv[Kv < 1e-300] = 1e-300
        u = a / Kv
        KTu = K.T @ u
        KTu[KTu < 1e-300] = 1e-300
        v = b / KTu
    gamma = u[:, None] * K * v[None, :]
    s = float(gamma.sum())
    if s > 0:
        gamma /= s
    return gamma.astype(np.float32)


def compute_gamma_v8(Z1: np.ndarray, Z2: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float]]:
    times = parse_int_list(args.multiscale_times)
    D1, D2 = joint_diffusion_coordinates(
        Z1, Z2,
        knn_k=args.diffusion_knn_k,
        n_components=min(args.diffusion_components, max(1, min(Z1.shape[0] + Z2.shape[0] - 2, Z1.shape[1] if Z1.ndim == 2 else 1))),
        multiscale_times=times,
        seed=args.seed,
    )
    C_diff = cdist(D1, D2, metric="euclidean").astype(np.float64)
    med = np.nanmedian(C_diff)
    if np.isfinite(med) and med > 0:
        C_diff /= med
    direct_w = float(getattr(args, "direct_feature_weight", 0.0))
    if direct_w > 0:
        d = min(Z1.shape[1], Z2.shape[1])
        C_feat = cdist(row_median_norm(Z1[:, :d]), row_median_norm(Z2[:, :d]), metric="euclidean").astype(np.float64)
        med2 = np.nanmedian(C_feat)
        if np.isfinite(med2) and med2 > 0:
            C_feat /= med2
        cost = C_diff + direct_w * C_feat
    else:
        cost = C_diff
    G = sinkhorn_from_cost(cost, args.epsilon, args.sinkhorn_iters)
    return G, {"direct_diffusion_cost_median": float(np.nanmedian(C_diff)), "direct_feature_weight": direct_w}


def _normalised_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    D = cdist(X, X, metric=metric).astype(np.float64)
    if metric == "cosine":
        D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
        D = np.clip(D, 0.0, 2.0)
    D -= D.min()
    med = float(np.nanmedian(D))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return D / med


def compute_gamma_v9(Z1: np.ndarray, Z2: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float]]:
    """Dense entropic GW for rectangular matrices.

    This is designed for label-transfer benchmarking sizes. For very large full atlases,
    use the original disk-streaming v9 script or increase compute resources.
    """
    n1, n2 = Z1.shape[0], Z2.shape[0]
    max_dense = int(getattr(args, "max_dense_gw_cells", 20000))
    if max(n1, n2) > max_dense:
        raise MemoryError(
            f"v9 dense GW requested for n1={n1}, n2={n2}, larger than --max_dense_gw_cells={max_dense}. "
            "Use a subsample, increase the guard, or adapt the original disk-streaming v9 implementation."
        )
    times = parse_int_list(args.multiscale_times)
    D1, D2 = joint_diffusion_coordinates(
        Z1, Z2,
        knn_k=args.diffusion_knn_k,
        n_components=args.diffusion_components,
        multiscale_times=times,
        seed=args.seed,
    )
    log("[GW] computing within-modality distance matrices")
    Dx_diff = _normalised_distances(D1, "euclidean")
    Dy_diff = _normalised_distances(D2, "euclidean")
    Dx_feat = _normalised_distances(row_median_norm(Z1), "cosine")
    Dy_feat = _normalised_distances(row_median_norm(Z2), "cosine")
    mu = np.full(n1, 1.0 / n1, dtype=np.float64)
    nu = np.full(n2, 1.0 / n2, dtype=np.float64)
    gamma = np.outer(mu, nu).astype(np.float64)
    alpha = float(getattr(args, "gw_alpha", 1.0))
    beta = float(getattr(args, "gw_beta", 0.25))
    rounds = int(max(1, getattr(args, "gw_refine_rounds", 5)))
    log(f"[GW] rounds={rounds}, alpha={alpha}, beta={beta}")
    const_diff = ((Dx_diff ** 2) @ mu)[:, None] + ((Dy_diff ** 2) @ nu)[None, :]
    const_feat = ((Dx_feat ** 2) @ mu)[:, None] + ((Dy_feat ** 2) @ nu)[None, :]
    for r in range(rounds):
        log(f"[GW] round {r + 1}/{rounds}")
        C_diff = const_diff - 2.0 * (Dx_diff @ gamma @ Dy_diff.T)
        C_feat = const_feat - 2.0 * (Dx_feat @ gamma @ Dy_feat.T)
        C_diff -= C_diff.min(); C_feat -= C_feat.min()
        md = np.nanmedian(C_diff); mf = np.nanmedian(C_feat)
        if np.isfinite(md) and md > 0: C_diff /= md
        if np.isfinite(mf) and mf > 0: C_feat /= mf
        cost = alpha * C_diff + beta * C_feat
        gamma = sinkhorn_from_cost(cost, args.epsilon, args.sinkhorn_iters).astype(np.float64)
    return gamma.astype(np.float32), {"gw_alpha": alpha, "gw_beta": beta, "gw_refine_rounds": rounds}


def compute_gamma_v10(Z1: np.ndarray, Z2: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fused direct + structural gamma. Default alpha is set to 1.0 by the wrapper."""
    alpha = float(np.clip(getattr(args, "v10_alpha", 1.0), 0.0, 1.0))
    times = parse_int_list(args.multiscale_times)
    D1, D2 = joint_diffusion_coordinates(
        Z1, Z2,
        knn_k=args.diffusion_knn_k,
        n_components=args.diffusion_components,
        multiscale_times=times,
        seed=args.seed,
    )
    C_dir = cdist(D1, D2, metric="euclidean").astype(np.float64)
    md = np.nanmedian(C_dir)
    if np.isfinite(md) and md > 0:
        C_dir /= md
    if alpha >= 1.0 - 1e-12:
        log("[V10] alpha=1.0; using direct term only, equivalent to pure v8 cost inside the v10 wrapper")
        return sinkhorn_from_cost(C_dir, args.epsilon, args.sinkhorn_iters), {"v10_alpha": alpha, "v10_skipped_gw_because_alpha_1": 1.0}
    n1, n2 = D1.shape[0], D2.shape[0]
    Dx = _normalised_distances(D1, "euclidean")
    Dy = _normalised_distances(D2, "euclidean")
    mu = np.full(n1, 1.0 / n1, dtype=np.float64)
    nu = np.full(n2, 1.0 / n2, dtype=np.float64)
    const = ((Dx ** 2) @ mu)[:, None] + ((Dy ** 2) @ nu)[None, :]
    gamma = np.outer(mu, nu).astype(np.float64)
    rounds = int(max(1, getattr(args, "gw_refine_rounds", 5)))
    for r in range(rounds):
        log(f"[V10] fused round {r + 1}/{rounds}, alpha={alpha}")
        C_gw = const - 2.0 * (Dx @ gamma @ Dy.T)
        C_gw -= C_gw.min()
        mg = np.nanmedian(C_gw)
        if np.isfinite(mg) and mg > 0:
            C_gw /= mg
        cost = alpha * C_dir + (1.0 - alpha) * C_gw
        gamma = sinkhorn_from_cost(cost, args.epsilon, args.sinkhorn_iters).astype(np.float64)
    return gamma.astype(np.float32), {"v10_alpha": alpha, "gw_refine_rounds": rounds}


# -----------------------------------------------------------------------------
# Label loading, transfer, ambiguity gating, metrics
# -----------------------------------------------------------------------------


def read_labels(path: str | Path, id_col: str = "cell_id", label_col: str = "cell_type",
                batch_col: str = "batch") -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        # Try common alternatives; otherwise use first column.
        for cand in ["cell_id", "barcode", "barcodes", "cell", "index", "id"]:
            if cand in df.columns:
                id_col = cand
                break
        else:
            id_col = df.columns[0]
    if label_col not in df.columns:
        for cand in ["cell_type", "label", "labels", "annotation", "celltype", "cell_type_label"]:
            if cand in df.columns:
                label_col = cand
                break
        else:
            raise ValueError(f"Could not find label column in {path}; requested {label_col}; columns={list(df.columns)}")
    out = pd.DataFrame({
        "cell_id": df[id_col].astype(str),
        "true_label": df[label_col].astype(str),
    })
    if batch_col in df.columns:
        out["batch"] = df[batch_col].astype(str)
    return out.drop_duplicates("cell_id")


def align_label_array(cell_ids: np.ndarray, labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mp = dict(zip(labels_df["cell_id"].astype(str), labels_df["true_label"].astype(str)))
    vals = np.array([mp.get(str(c), None) for c in cell_ids], dtype=object)
    mask = np.array([v is not None and str(v).lower() not in {"nan", "none", ""} for v in vals], dtype=bool)
    return vals, mask


def label_probability_from_weights(weights: np.ndarray, source_labels: np.ndarray,
                                   source_label_mask: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights, dtype=np.float64)
    good = source_label_mask & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return {}
    total = float(w[good].sum())
    if total <= 0:
        return {}
    probs: Dict[str, float] = defaultdict(float)
    for lab, val in zip(source_labels[good], w[good]):
        probs[str(lab)] += float(val) / total
    return dict(probs)


def topk_from_probs(probs: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def entropy_normalized(p: np.ndarray) -> Tuple[float, float]:
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return float("nan"), float("nan")
    p = p / p.sum()
    h = -float(np.sum(p * np.log(p + 1e-300)))
    h_norm = h / math.log(max(2, p.size))
    return h, h_norm



def _source_axis_for_transfer(gamma: np.ndarray, direction: str) -> np.ndarray:
    """Return source x target normalized weights for top-match label transfer.

    The correspondence matrix Gamma always has rows=modality 1 and columns=modality 2.
    For mod1 -> mod2 transfer, each reference/source cell is a row and chooses a
    target column. For mod2 -> mod1 transfer, each reference/source cell is a
    column and chooses a target row, so we transpose Gamma first.

    This is intentionally source/reference oriented: label transfer follows the
    best correspondence of each reference cell, rather than smoothing labels over
    all target cells in an embedding space.
    """
    G = np.asarray(gamma, dtype=np.float64)
    if direction == "mod1_to_mod2":
        W = G.copy()      # source mod1 x target mod2
    elif direction == "mod2_to_mod1":
        W = G.T.copy()    # source mod2 x target mod1
    else:
        raise ValueError(direction)
    denom = W.sum(axis=1, keepdims=True)
    return safe_divide(W, denom)


def _target_axis_for_transfer(gamma: np.ndarray, direction: str) -> np.ndarray:
    """Return target x source normalized weights, used only for target summaries."""
    G = np.asarray(gamma, dtype=np.float64)
    if direction == "mod1_to_mod2":
        W = G.T.copy()
    elif direction == "mod2_to_mod1":
        W = G.copy()
    else:
        raise ValueError(direction)
    denom = W.sum(axis=1, keepdims=True)
    return safe_divide(W, denom)


def source_reciprocal_flags(gamma: np.ndarray, direction: str, source_top_target: np.ndarray) -> np.ndarray:
    """For each reference/source cell, check whether the top target also points back."""
    G = np.asarray(gamma)
    if direction == "mod1_to_mod2":
        # source i is mod1 row, target j is mod2 column
        target_best_source = np.argmax(G, axis=0)  # per mod2 target, best mod1 source
        return target_best_source[source_top_target] == np.arange(G.shape[0])
    elif direction == "mod2_to_mod1":
        # source j is mod2 column, target i is mod1 row
        target_best_source = np.argmax(G, axis=1)  # per mod1 target, best mod2 source
        return target_best_source[source_top_target] == np.arange(G.shape[1])
    raise ValueError(direction)


def source_collision_info(source_top_target: np.ndarray, source_top_value: np.ndarray, n_targets: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve multiple reference cells claiming the same target cell.

    Returns per-source claim_count_for_chosen_target, winner_source_for_chosen_target,
    and source_is_collision_winner. Only the strongest source, by normalized gamma
    top-match probability, wins a collision. Lower-confidence claimants are kept
    as ambiguous for a later correspondence refinement round.
    """
    source_top_target = np.asarray(source_top_target, dtype=int)
    source_top_value = np.asarray(source_top_value, dtype=float)
    n_sources = len(source_top_target)
    counts_by_target = np.bincount(source_top_target, minlength=n_targets)
    winner_by_target = np.full(n_targets, -1, dtype=int)
    for t in range(n_targets):
        src = np.where(source_top_target == t)[0]
        if src.size:
            # Strongest claimant wins; stable tie break by smaller source index.
            local = src[np.lexsort((src, -source_top_value[src]))][0]
            winner_by_target[t] = int(local)
    claim_count = counts_by_target[source_top_target].astype(int)
    winner_for_source = winner_by_target[source_top_target].astype(int)
    source_is_winner = winner_for_source == np.arange(n_sources)
    return claim_count, winner_for_source, source_is_winner


def _topk_indices_and_values(w: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if w.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    k = int(max(1, min(k, w.size)))
    # argpartition is faster for large rectangular gamma; final order is sorted.
    idx = np.argpartition(-w, kth=k-1)[:k]
    idx = idx[np.argsort(-w[idx])]
    return idx.astype(int), w[idx].astype(float)


def transfer_one_direction(
    gamma: np.ndarray,
    ids1: np.ndarray,
    ids2: np.ndarray,
    labels1_df: pd.DataFrame,
    labels2_df: pd.DataFrame,
    direction: str,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Gamma top-match label transfer.

    This is deliberately NOT a weighted embedding-neighbour classifier.
    For every reference/source cell, we identify its highest-gamma target cell.
    If the source-side gamma vector is ambiguous, non-reciprocal, or loses a
    target collision, we do not transfer its label. The source cell and its
    candidate targets are written to next-round files for re-inference.
    """
    if direction == "mod1_to_mod2":
        source_ids, target_ids = ids1, ids2
        source_df, target_df = labels1_df, labels2_df
    elif direction == "mod2_to_mod1":
        source_ids, target_ids = ids2, ids1
        source_df, target_df = labels2_df, labels1_df
    else:
        raise ValueError(direction)

    source_labels, source_mask = align_label_array(source_ids, source_df)
    target_true, target_mask = align_label_array(target_ids, target_df)

    W_source = _source_axis_for_transfer(gamma, direction)  # source x target
    n_sources, n_targets = W_source.shape
    if n_sources != len(source_ids) or n_targets != len(target_ids):
        raise ValueError(
            f"Internal shape mismatch for {direction}: W_source={W_source.shape}, "
            f"source_ids={len(source_ids)}, target_ids={len(target_ids)}"
        )

    source_top_target = np.argmax(W_source, axis=1).astype(int)
    source_top_value = W_source[np.arange(n_sources), source_top_target].astype(float)
    rec = source_reciprocal_flags(gamma, direction, source_top_target)
    claim_count, claim_winner, source_is_winner = source_collision_info(source_top_target, source_top_value, n_targets)

    records: List[Dict[str, object]] = []
    prob_records: List[Dict[str, object]] = []
    next_k = int(max(1, getattr(args, "next_round_topk", 20)))
    ratio_max = float(getattr(args, "max_top2_to_top1_ratio", 0.95))

    for s_idx, sid in enumerate(source_ids):
        w = W_source[s_idx]
        top_idx, top_val = _topk_indices_and_values(w, max(3, next_k))
        top_target_idx = int(top_idx[0]) if top_idx.size else -1
        top1_match_prob = float(top_val[0]) if top_val.size else float("nan")
        top2_match_prob = float(top_val[1]) if top_val.size > 1 else 0.0
        top3_match_prob = float(top_val[2]) if top_val.size > 2 else 0.0
        match_gap = top1_match_prob - top2_match_prob
        top2_ratio = float(top2_match_prob / max(top1_match_prob, 1e-300)) if np.isfinite(top1_match_prob) else float("nan")
        h, h_norm = entropy_normalized(w)
        effective = float(math.exp(h)) if np.isfinite(h) else float("nan")
        source_label = source_labels[s_idx] if source_mask[s_idx] else None
        target_label = target_true[top_target_idx] if (top_target_idx >= 0 and target_mask[top_target_idx]) else None
        reciprocal = bool(rec[s_idx]) if s_idx < len(rec) else False
        collision_count = int(claim_count[s_idx]) if s_idx < len(claim_count) else 0
        collision_loser = bool(collision_count > 1 and not source_is_winner[s_idx]) if s_idx < len(source_is_winner) else False

        fail_reasons: List[str] = []
        if source_label is None or str(source_label).lower() in {"nan", "none", ""}:
            fail_reasons.append("missing_reference_label")
        if top_target_idx < 0:
            fail_reasons.append("no_target_match")
        if top1_match_prob < args.min_match_prob:
            fail_reasons.append("low_match_probability")
        if match_gap < args.min_match_gap:
            fail_reasons.append("low_match_gap")
        if np.isfinite(top2_ratio) and top2_ratio > ratio_max:
            fail_reasons.append("top2_too_close_to_top1")
        if np.isfinite(h_norm) and h_norm > args.max_entropy:
            fail_reasons.append("high_gamma_entropy")
        if args.require_reciprocal and not reciprocal:
            fail_reasons.append("non_reciprocal")
        if args.reject_collisions and collision_loser:
            fail_reasons.append("collision_loser")

        ambiguity_reasons = {
            "low_match_probability", "low_match_gap", "top2_too_close_to_top1",
            "high_gamma_entropy", "non_reciprocal", "collision_loser"
        }
        if not fail_reasons:
            status = "accepted"
        elif any(r in ambiguity_reasons for r in fail_reasons):
            status = "ambiguous"
        else:
            status = "unassigned"

        raw_pred_label = str(source_label) if source_label is not None else "unassigned"
        pred_label = raw_pred_label if status == "accepted" else "unassigned"
        raw_correct = bool(raw_pred_label == str(target_label)) if target_label is not None else None
        accepted_correct = bool(raw_correct) if status == "accepted" and target_label is not None else None
        exact_pair = bool(str(sid) == str(target_ids[top_target_idx])) if top_target_idx >= 0 else None
        accepted_exact_pair = bool(exact_pair) if status == "accepted" and exact_pair is not None else None
        score = float(
            max(0.0, top1_match_prob)
            * max(0.0, match_gap)
            * max(0.0, 1.0 - (h_norm if np.isfinite(h_norm) else 1.0))
            * (1.0 if reciprocal else 0.75)
            * (0.5 if collision_loser else 1.0)
        )

        candidate_ids = [str(target_ids[i]) for i in top_idx[:next_k]]
        candidate_probs = [float(v) for v in top_val[:next_k]]
        row = {
            "direction": direction,
            "source_cell_id": str(sid),
            "source_index": int(s_idx),
            "reference_cell_id": str(sid),
            "reference_index": int(s_idx),
            "reference_label": raw_pred_label,
            "matched_target_cell_id": str(target_ids[top_target_idx]) if top_target_idx >= 0 else "",
            "matched_target_index": int(top_target_idx),
            "target_cell_id": str(target_ids[top_target_idx]) if top_target_idx >= 0 else "",
            "target_index": int(top_target_idx),
            "n_targets_total": int(n_targets),
            "true_label": target_label,
            "predicted_label": pred_label,
            "raw_predicted_label": raw_pred_label,
            "raw_correct": raw_correct,
            "accepted_correct": accepted_correct,
            "raw_exact_pair_correct": exact_pair,
            "accepted_exact_pair_correct": accepted_exact_pair,
            "transfer_status": status,
            "failure_reasons": ";".join(sorted(set(fail_reasons))),
            "top_match_target_cell_id": str(target_ids[top_target_idx]) if top_target_idx >= 0 else "",
            "top_match_target_index": int(top_target_idx),
            "top1_match_probability": top1_match_prob,
            "top2_match_probability": top2_match_prob,
            "top3_match_probability": top3_match_prob,
            "top1_top2_match_gap": float(match_gap),
            "top2_to_top1_ratio": top2_ratio,
            "gamma_entropy": h,
            "gamma_entropy_normalized": h_norm,
            "effective_number_of_matches": effective,
            "reciprocal_match": reciprocal,
            "collision_count": collision_count,
            "collision_winner_source_index": int(claim_winner[s_idx]) if s_idx < len(claim_winner) else -1,
            "collision_loser": collision_loser,
            "acceptance_score": score,
            "next_round_candidate_target_cell_ids": ";".join(candidate_ids),
            "next_round_candidate_target_probabilities": ";".join(f"{x:.8g}" for x in candidate_probs),
        }
        records.append(row)

        # This is a one-cell reference transfer, not a smoothed classifier.  The
        # probability table is kept for compatibility with downstream code.
        prob_records.append({
            "direction": direction,
            "source_cell_id": str(sid),
            "matched_target_cell_id": row["matched_target_cell_id"],
            "predicted_label": raw_pred_label,
            "probability": 1.0 if source_label is not None else 0.0,
        })

    res = pd.DataFrame(records)
    prob_df = pd.DataFrame(prob_records)
    metrics = compute_metrics_from_results(res, direction=direction)
    return res, prob_df, metrics

def _metric_or_nan(fn, y_true: Sequence[str], y_pred: Sequence[str], **kwargs) -> float:
    try:
        if len(y_true) == 0:
            return float("nan")
        return float(fn(y_true, y_pred, **kwargs))
    except Exception:
        return float("nan")


def topk_accuracy_from_df(df: pd.DataFrame, k: int = 3) -> float:
    if df.empty or "true_label" not in df.columns:
        return float("nan")
    vals = []
    for _, r in df.iterrows():
        truth = r.get("true_label", None)
        if truth is None or str(truth).lower() in {"nan", "none", ""}:
            continue
        pred_labels = [str(r.get("raw_predicted_label", ""))]
        if k >= 2:
            pred_labels.append(str(r.get("label_top2", "")))
        if k >= 3:
            pred_labels.append(str(r.get("label_top3", "")))
        vals.append(str(truth) in pred_labels[:k])
    return float(np.mean(vals)) if vals else float("nan")



def compute_metrics_from_results(df: pd.DataFrame, direction: str) -> Dict[str, float]:
    """Compute standard label-transfer metrics from source→target gamma matches."""
    truth_mask = df["true_label"].notna() & ~df["true_label"].astype(str).str.lower().isin(["nan", "none", ""])
    dtrue = df.loc[truth_mask].copy()
    accepted = dtrue[dtrue["transfer_status"] == "accepted"].copy()
    y_true_all = dtrue["true_label"].astype(str).to_numpy()
    y_raw_all = dtrue["raw_predicted_label"].astype(str).to_numpy()
    y_true_acc = accepted["true_label"].astype(str).to_numpy()
    y_pred_acc = accepted["raw_predicted_label"].astype(str).to_numpy()

    raw_unique_targets = int(df["matched_target_cell_id"].replace("", np.nan).nunique()) if "matched_target_cell_id" in df.columns else 0
    n_targets_total = int(pd.to_numeric(df.get("n_targets_total", pd.Series([raw_unique_targets])), errors="coerce").dropna().max()) if len(df) else 0
    n_targets_accepted = int(accepted["matched_target_cell_id"].replace("", np.nan).nunique()) if not accepted.empty else 0
    exact_mask = dtrue["raw_exact_pair_correct"].notna() if "raw_exact_pair_correct" in dtrue.columns else pd.Series(False, index=dtrue.index)
    exact_acc_mask = accepted["accepted_exact_pair_correct"].notna() if "accepted_exact_pair_correct" in accepted.columns else pd.Series(False, index=accepted.index)

    coverage_value = float(len(accepted) / len(dtrue)) if len(dtrue) else float("nan")
    out: Dict[str, float] = {
        "n_reference_cells": float(len(df)),
        "n_reference_cells_with_target_truth": float(len(dtrue)),
        "n_unique_targets_claimed_raw": float(raw_unique_targets),
        "n_targets_total": float(n_targets_total),
        "n_unique_targets_accepted": float(n_targets_accepted),
        "n_accepted": float((df["transfer_status"] == "accepted").sum()),
        "n_ambiguous": float((df["transfer_status"] == "ambiguous").sum()),
        "n_suspicious": float((df["transfer_status"] == "ambiguous").sum()),  # compatibility name
        "n_unassigned": float((df["transfer_status"] == "unassigned").sum()),
        "coverage": coverage_value,
        "unclassified_rate": float(1.0 - coverage_value) if np.isfinite(coverage_value) else float("nan"),
        "target_coverage": float(n_targets_accepted / max(1, n_targets_total)) if len(df) else float("nan"),
        "ambiguous_fraction": float((df["transfer_status"] == "ambiguous").mean()) if len(df) else float("nan"),
        "collision_loser_fraction": float(pd.to_numeric(df.get("collision_loser", pd.Series([], dtype=float)), errors="coerce").fillna(False).astype(bool).mean()) if len(df) else float("nan"),
        "raw_overall_accuracy": _metric_or_nan(accuracy_score, y_true_all, y_raw_all),
        "raw_balanced_accuracy": _metric_or_nan(balanced_accuracy_score, y_true_all, y_raw_all),
        "raw_macro_f1": _metric_or_nan(f1_score, y_true_all, y_raw_all, average="macro", zero_division=0),
        "raw_weighted_f1": _metric_or_nan(f1_score, y_true_all, y_raw_all, average="weighted", zero_division=0),
        "raw_cohen_kappa": _metric_or_nan(cohen_kappa_score, y_true_all, y_raw_all),
        "raw_mcc": _metric_or_nan(matthews_corrcoef, y_true_all, y_raw_all),
        "raw_top3_accuracy": topk_accuracy_from_df(dtrue, k=3),
        "accepted_overall_accuracy": _metric_or_nan(accuracy_score, y_true_acc, y_pred_acc),
        "accepted_balanced_accuracy": _metric_or_nan(balanced_accuracy_score, y_true_acc, y_pred_acc),
        "accepted_macro_f1": _metric_or_nan(f1_score, y_true_acc, y_pred_acc, average="macro", zero_division=0),
        "accepted_weighted_f1": _metric_or_nan(f1_score, y_true_acc, y_pred_acc, average="weighted", zero_division=0),
        "accepted_macro_precision": _metric_or_nan(precision_score, y_true_acc, y_pred_acc, average="macro", zero_division=0),
        "accepted_macro_recall": _metric_or_nan(recall_score, y_true_acc, y_pred_acc, average="macro", zero_division=0),
        "accepted_cohen_kappa": _metric_or_nan(cohen_kappa_score, y_true_acc, y_pred_acc),
        "accepted_mcc": _metric_or_nan(matthews_corrcoef, y_true_acc, y_pred_acc),
        "coverage_adjusted_accuracy": (
            _metric_or_nan(accuracy_score, y_true_acc, y_pred_acc) * coverage_value
            if np.isfinite(_metric_or_nan(accuracy_score, y_true_acc, y_pred_acc)) and np.isfinite(coverage_value) else float("nan")
        ),
        "coverage_adjusted_macro_f1": (
            _metric_or_nan(f1_score, y_true_acc, y_pred_acc, average="macro", zero_division=0) * coverage_value
            if np.isfinite(_metric_or_nan(f1_score, y_true_acc, y_pred_acc, average="macro", zero_division=0)) and np.isfinite(coverage_value) else float("nan")
        ),
        "raw_exact_pair_accuracy": float(dtrue.loc[exact_mask, "raw_exact_pair_correct"].astype(bool).mean()) if len(dtrue.loc[exact_mask]) else float("nan"),
        "accepted_exact_pair_accuracy": float(accepted.loc[exact_acc_mask, "accepted_exact_pair_correct"].astype(bool).mean()) if len(accepted.loc[exact_acc_mask]) else float("nan"),
        "mean_gamma_entropy_norm": float(np.nanmean(df["gamma_entropy_normalized"].to_numpy(dtype=float))) if len(df) else float("nan"),
        "mean_match_gap": float(np.nanmean(df["top1_top2_match_gap"].to_numpy(dtype=float))) if len(df) else float("nan"),
        "mean_top1_match_probability": float(np.nanmean(df["top1_match_probability"].to_numpy(dtype=float))) if len(df) else float("nan"),
    }
    # Accuracy among top-confidence cells at multiple coverage levels.
    curve = coverage_accuracy_curve(dtrue)
    for frac in [0.25, 0.50, 0.75, 1.00]:
        row = curve[np.isclose(curve["coverage"], frac)] if not curve.empty else pd.DataFrame()
        out[f"accuracy_at_{int(frac * 100)}pct_confidence"] = float(row["accuracy"].iloc[0]) if not row.empty else float("nan")
    return {f"{direction}__{k}": v for k, v in out.items()}

def coverage_accuracy_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["coverage", "n", "accuracy", "min_acceptance_score"])
    d = df.copy()
    d = d[d["true_label"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["coverage", "n", "accuracy", "min_acceptance_score"])
    d = d.sort_values("acceptance_score", ascending=False).reset_index(drop=True)
    n = len(d)
    rows = []
    for frac in [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]:
        k = max(1, int(round(frac * n)))
        sub = d.iloc[:k]
        acc = float((sub["true_label"].astype(str) == sub["raw_predicted_label"].astype(str)).mean())
        rows.append({
            "coverage": frac,
            "n": k,
            "accuracy": acc,
            "min_acceptance_score": float(sub["acceptance_score"].min()),
        })
    return pd.DataFrame(rows)



def _write_txt_lines(path: Path, values: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for v in values:
            if v is not None and str(v) != "":
                fh.write(str(v) + "\n")


def save_direction_outputs(outdir: Path, prefix: str, results: pd.DataFrame,
                           probs: pd.DataFrame) -> None:
    """Save gamma top-match label-transfer outputs and next-round inputs."""
    results.to_csv(outdir / f"{prefix}_reference_match_table.csv", index=False)
    # Compatibility name used by the previous package.
    results.to_csv(outdir / f"{prefix}_transferred_labels.csv", index=False)
    probs.to_csv(outdir / f"{prefix}_label_probabilities.csv", index=False)

    accepted = results[results["transfer_status"] == "accepted"].copy()
    ambiguous = results[results["transfer_status"] == "ambiguous"].copy()
    unassigned = results[results["transfer_status"] == "unassigned"].copy()
    accepted.to_csv(outdir / f"{prefix}_accepted_label_transfers.csv", index=False)
    accepted.to_csv(outdir / f"{prefix}_accepted_transfers.csv", index=False)  # compatibility
    ambiguous.to_csv(outdir / f"{prefix}_ambiguous_reference_cells.csv", index=False)
    ambiguous.to_csv(outdir / f"{prefix}_suspicious_transfers.csv", index=False)  # compatibility
    unassigned.to_csv(outdir / f"{prefix}_unassigned_reference_cells.csv", index=False)
    unassigned.to_csv(outdir / f"{prefix}_unassigned_cells.csv", index=False)  # compatibility

    # Target-level annotation table: one accepted reference label per target, choosing
    # the highest acceptance score if collisions were allowed.
    if not accepted.empty:
        target_ann = accepted.sort_values("acceptance_score", ascending=False).drop_duplicates("matched_target_cell_id")
        target_ann = target_ann[[
            "matched_target_cell_id", "matched_target_index", "raw_predicted_label", "true_label",
            "source_cell_id", "source_index", "top1_match_probability", "top1_top2_match_gap",
            "gamma_entropy_normalized", "reciprocal_match", "acceptance_score", "raw_correct",
            "raw_exact_pair_correct",
        ]].rename(columns={
            "matched_target_cell_id": "target_cell_id",
            "matched_target_index": "target_index",
            "raw_predicted_label": "transferred_label",
            "source_cell_id": "reference_cell_id",
            "source_index": "reference_index",
        })
    else:
        target_ann = pd.DataFrame(columns=[
            "target_cell_id", "target_index", "transferred_label", "true_label",
            "reference_cell_id", "reference_index", "top1_match_probability", "top1_top2_match_gap",
            "gamma_entropy_normalized", "reciprocal_match", "acceptance_score", "raw_correct",
            "raw_exact_pair_correct",
        ])
    target_ann.to_csv(outdir / f"{prefix}_target_annotations.csv", index=False)

    truth = results[results["true_label"].notna()].copy()
    curve = coverage_accuracy_curve(truth)
    curve.to_csv(outdir / f"{prefix}_coverage_accuracy_curve.csv", index=False)

    if not accepted.empty:
        labels = sorted(set(accepted["true_label"].dropna().astype(str)) | set(accepted["raw_predicted_label"].dropna().astype(str)))
        if labels:
            cm = confusion_matrix(accepted["true_label"].astype(str), accepted["raw_predicted_label"].astype(str), labels=labels)
            pd.DataFrame(cm, index=[f"true__{x}" for x in labels], columns=[f"pred__{x}" for x in labels]).to_csv(outdir / f"{prefix}_accepted_confusion_matrix.csv")

    # Files for the next ambiguity-resolution round.
    next_dir = outdir / "next_round_inputs"
    next_dir.mkdir(parents=True, exist_ok=True)
    _write_txt_lines(next_dir / f"{prefix}_ambiguous_reference_cell_ids.txt", ambiguous.get("source_cell_id", pd.Series(dtype=str)).astype(str).tolist())
    _write_txt_lines(next_dir / f"{prefix}_ambiguous_reference_indices.txt", ambiguous.get("source_index", pd.Series(dtype=int)).astype(str).tolist())

    cand_ids: List[str] = []
    if "next_round_candidate_target_cell_ids" in ambiguous.columns:
        for x in ambiguous["next_round_candidate_target_cell_ids"].dropna().astype(str):
            cand_ids.extend([v for v in x.split(";") if v])
    # Stable unique order.
    seen = set(); cand_unique = []
    for x in cand_ids:
        if x not in seen:
            seen.add(x); cand_unique.append(x)
    _write_txt_lines(next_dir / f"{prefix}_candidate_target_cell_ids_topk.txt", cand_unique)

    if not ambiguous.empty:
        ambiguous[[
            "source_cell_id", "source_index", "reference_label", "matched_target_cell_id",
            "matched_target_index", "failure_reasons", "top1_match_probability",
            "top2_match_probability", "top1_top2_match_gap", "top2_to_top1_ratio",
            "gamma_entropy_normalized", "reciprocal_match", "collision_loser",
            "next_round_candidate_target_cell_ids",
        ]].to_csv(next_dir / f"{prefix}_next_round_manifest.csv", index=False)
    else:
        pd.DataFrame().to_csv(next_dir / f"{prefix}_next_round_manifest.csv", index=False)



def write_standardized_reanalysis_predictions(results: pd.DataFrame, out_path: Path, dataset_id: str, method: str) -> None:
    """Write a compact per-transfer prediction table for future metric recomputation."""
    if results is None or results.empty:
        pd.DataFrame(columns=[
            "dataset_id", "method", "direction", "target_cell_id", "true_label",
            "predicted_label", "prediction_score", "transfer_status",
            "reference_cell_id", "reference_label", "matched_target_index", "source_index",
        ]).to_csv(out_path, index=False)
        return
    df = results.copy()
    out = pd.DataFrame({
        "dataset_id": dataset_id,
        "method": method,
        "direction": df.get("direction", pd.Series([None] * len(df))).astype(str),
        "target_cell_id": df.get("matched_target_cell_id", pd.Series([None] * len(df))).astype(str),
        "true_label": df.get("true_label", pd.Series([None] * len(df))).astype(str),
        "predicted_label": df.get("raw_predicted_label", df.get("reference_label", pd.Series([None] * len(df)))).astype(str),
        "prediction_score": pd.to_numeric(df.get("acceptance_score", df.get("top1_match_probability", pd.Series([np.nan] * len(df)))), errors="coerce"),
        "transfer_status": df.get("transfer_status", pd.Series(["accepted"] * len(df))).astype(str),
        "reference_cell_id": df.get("source_cell_id", pd.Series([None] * len(df))).astype(str),
        "reference_label": df.get("reference_label", pd.Series([None] * len(df))).astype(str),
        "matched_target_index": df.get("matched_target_index", pd.Series([np.nan] * len(df))),
        "source_index": df.get("source_index", pd.Series([np.nan] * len(df))),
    })
    # Preserve useful ambiguity/correspondence columns when present.
    for c in [
        "top1_match_probability", "top2_match_probability", "top1_top2_match_gap",
        "top2_to_top1_ratio", "gamma_entropy_normalized", "effective_number_of_matches",
        "reciprocal_match", "collision_count", "collision_loser", "failure_reasons",
        "raw_correct", "raw_exact_pair_correct",
    ]:
        if c in df.columns:
            out[c] = df[c].values
    out.to_csv(out_path, index=False)

# -----------------------------------------------------------------------------
# Main runners
# -----------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--x1", default=None, help="Modality-1 matrix, rows=cells by default")
    p.add_argument("--x2", default=None, help="Modality-2 matrix, rows=cells by default")
    p.add_argument("--barcodes1", default=None)
    p.add_argument("--barcodes2", default=None)
    p.add_argument("--embeddings_csv", default=None, help="Optional integrated embedding CSV with cell_id, modality, emb_* columns")
    p.add_argument("--modality1_name", default="mod1")
    p.add_argument("--modality2_name", default="mod2")
    p.add_argument("--labels1", required=True)
    p.add_argument("--labels2", required=True)
    p.add_argument("--label_id_col", default="cell_id")
    p.add_argument("--label_col", default="cell_type")
    p.add_argument("--batch_col", default="batch")
    p.add_argument("--orientation1", choices=["cell_by_feature", "feature_by_cell"], default="cell_by_feature")
    p.add_argument("--orientation2", choices=["cell_by_feature", "feature_by_cell"], default="cell_by_feature")
    p.add_argument("--transfer_direction", choices=["mod1_to_mod2", "mod2_to_mod1", "both"], default="both")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--diffusion_knn_k", type=int, default=30)
    p.add_argument("--diffusion_components", type=int, default=30)
    p.add_argument("--multiscale_times", default="1,2,4")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--sinkhorn_iters", type=int, default=500)
    p.add_argument("--gw_alpha", type=float, default=1.0)
    p.add_argument("--gw_beta", type=float, default=0.25)
    p.add_argument("--gw_refine_rounds", type=int, default=5)
    p.add_argument("--v10_alpha", type=float, default=1.0)
    p.add_argument("--direct_feature_weight", type=float, default=0.0)
    p.add_argument("--max_dense_gw_cells", type=int, default=20000)
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--no_row_l1_norm", action="store_true")
    p.add_argument("--min_label_prob", type=float, default=0.50, help="Compatibility only; gamma-top-match transfer does not use weighted label probabilities")
    p.add_argument("--min_label_margin", type=float, default=0.05, help="Compatibility only; gamma-top-match transfer does not use weighted label probabilities")
    p.add_argument("--min_match_prob", type=float, default=0.0)
    p.add_argument("--min_match_gap", type=float, default=0.0)
    p.add_argument("--max_entropy", type=float, default=0.90)
    p.add_argument("--max_top2_to_top1_ratio", type=float, default=0.95, help="Mark reference cell ambiguous when top2/top1 gamma ratio exceeds this value")
    p.add_argument("--next_round_topk", type=int, default=20, help="Number of candidate target cells saved for ambiguous references in next-round manifest")
    p.add_argument("--require_reciprocal", action="store_true")
    p.add_argument("--reject_collisions", action="store_true", default=True)
    p.add_argument("--allow_collisions", dest="reject_collisions", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p


def run_gamma_label_transfer(args: argparse.Namespace, backend: str) -> None:
    outdir = mkdir(args.out_dir)
    done = outdir / "label_transfer_metrics.json"
    if done.exists() and not args.force:
        log(f"[SKIP] {done} already exists; pass --force to rerun")
        return
    t0 = time.time()
    log("=" * 80)
    log(f"[RUN] dataset={args.dataset_id} backend={backend}")
    log(f"[OUT] {outdir}")
    log("=" * 80)

    if args.embeddings_csv:
        Z1, Z2, ids1, ids2 = load_integrated_embeddings(args.embeddings_csv, args.modality1_name, args.modality2_name)
        log(f"[EMBEDDINGS] loaded {args.embeddings_csv}: mod1={Z1.shape}, mod2={Z2.shape}")
    else:
        if not args.x1 or not args.x2:
            raise ValueError("Provide either --embeddings_csv or both --x1/--x2")
        Z1, Z2, ids1, ids2 = prepare_input_embeddings(
            args.x1, args.x2, args.barcodes1, args.barcodes2,
            args.orientation1, args.orientation2,
            args.pca_dim, args.seed,
            log1p=not args.no_log1p,
            row_l1_norm=not args.no_row_l1_norm,
        )
    np.save(outdir / "mod1_input_embedding.npy", Z1.astype(np.float32))
    np.save(outdir / "mod2_input_embedding.npy", Z2.astype(np.float32))
    pd.DataFrame({"cell_id": ids1}).to_csv(outdir / "mod1_cell_ids.csv", index=False)
    pd.DataFrame({"cell_id": ids2}).to_csv(outdir / "mod2_cell_ids.csv", index=False)

    if backend == "v8":
        gamma, gamma_info = compute_gamma_v8(Z1, Z2, args)
    elif backend == "v9":
        gamma, gamma_info = compute_gamma_v9(Z1, Z2, args)
    elif backend == "v10":
        gamma, gamma_info = compute_gamma_v10(Z1, Z2, args)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    log(f"[GAMMA] shape={gamma.shape}; sum={gamma.sum():.6f}; max={gamma.max():.6g}")
    np.save(outdir / "gamma.npy", gamma.astype(np.float32))
    # Compatibility names used by earlier scripts.
    np.save(outdir / "gamma_common.npy", gamma.astype(np.float32))
    row_sum = gamma.sum(axis=1, keepdims=True)
    col_sum = gamma.sum(axis=0, keepdims=True)
    np.save(outdir / "gamma_row_normalized.npy", safe_divide(gamma, row_sum).astype(np.float32))
    np.save(outdir / "gamma_col_normalized.npy", safe_divide(gamma, col_sum).astype(np.float32))
    np.savez_compressed(outdir / "gamma_metadata.npz", row_names=ids1.astype(str), col_names=ids2.astype(str))

    labels1_df = read_labels(args.labels1, args.label_id_col, args.label_col, args.batch_col)
    labels2_df = read_labels(args.labels2, args.label_id_col, args.label_col, args.batch_col)
    directions = ["mod1_to_mod2", "mod2_to_mod1"] if args.transfer_direction == "both" else [args.transfer_direction]
    all_results = []
    all_probs = []
    metrics: Dict[str, float | str | int] = {
        "dataset_id": args.dataset_id,
        "backend": backend,
        "runtime_seconds": float(time.time() - t0),
        "n_mod1": int(len(ids1)),
        "n_mod2": int(len(ids2)),
        **gamma_info,
    }
    for direction in directions:
        log(f"[TRANSFER] {direction}")
        res, probs, m = transfer_one_direction(gamma, ids1, ids2, labels1_df, labels2_df, direction, args)
        prefix = "mod1_to_mod2" if direction == "mod1_to_mod2" else "mod2_to_mod1"
        save_direction_outputs(outdir, prefix, res, probs)
        all_results.append(res)
        all_probs.append(probs)
        metrics.update(m)
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_probs_df = pd.concat(all_probs, ignore_index=True)
    all_results_df.to_csv(outdir / "transferred_labels.csv", index=False)
    all_probs_df.to_csv(outdir / "label_probabilities.csv", index=False)
    write_standardized_reanalysis_predictions(all_results_df, outdir / "all_predictions_standardized.csv", args.dataset_id, f"label_transfer_diffusion_{backend}" if backend != "v10" else "label_transfer_diffusion_v10_alpha1")
    # Mean bidirectional headline metrics when both directions are available.
    for k in [
        "coverage", "unclassified_rate", "coverage_adjusted_accuracy", "coverage_adjusted_macro_f1", "target_coverage", "ambiguous_fraction",
        "raw_overall_accuracy", "raw_balanced_accuracy", "raw_macro_f1", "raw_weighted_f1", "raw_cohen_kappa", "raw_mcc",
        "accepted_overall_accuracy", "accepted_balanced_accuracy", "accepted_macro_f1", "accepted_weighted_f1",
        "accepted_cohen_kappa", "accepted_mcc", "raw_exact_pair_accuracy", "accepted_exact_pair_accuracy",
    ]:
        vals = [metrics.get(f"{d}__{k}") for d in directions]
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        metrics[f"mean__{k}"] = float(np.mean(vals)) if vals else float("nan")
    metrics["runtime_seconds"] = float(time.time() - t0)
    metrics["status"] = "completed"
    metrics["message"] = "completed"
    with open(outdir / "label_transfer_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, allow_nan=True)
    with open(outdir / "method_status.json", "w") as fh:
        json.dump({"dataset_id": args.dataset_id, "method": metrics.get("backend", backend), "status": "completed", "message": "completed"}, fh, indent=2)
    pd.DataFrame([{"dataset_id": args.dataset_id, "method": metrics.get("backend", backend), "status": "completed", "message": "completed"}]).to_csv(outdir / "method_status.csv", index=False)
    headline_cols = ["backend", "method", "dataset_id", "status", "message"] + [k for k in metrics if str(k).startswith("mean__")]
    pd.DataFrame([{k: metrics.get(k, np.nan) for k in headline_cols}]).to_csv(outdir / "metrics_headline.csv", index=False)
    with open(outdir / "run_settings.json", "w") as fh:
        json.dump(vars(args) | {"backend": backend}, fh, indent=2, default=str)
    log(f"[SAVE] {outdir / 'label_transfer_metrics.json'}")
    log(f"[DONE] {args.dataset_id} backend={backend} runtime={time.time() - t0:.1f}s")


# -----------------------------------------------------------------------------
# Baseline label transfer from integrated embedding CSVs, e.g. Seurat/Harmony.
# -----------------------------------------------------------------------------


def gamma_from_knn_embeddings(Z_source: np.ndarray, Z_target: np.ndarray, k: int,
                              temperature: float = 1.0) -> np.ndarray:
    k = int(min(max(1, k), Z_source.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(Z_source)
    dist, ind = nn.kneighbors(Z_target, return_distance=True)
    positive = dist[dist > 0]
    sigma = float(np.nanmedian(positive)) if positive.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    weights = np.exp(-(dist ** 2) / (2.0 * (temperature * sigma) ** 2 + 1e-12))
    weights = safe_divide(weights, weights.sum(axis=1, keepdims=True))
    G = np.zeros((Z_source.shape[0], Z_target.shape[0]), dtype=np.float32)
    for t in range(Z_target.shape[0]):
        G[ind[t], t] = weights[t].astype(np.float32)
    # Scale columns to uniform target mass for compatibility with transfer code.
    G = G / max(1, Z_target.shape[0])
    return G


def run_embedding_knn_label_transfer(args: argparse.Namespace) -> None:
    outdir = mkdir(args.out_dir)
    if (outdir / "label_transfer_metrics.json").exists() and not args.force:
        log(f"[SKIP] {outdir / 'label_transfer_metrics.json'} exists; pass --force to rerun")
        return
    t0 = time.time()
    Z1, Z2, ids1, ids2 = load_integrated_embeddings(args.embeddings_csv, args.modality1_name, args.modality2_name)
    # Build a symmetric-ish rectangular gamma by averaging the two kNN directions.
    log(f"[KNN] embeddings: mod1={Z1.shape}, mod2={Z2.shape}, k={args.knn_k}")
    G12 = gamma_from_knn_embeddings(Z1, Z2, args.knn_k, args.knn_temperature)  # source mod1 -> target mod2: n1 x n2
    G21 = gamma_from_knn_embeddings(Z2, Z1, args.knn_k, args.knn_temperature).T  # n1 x n2
    gamma = (G12 + G21) / 2.0
    s = gamma.sum()
    if s > 0:
        gamma /= s
    np.save(outdir / "gamma.npy", gamma.astype(np.float32))
    np.save(outdir / "gamma_common.npy", gamma.astype(np.float32))
    np.savez_compressed(outdir / "gamma_metadata.npz", row_names=ids1.astype(str), col_names=ids2.astype(str))
    labels1_df = read_labels(args.labels1, args.label_id_col, args.label_col, args.batch_col)
    labels2_df = read_labels(args.labels2, args.label_id_col, args.label_col, args.batch_col)
    directions = ["mod1_to_mod2", "mod2_to_mod1"] if args.transfer_direction == "both" else [args.transfer_direction]
    all_results = []
    all_probs = []
    metrics: Dict[str, float | str | int] = {
        "dataset_id": args.dataset_id,
        "backend": args.method_name,
        "runtime_seconds": float(time.time() - t0),
        "n_mod1": int(len(ids1)),
        "n_mod2": int(len(ids2)),
        "knn_k": int(args.knn_k),
    }
    for direction in directions:
        res, probs, m = transfer_one_direction(gamma, ids1, ids2, labels1_df, labels2_df, direction, args)
        save_direction_outputs(outdir, direction, res, probs)
        all_results.append(res); all_probs.append(probs); metrics.update(m)
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_probs_df = pd.concat(all_probs, ignore_index=True)
    all_results_df.to_csv(outdir / "transferred_labels.csv", index=False)
    all_probs_df.to_csv(outdir / "label_probabilities.csv", index=False)
    write_standardized_reanalysis_predictions(all_results_df, outdir / "all_predictions_standardized.csv", args.dataset_id, args.method_name)
    for k in [
        "coverage", "unclassified_rate", "coverage_adjusted_accuracy", "coverage_adjusted_macro_f1", "target_coverage", "ambiguous_fraction",
        "raw_overall_accuracy", "raw_balanced_accuracy", "raw_macro_f1", "raw_weighted_f1", "raw_cohen_kappa", "raw_mcc",
        "accepted_overall_accuracy", "accepted_balanced_accuracy", "accepted_macro_f1", "accepted_weighted_f1",
        "accepted_cohen_kappa", "accepted_mcc", "raw_exact_pair_accuracy", "accepted_exact_pair_accuracy",
    ]:
        vals = [metrics.get(f"{d}__{k}") for d in directions]
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        metrics[f"mean__{k}"] = float(np.mean(vals)) if vals else float("nan")
    metrics["runtime_seconds"] = float(time.time() - t0)
    metrics["status"] = "completed"
    metrics["message"] = "completed"
    with open(outdir / "label_transfer_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, allow_nan=True)
    with open(outdir / "method_status.json", "w") as fh:
        json.dump({"dataset_id": args.dataset_id, "method": args.method_name, "status": "completed", "message": "completed"}, fh, indent=2)
    pd.DataFrame([{"dataset_id": args.dataset_id, "method": args.method_name, "status": "completed", "message": "completed"}]).to_csv(outdir / "method_status.csv", index=False)
    headline_cols = ["backend", "method", "dataset_id", "status", "message"] + [k for k in metrics if str(k).startswith("mean__")]
    pd.DataFrame([{k: metrics.get(k, np.nan) for k in headline_cols}]).to_csv(outdir / "metrics_headline.csv", index=False)
    with open(outdir / "run_settings.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    log(f"[DONE] embedding-kNN label transfer: {outdir}")


def build_gamma_argparser(default_backend: str = "v8", default_v10_alpha: float = 1.0) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Ambiguity-aware label transfer using diffusion {default_backend} gamma")
    add_common_args(p)
    p.set_defaults(v10_alpha=default_v10_alpha)
    return p


def build_embedding_knn_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ambiguity-aware label transfer from an integrated embedding using weighted kNN")
    add_common_args(p)
    p.add_argument("--method_name", default="embedding_knn")
    p.add_argument("--knn_k", type=int, default=15)
    p.add_argument("--knn_temperature", type=float, default=1.0)
    return p
