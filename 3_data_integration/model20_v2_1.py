"""
model20_v2_1.py — model20 + anchor-consolidation for cell-type-preserving integration
============================================================================

Purpose
-------
model20 is a deliberately conservative Bladder-only successor to the
model14/model15/model17 family.  The benchmark showed two distinct failure
modes:

  * model15/model17 preserve Bladder cell-type structure very well
    (high NMI/AMI/ARI), but source/modality mixing remains poor.
  * model18/model18_3/model19 introduce dynamic Γ machinery, but on Bladder
    they lose the stable cell-type structure and collapse toward modality-
    separated or low-NMI embeddings.

Therefore model20_v2_1 keeps the load-bearing fixed joint-Laplacian anchor from
model17 and adds only local, guarded mixing losses plus one biology-preserving
anchor-consolidation term.  It does NOT rebuild the
anchor from dynamic Γ, because that was the Bladder feedback-loop failure.
Instead, dynamic Γ is refreshed from z every N epochs and used only as a weak
same-pseudo-cluster attraction signal.

New pieces relative to model17
------------------------------
1. Anchor-derived pseudo-clusters (default k=3 for Bladder).  These are
   unsupervised clusters from the fixed anchor, not cell labels.
2. Cluster-conditional centroid alignment.  Within each pseudo-cluster, the
   mod1 centroid and mod2 centroid are pulled together.  This targets mixing
   while avoiding cross-cell-type pulls.
3. Cluster-conditional MMD.  MMD is computed inside each pseudo-cluster, not
   globally, preventing the global over-mixing that damages ARI.
4. Class-balanced domain adversarial loss with warmup.  Modality labels are
   supervised, but cell-type labels are never used for training.
5. Γ-pair attraction restricted to high-confidence reciprocal Γ pairs and
   same pseudo-cluster cells.  Initial Γ is computed from PCA space; dynamic Γ
   is refreshed from z-space every --dynamic_gamma_every epochs but only used
   as a weak pair loss, never to rebuild the anchor.
6. NEW in v2: anchor-consolidation loss.  A detached centroid is computed for
   each unsupervised anchor-derived pseudo-cluster in the current embedding;
   cells are gently pulled toward their own pseudo-cluster centroid.  This is
   intended to recover the model19-style ARI/NMI benefit on datasets11_10pct
   without increasing cross-modal over-mixing forces.

Design principle
----------------
For Bladder, preserve model17's structural prior first; improve mixing second.
If mixing losses become too strong, NMI/AMI/ARI will fall.  Start with the
runner defaults and tune w_cluster_centroid_align / w_dann / w_dynamic_gamma
conservatively.
"""


# model20_v2_1 modification note:
# This file intentionally keeps the model20_v2 architecture unchanged,
# but increases the impact of Gamma by using stronger initial reciprocal
# Gamma-pair attraction and stronger dynamic z-space Gamma-pair attraction.
# The pair selection remains reciprocal and same-pseudo-cluster restricted to
# avoid uncontrolled over-mixing.
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def log(msg: str):
    print(msg, flush=True)


# =============================================================================
# Data loading -- copies of the m16_1 helpers (well-tested, no changes)
# =============================================================================

def load_features(path: str) -> np.ndarray:
    """Load .mtx, .npy, .csv, or .csv.gz into a dense float32 array
    (cells x features)."""
    p = path.lower()
    if p.endswith(".mtx"):
        import scipy.io as sio
        X = sio.mmread(path)
        if hasattr(X, "todense"):
            X = X.todense()
        return np.asarray(X, dtype=np.float32)
    if p.endswith(".npy"):
        return np.load(path).astype(np.float32)
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def load_barcodes(path: str) -> np.ndarray:
    """Load a one-column TSV of cell barcodes (one barcode per matrix row)."""
    df = pd.read_csv(path, header=None, sep="\t", dtype=str, usecols=[0])
    return df.iloc[:, 0].astype(str).to_numpy()


def safe_pca(X: np.ndarray, n_components: int, seed: int, name: str) -> np.ndarray:
    """Standardize then TruncatedSVD to n_components."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    if n_comp <= 0:
        log(f"[PCA] {name}: skipping (n_comp <= 0); shape {X.shape}")
        return X.astype(np.float32)
    Xs = StandardScaler(with_mean=False).fit_transform(X)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    out = svd.fit_transform(Xs).astype(np.float32)
    var = float(svd.explained_variance_ratio_.sum())
    log(f"[PCA] {name}: {X.shape} -> {out.shape}  cum_var={var:.3f}")
    return out


def safe_combined_pca_centered(
    X1: np.ndarray, X2: np.ndarray, n_components: int, seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-modality standardize, then COMBINED PCA on [X1_std; X2_std].

    Used when the two modalities share the same raw feature space (e.g.
    Tabula Muris with 489 shared genes between 10X and FACS), to provide
    a shared coordinate system for the JOINT-DIFFUSION Γ computation.
    Per-modality standardisation removes protocol-specific mean/scale
    offsets that would otherwise dominate the first principal components.

    Returns (X1_combined, X2_combined) or (None, None) if not applicable.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    if X1.shape[1] != X2.shape[1]:
        return None, None
    n1 = X1.shape[0]
    X_stack = np.vstack([X1, X2])
    n_comp = min(n_components, X_stack.shape[1] - 1, X_stack.shape[0] - 1)
    if n_comp <= 0:
        return None, None

    sc1 = StandardScaler(with_mean=True, with_std=True)
    sc2 = StandardScaler(with_mean=True, with_std=True)
    X1s = sc1.fit_transform(X1)
    X2s = sc2.fit_transform(X2)
    X_combined = np.vstack([X1s, X2s]).astype(np.float32)

    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    out = svd.fit_transform(X_combined).astype(np.float32)
    var = float(svd.explained_variance_ratio_.sum())
    log(f"[PCA-COMBINED] per-modality standardize + combined SVD: "
        f"{X_combined.shape} -> {out.shape}  cum_var={var:.3f}")
    return out[:n1], out[n1:]


def load_batch_labels(labels_path: str, barcodes: np.ndarray,
                      id_col: str, batch_col: str) -> np.ndarray:
    """Return per-cell batch label aligned with `barcodes` order."""
    df = pd.read_csv(labels_path)
    if id_col not in df.columns:
        raise SystemExit(f"missing id column '{id_col}' in {labels_path}")
    if batch_col not in df.columns:
        log(f"[WARN] no '{batch_col}' in {labels_path}; using single batch")
        return np.full(len(barcodes), "default", dtype=object)
    bc_to_batch = dict(zip(df[id_col].astype(str), df[batch_col].astype(str)))
    out = np.array([bc_to_batch.get(str(bc), "__unknown__")
                    for bc in barcodes], dtype=object)
    n_unk = int((out == "__unknown__").sum())
    if n_unk > 0:
        log(f"[WARN] {n_unk}/{len(out)} barcodes had no batch label")
    return out


# =============================================================================
# Γ (diffusion-distance correspondence) -- JOINT-DIFFUSION variant
# =============================================================================
#
# Used ONCE at init for cross-protocol data. NEVER refreshed (the refresh
# was the source of m16_1 v3.3's feedback-loop failure on Bladder).
#
# Pipeline (joint-diffusion, the mathematically-correct variant):
#   1. Stack [X1; X2] in a shared feature space (combined-PCA centered).
#   2. Build a kNN graph on the joint stack.
#   3. Symmetric-normalised affinity -> ONE eigsh -> joint diffusion
#      coordinates D in a SINGLE basis.
#   4. Split D back into D1, D2 by row.
#   5. Multi-scale (concatenate D weighted by eigval^t for t in {1,2,4}).
#   6. Cost = pairwise L2 distance between row-medians-normalised D1 vs D2.
#   7. Sinkhorn-regularised transport plan Γ = exp(-cost / epsilon)
#      iterated to row/column doubly-stochastic.
#
# (Compare to the literal m10/m15 implementation that does TWO
# independent per-modality eigsh calls; on synthetic data with K=3 that
# gives 33% top-1 match rate because scipy.sparse.linalg.eigsh returns
# eigenvectors with arbitrary per-column signs, making D1 and D2
# basis-incompatible. The joint variant gives 99.8% on the same
# synthetic test.)
# =============================================================================


def _diffusion_coordinates_joint(
    X_all: np.ndarray, knn_k: int, n_components: int,
    multiscale_times: Tuple[int, ...] = (1, 2, 4),
    seed: int = 42,
) -> np.ndarray:
    """Joint diffusion coordinates of [X1; X2] in a single basis.

    Returns an (n1+n2, n_components * len(multiscale_times)) array.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import NearestNeighbors

    n = X_all.shape[0]
    k = min(knn_k + 1, n - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X_all)
    dists, idx = nn.kneighbors(X_all)

    # Gaussian-kernel weights using a self-tuning bandwidth (median of k-NN)
    sigma = np.median(dists[:, -1]) + 1e-8
    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].flatten()
    weights = np.exp(-(dists[:, 1:].flatten() ** 2) / (2.0 * sigma ** 2))
    weights = weights.astype(np.float64)
    A = csr_matrix((weights, (rows, cols)), shape=(n, n))
    A = 0.5 * (A + A.T)

    d = np.asarray(A.sum(axis=1)).flatten().astype(np.float64)
    d[d < 1e-8] = 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = csr_matrix(
        (d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n),
    )
    S = D_inv_sqrt @ A @ D_inv_sqrt

    k_eig = min(n_components + 1, n - 2)
    try:
        evals, evecs = eigsh(S, k=k_eig, which="LA", maxiter=4000)
    except Exception as ex:
        log(f"[GAMMA] joint eigsh failed ({ex}); falling back to stacked PCA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, X_all.shape[1] - 1),
                  random_state=seed)
        return pca.fit_transform(X_all).astype(np.float32)

    order = np.argsort(evals)[::-1]
    evals = evals[order][1:]      # drop trivial constant eigenvector
    evecs = evecs[:, order][:, 1:]
    evals = np.clip(evals, 0.0, 1.0)

    blocks = []
    for t in multiscale_times:
        blocks.append((evecs * (evals[None, :] ** float(t))).astype(np.float32))
    return np.concatenate(blocks, axis=1)


def _sinkhorn(cost: np.ndarray, epsilon: float, n_iters: int) -> np.ndarray:
    """Sinkhorn-Knopp on a normalised cost matrix.

    Returns Γ (n1 x n2). Both rows and columns are pushed toward uniform
    marginals 1/n1 and 1/n2 respectively.
    """
    n1, n2 = cost.shape
    # Median-normalise the cost so epsilon has a stable interpretation
    c_med = np.median(cost) + 1e-12
    K = np.exp(-(cost / c_med) / float(epsilon)).astype(np.float64)
    u = np.full(n1, 1.0 / n1)
    v = np.full(n2, 1.0 / n2)
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2
    for _ in range(n_iters):
        Kv = K @ v
        Kv[Kv < 1e-30] = 1e-30
        u = a / Kv
        Ku = K.T @ u
        Ku[Ku < 1e-30] = 1e-30
        v = b / Ku
    return (u[:, None] * K * v[None, :]).astype(np.float32)


def compute_gamma_diffusion(
    X1_in: np.ndarray, X2_in: np.ndarray,
    knn_k: int, n_components: int,
    multiscale_times: Tuple[int, ...] = (1, 2, 4),
    epsilon: float = 0.1, sinkhorn_iters: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """JOINT-diffusion Γ correspondence between mod1 and mod2.

    The inputs should already share a coordinate system (e.g.
    combined-PCA-centered coordinates from `safe_combined_pca_centered`).
    The joint diffusion gives a SINGLE eigenbasis used by both
    modalities, eliminating the basis-incompatibility failure mode of
    independent per-modality eigsh.

    Returns Γ of shape (n1, n2). Higher Γ[i, j] means mod1[i] and
    mod2[j] are more likely the same cell type.
    """
    n1, n2 = X1_in.shape[0], X2_in.shape[0]
    d = min(X1_in.shape[1], X2_in.shape[1])
    if X1_in.shape[1] != X2_in.shape[1]:
        log(f"[GAMMA] WARN: input dims differ ({X1_in.shape[1]} vs "
            f"{X2_in.shape[1]}); truncating to {d}")
    X_all = np.vstack([X1_in[:, :d], X2_in[:, :d]]).astype(np.float32)
    log(f"[GAMMA] joint stack X_all={X_all.shape}  (n1={n1}, n2={n2})")

    coords = _diffusion_coordinates_joint(
        X_all, knn_k=knn_k, n_components=n_components,
        multiscale_times=multiscale_times, seed=seed,
    )
    D1 = coords[:n1]
    D2 = coords[n1:]
    log(f"[GAMMA] joint diffusion coords {coords.shape} "
        f"-> D1 {D1.shape}, D2 {D2.shape}")

    # Row-median normalise so per-cell magnitude differences don't dominate
    def _rmed_norm(X):
        m = np.median(np.linalg.norm(X, axis=1)) + 1e-8
        return X / m
    D1n = _rmed_norm(D1)
    D2n = _rmed_norm(D2)

    # Pairwise L2 cost
    from scipy.spatial.distance import cdist
    cost = cdist(D1n, D2n, metric="euclidean").astype(np.float64)

    gamma = _sinkhorn(cost, epsilon, sinkhorn_iters)
    log(f"[GAMMA] Γ done: shape={gamma.shape}  sum={gamma.sum():.4f}  "
        f"max={gamma.max():.3e}  mean={gamma.mean():.3e}")
    return gamma


def gamma_top_partners(
    gamma: np.ndarray, k: int, rel_threshold: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each row, top-k columns with γ[i,j] >= rel_threshold * median(row).
    Filtered slots are marked -1.
    """
    n1, n2 = gamma.shape
    if k <= 0:
        return (np.empty((n1, 0), dtype=np.int64),
                np.empty((n2, 0), dtype=np.int64))
    k1 = min(k, n2); k2 = min(k, n1)
    p_1to2 = np.argsort(-gamma, axis=1)[:, :k1].astype(np.int64)
    row_med = np.median(gamma, axis=1, keepdims=True) + 1e-30
    row_th = (row_med * rel_threshold).squeeze(1)
    chosen = np.take_along_axis(gamma, p_1to2, axis=1)
    p_1to2[chosen < row_th[:, None]] = -1
    p_2to1 = np.argsort(-gamma, axis=0).T[:, :k2].astype(np.int64)
    col_med = np.median(gamma, axis=0) + 1e-30
    col_th = col_med * rel_threshold
    chosen2 = np.take_along_axis(gamma.T, p_2to1, axis=1)
    p_2to1[chosen2 < col_th[:, None]] = -1
    n_kept_1 = int((p_1to2 >= 0).sum())
    n_kept_2 = int((p_2to1 >= 0).sum())
    log(f"[GAMMA] partners: 1to2={n_kept_1}/{p_1to2.size}  "
        f"2to1={n_kept_2}/{p_2to1.size}  (rel_threshold={rel_threshold})")
    return p_1to2, p_2to1


# =============================================================================
# Model: per-modality MLP encoder/decoder
# =============================================================================

class MLPBlock(nn.Module):
    """Linear -> LayerNorm -> GELU -> Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.gelu(self.norm(self.linear(x))))


class ModalityEncoder(nn.Module):
    """Per-modality MLP encoder. in_dim -> hidden... -> emb_dim (linear out)."""

    def __init__(self, in_dim: int, hidden_dims: List[int],
                 emb_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(MLPBlock(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModalityDecoder(nn.Module):
    """Per-modality MLP decoder. emb_dim -> hidden... -> out_dim (linear out)."""

    def __init__(self, emb_dim: int, hidden_dims: List[int],
                 out_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = emb_dim
        for h in hidden_dims:
            layers.append(MLPBlock(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# =============================================================================
# Losses
# =============================================================================

def mmd_loss(z1: torch.Tensor, z2: torch.Tensor,
             n_sample: int = 512,
             sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
             ) -> torch.Tensor:
    """Multi-bandwidth Gaussian-kernel MMD² between z1 and z2."""
    if n_sample > 0 and z1.shape[0] > n_sample:
        idx = torch.randperm(z1.shape[0], device=z1.device)[:n_sample]
        z1 = z1[idx]
    if n_sample > 0 and z2.shape[0] > n_sample:
        idx = torch.randperm(z2.shape[0], device=z2.device)[:n_sample]
        z2 = z2[idx]
    if z1.shape[0] < 2 or z2.shape[0] < 2:
        return z1.new_tensor(0.0)
    d11 = torch.cdist(z1, z1, p=2.0) ** 2
    d22 = torch.cdist(z2, z2, p=2.0) ** 2
    d12 = torch.cdist(z1, z2, p=2.0) ** 2
    mmd2 = z1.new_tensor(0.0)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        K11 = torch.exp(-gamma * d11)
        K22 = torch.exp(-gamma * d22)
        K12 = torch.exp(-gamma * d12)
        mmd2 = mmd2 + K11.mean() + K22.mean() - 2.0 * K12.mean()
    return mmd2 / float(len(sigmas))


def multi_batch_mmd(z: torch.Tensor, batch_int_t: torch.Tensor,
                    min_per_batch: int = 8,
                    n_sample: int = 128,
                    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
                    ) -> Tuple[torch.Tensor, int]:
    """Average pairwise MMD across all (batch_i, batch_j) pairs."""
    unique_b = torch.unique(batch_int_t).tolist()
    n_b = len(unique_b)
    if n_b < 2:
        return z.new_tensor(0.0), 0
    total = z.new_tensor(0.0)
    n_pairs = 0
    for i in range(n_b):
        zi = z[batch_int_t == unique_b[i]]
        if zi.shape[0] < min_per_batch:
            continue
        for j in range(i + 1, n_b):
            zj = z[batch_int_t == unique_b[j]]
            if zj.shape[0] < min_per_batch:
                continue
            total = total + mmd_loss(zi, zj, n_sample=n_sample, sigmas=sigmas)
            n_pairs += 1
    if n_pairs == 0:
        return z.new_tensor(0.0), 0
    return total / float(n_pairs), n_pairs


def compute_mnn_pairs(z: torch.Tensor, n1: int, k: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Mutual nearest neighbours between mod1 (z[:n1]) and mod2 (z[n1:])."""
    from sklearn.neighbors import NearestNeighbors

    z_np = z.detach().cpu().numpy()
    n2 = z_np.shape[0] - n1
    if n1 < 2 or n2 < 2:
        empty = torch.empty(0, dtype=torch.long, device=z.device)
        return empty, empty, 0
    z1 = z_np[:n1]
    z2 = z_np[n1:]
    k1 = min(k, n2)
    k2 = min(k, n1)
    nn12 = NearestNeighbors(n_neighbors=k1).fit(z2)
    _, idx_1to2 = nn12.kneighbors(z1)
    nn21 = NearestNeighbors(n_neighbors=k2).fit(z1)
    _, idx_2to1 = nn21.kneighbors(z2)
    i_arr = np.repeat(np.arange(n1), k1)
    j_arr = idx_1to2.flatten()
    j_lookup = idx_2to1[j_arr]
    is_mnn = (j_lookup == i_arr[:, None]).any(axis=1)
    i_mnn = i_arr[is_mnn]
    j_mnn = j_arr[is_mnn]
    if len(i_mnn) == 0:
        empty = torch.empty(0, dtype=torch.long, device=z.device)
        return empty, empty, 0
    a = torch.from_numpy(i_mnn).long().to(z.device)
    b = torch.from_numpy(j_mnn + n1).long().to(z.device)
    return a, b, int(len(i_mnn))


def mnn_attraction_loss(z: torch.Tensor, a: torch.Tensor,
                        b: torch.Tensor) -> torch.Tensor:
    """MSE pull on MNN pairs."""
    if a.numel() == 0:
        return z.new_tensor(0.0)
    diff = z[a] - z[b]
    return (diff * diff).sum(dim=1).mean()


def variance_loss(z: torch.Tensor, var_target: float) -> torch.Tensor:
    """VICReg variance term: hinge on per-dim std."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    return F.relu(var_target - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: sum of squared off-diagonal of cov / D."""
    n, d = z.shape
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag * off_diag).sum() / float(d)


def barcode_distillation_loss(
    z: torch.Tensor, n1: int,
    corr_i: torch.Tensor, corr_j: torch.Tensor,
    teacher_modality: int,
) -> torch.Tensor:
    """Asymmetric teacher-student MSE on barcode-matched pairs.

    teacher_modality: 1 for mod1 teacher (mod2 pulls toward mod1)
                      2 for mod2 teacher (mod1 pulls toward mod2)
    The teacher side is detached -- gradients only flow into the student.

    On model17 this is ONLY used for paired data; the m16_1 z-MNN
    fallback (which fed the cross-protocol feedback loop) is gone.
    """
    if corr_i.numel() == 0:
        return z.new_tensor(0.0)
    z1_pairs = z[corr_i]            # mod1 cells in pair
    z2_pairs = z[corr_j + n1]       # mod2 cells in pair (offset by n1)
    if teacher_modality == 1:
        teacher = z1_pairs.detach()
        student = z2_pairs
    else:
        teacher = z2_pairs.detach()
        student = z1_pairs
    return F.mse_loss(student, teacher)


# =============================================================================
# Anchor: joint Laplacian eigenmap (FIXED, never refreshed)
# =============================================================================
#
# The single most important structural prior. Computed ONCE at init.
# Two paths for cross-modal edges depending on data type:
#
#   PAIRED (barcode overlap >= 50%):
#     Cross-modal edges from barcode matching -- perfect 1-to-1
#     correspondence. Reliable, no refresh needed.
#
#   CROSS-PROTOCOL (barcode overlap < 50%):
#     Cross-modal edges from Γ top-k partners with relative-threshold
#     filtering. Γ is computed ONCE from combined-PCA-centered
#     coordinates using JOINT diffusion (single eigsh, shared basis).
#     This was the load-bearing m14/m15 mechanism on Bladder.
#
# DELIBERATELY REMOVED from m16_1:
#   - combined-PCA MNN fallback   (Γ is strictly better when available)
#   - per-modality PCA MNN fallback   (legacy, never best)
#   - dynamic refresh                  (feedback loop on cross-protocol)
# =============================================================================


def precompute_anchor(
    X1p: np.ndarray, X2p: np.ndarray,
    bc1: np.ndarray, bc2: np.ndarray,
    emb_dim: int,
    k_intra: int,
    k_inter: int,
    seed: int,
    gamma_partners_1to2: Optional[np.ndarray] = None,
    gamma_partners_2to1: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    """Joint Laplacian-eigenmap target embedding for both modalities.

    Returns (anchor (N x emb_dim), mode_string).
    mode_string is "paired" or "gamma" or "no_cross" for logging.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    n1, n2 = X1p.shape[0], X2p.shape[0]
    N = n1 + n2
    log(f"[ANCHOR] building joint graph: n1={n1} n2={n2} "
        f"k_intra={k_intra} k_inter={k_inter}")

    row_ids: List[int] = []
    col_ids: List[int] = []

    # ---- intra-modal kNN: mod1 ----
    k1 = min(k_intra + 1, n1)
    nn1 = NearestNeighbors(n_neighbors=k1).fit(X1p)
    _, idx1 = nn1.kneighbors(X1p)
    for i in range(n1):
        for j in idx1[i, 1:]:  # skip self
            row_ids.append(i)
            col_ids.append(int(j))

    # ---- intra-modal kNN: mod2 ----
    k2 = min(k_intra + 1, n2)
    nn2 = NearestNeighbors(n_neighbors=k2).fit(X2p)
    _, idx2 = nn2.kneighbors(X2p)
    for i in range(n2):
        for j in idx2[i, 1:]:
            row_ids.append(n1 + i)
            col_ids.append(n1 + int(j))

    # ---- cross-modal edges: 2 paths only ----
    bc1_to_idx = {b: i for i, b in enumerate(bc1)}
    paired_i, paired_j = [], []
    for j, bc in enumerate(bc2):
        ii = bc1_to_idx.get(bc)
        if ii is not None:
            paired_i.append(ii)
            paired_j.append(j)
    n_paired = len(paired_i)
    paired_threshold = 0.5 * min(n1, n2)

    if n_paired >= paired_threshold:
        # PAIRED data: barcode matching is the cross-modal source.
        log(f"[ANCHOR] paired mode: {n_paired} barcode-matched cross-modal "
            f"pairs (>= {int(paired_threshold)} threshold)")
        for ii, jj in zip(paired_i, paired_j):
            row_ids.append(int(ii))
            col_ids.append(n1 + int(jj))
            row_ids.append(n1 + int(jj))
            col_ids.append(int(ii))
        mode = "paired"
    elif (gamma_partners_1to2 is not None
          and gamma_partners_2to1 is not None):
        # CROSS-PROTOCOL: Γ partners are the cross-modal source.
        n_added = 0
        for i in range(n1):
            for j in gamma_partners_1to2[i]:
                if j >= 0:
                    row_ids.append(int(i))
                    col_ids.append(n1 + int(j))
                    n_added += 1
        for j in range(n2):
            for i in gamma_partners_2to1[j]:
                if i >= 0:
                    row_ids.append(n1 + int(j))
                    col_ids.append(int(i))
                    n_added += 1
        log(f"[ANCHOR] cross-protocol Γ mode: {n_added} cross-modal "
            f"edges from joint-diffusion Γ partners")
        mode = "gamma"
    else:
        log(f"[ANCHOR] WARN: no cross-modal source available "
            f"(paired={n_paired}, no Γ partners supplied) -- anchor will "
            f"not link the modalities; relying on MMD in z-space.")
        mode = "no_cross"

    # ---- Build sparse symmetric adjacency ----
    data = np.ones(len(row_ids), dtype=np.float32)
    A = csr_matrix((data, (row_ids, col_ids)), shape=(N, N))
    A = (A + A.T)
    A.data = np.minimum(A.data, 1.0)

    # ---- Symmetric-normalised affinity ----
    d = np.asarray(A.sum(axis=1)).flatten().astype(np.float64)
    d[d < 1e-8] = 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = csr_matrix(
        (d_inv_sqrt, (np.arange(N), np.arange(N))), shape=(N, N),
    )
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # ---- Top emb_dim+1 eigenvectors, drop trivial constant ----
    k_eig = min(emb_dim + 1, N - 2)
    try:
        vals_e, vecs_e = eigsh(A_norm, k=k_eig, which="LA", maxiter=2000)
    except Exception as ex:
        log(f"[ANCHOR] eigsh failed ({ex}); using random fallback")
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((N, emb_dim)).astype(np.float32)
        vecs = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-8)
        return vecs.astype(np.float32), mode + "_random_fallback"

    order = np.argsort(vals_e)[::-1]
    vecs_e = vecs_e[:, order]
    vecs_e = vecs_e[:, 1:emb_dim + 1]
    if vecs_e.shape[1] < emb_dim:
        pad = emb_dim - vecs_e.shape[1]
        vecs_e = np.concatenate(
            [vecs_e, np.zeros((N, pad), dtype=np.float32)], axis=1,
        )
    mu = vecs_e.mean(0, keepdims=True)
    sd = vecs_e.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    vecs_e = (vecs_e - mu) / sd
    log(f"[ANCHOR] computed anchor shape {vecs_e.shape} (mode={mode})")
    return vecs_e.astype(np.float32), mode


def get_barcode_correspondence(
    bc1: np.ndarray, bc2: np.ndarray, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """For paired data, return (corr_i, corr_j, n) of mod1<->mod2 cell
    indices from barcode matching."""
    bc1_to_idx = {b: i for i, b in enumerate(bc1)}
    pi, pj = [], []
    for j, bc in enumerate(bc2):
        ii = bc1_to_idx.get(bc)
        if ii is not None:
            pi.append(ii)
            pj.append(j)
    if not pi:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)
    return (torch.tensor(pi, dtype=torch.long, device=device),
            torch.tensor(pj, dtype=torch.long, device=device),
            len(pi))


# =============================================================================
# model20 additions: pseudo-cluster and guarded modality-mixing losses
# =============================================================================

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Small classifier used through gradient reversal to remove modality signal."""
    def __init__(self, emb_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def dann_alpha(epoch: int, warmup: int, max_epochs: int, alpha_max: float) -> float:
    """DANN schedule: zero before warmup, smooth Ganin ramp afterward."""
    if epoch <= warmup or alpha_max <= 0:
        return 0.0
    denom = max(1, max_epochs - warmup)
    p = min(1.0, max(0.0, (epoch - warmup) / float(denom)))
    return float(alpha_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


def kmeans_labels_from_anchor(anchor_np: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Unsupervised pseudo-clusters from the fixed anchor embedding."""
    from sklearn.cluster import KMeans
    k_eff = max(2, min(int(k), max(2, anchor_np.shape[0] // 10)))
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init=20)
    return km.fit_predict(anchor_np).astype(np.int64)


def cluster_centroid_alignment_loss(
    z: torch.Tensor,
    n1: int,
    cluster_labels: torch.Tensor,
    n_clusters: int,
    min_per_modality: int = 8,
) -> Tuple[torch.Tensor, int]:
    """Pull mod1 and mod2 centroids together inside each pseudo-cluster."""
    losses = []
    mod1 = torch.arange(z.shape[0], device=z.device) < n1
    mod2 = ~mod1
    for c in range(n_clusters):
        mask = cluster_labels == c
        m1 = mask & mod1
        m2 = mask & mod2
        if int(m1.sum().item()) < min_per_modality or int(m2.sum().item()) < min_per_modality:
            continue
        losses.append(F.mse_loss(z[m1].mean(dim=0), z[m2].mean(dim=0)))
    if not losses:
        return z.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)


def cluster_conditional_mmd_loss(
    z: torch.Tensor,
    n1: int,
    cluster_labels: torch.Tensor,
    n_clusters: int,
    min_per_modality: int = 8,
    n_sample: int = 128,
) -> Tuple[torch.Tensor, int]:
    """MMD between modalities inside each pseudo-cluster."""
    losses = []
    mod1 = torch.arange(z.shape[0], device=z.device) < n1
    mod2 = ~mod1
    for c in range(n_clusters):
        mask = cluster_labels == c
        z1c = z[mask & mod1]
        z2c = z[mask & mod2]
        if z1c.shape[0] < min_per_modality or z2c.shape[0] < min_per_modality:
            continue
        losses.append(mmd_loss(z1c, z2c, n_sample=n_sample))
    if not losses:
        return z.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)




def anchor_consolidation_loss(
    z: torch.Tensor,
    cluster_labels: torch.Tensor,
    n_clusters: int,
    min_cluster_size: int = 10,
) -> Tuple[torch.Tensor, int]:
    """Pull cells toward detached centroids of their anchor-derived pseudo-clusters.

    This is the only architectural addition in model20_v2_1. It is deliberately
    label-free: pseudo-clusters come from the fixed anchor, not from cell-type
    labels.  The centroids are detached so the loss tightens the current
    representation without letting centroids chase individual cells.  It should
    counteract over-mixing in high-cell-type datasets while leaving the existing
    modality-mixing losses unchanged.
    """
    losses = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        if int(mask.sum().item()) < int(min_cluster_size):
            continue
        centroid = z[mask].mean(dim=0).detach()
        losses.append(((z[mask] - centroid) ** 2).sum(dim=1).mean())
    if not losses:
        return z.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)

def gamma_reciprocal_pairs(
    gamma: np.ndarray,
    cluster_labels_np: Optional[np.ndarray],
    n1: int,
    top_q: float,
    max_pairs: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """High-confidence reciprocal Γ top-1 pairs, optionally restricted to same pseudo-cluster."""
    n2 = gamma.shape[1]
    row_top = np.argmax(gamma, axis=1)
    row_val = gamma[np.arange(n1), row_top]
    col_top = np.argmax(gamma, axis=0)
    keep_i = []
    keep_j = []
    keep_v = []
    for i in range(n1):
        j = int(row_top[i])
        if j < 0 or j >= n2 or int(col_top[j]) != i:
            continue
        if cluster_labels_np is not None and cluster_labels_np[i] != cluster_labels_np[n1 + j]:
            continue
        keep_i.append(i); keep_j.append(j); keep_v.append(float(row_val[i]))
    if len(keep_i) == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, 0
    vals = np.asarray(keep_v)
    q = min(0.99, max(0.0, float(top_q)))
    th = np.quantile(vals, q) if len(vals) > 10 else vals.min()
    idx = np.where(vals >= th)[0]
    if max_pairs > 0 and len(idx) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_pairs, replace=False)
    a = torch.tensor(np.asarray(keep_i, dtype=np.int64)[idx], dtype=torch.long, device=device)
    b = torch.tensor(n1 + np.asarray(keep_j, dtype=np.int64)[idx], dtype=torch.long, device=device)
    return a, b, int(a.numel())


def dynamic_gamma_pairs_from_z(
    z: torch.Tensor,
    n1: int,
    cluster_labels_np: Optional[np.ndarray],
    epsilon: float,
    top_q: float,
    max_pairs: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Infer a small z-space Γ by Sinkhorn, then keep high-confidence reciprocal pairs."""
    from scipy.spatial.distance import cdist
    device = z.device
    z_np = z.detach().cpu().numpy().astype(np.float32)
    z1 = z_np[:n1]
    z2 = z_np[n1:]
    cost = cdist(z1, z2, metric="sqeuclidean").astype(np.float64)
    med = np.median(cost) + 1e-12
    K = np.exp(-(cost / med) / max(float(epsilon), 1e-4))
    n2 = z2.shape[0]
    a = np.ones(n1, dtype=np.float64) / float(n1)
    b = np.ones(n2, dtype=np.float64) / float(n2)
    u = np.ones(n1, dtype=np.float64)
    v = np.ones(n2, dtype=np.float64)
    for _ in range(50):
        Kv = K @ v; Kv[Kv < 1e-30] = 1e-30
        u = a / Kv
        KTu = K.T @ u; KTu[KTu < 1e-30] = 1e-30
        v = b / KTu
    gamma = (u[:, None] * K * v[None, :]).astype(np.float32)
    return gamma_reciprocal_pairs(
        gamma, cluster_labels_np=cluster_labels_np, n1=n1,
        top_q=top_q, max_pairs=max_pairs, seed=seed, device=device,
    )


# =============================================================================
# Argparser
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="model20 -- Bladder-only guarded modality mixing",
    )
    p.add_argument("--dataset_id", type=str, required=True)
    p.add_argument("--out_prefix", type=str, required=True)

    # Inputs
    p.add_argument("--x1", type=str, required=True)
    p.add_argument("--x2", type=str, required=True)
    p.add_argument("--barcodes1", type=str, required=True)
    p.add_argument("--barcodes2", type=str, required=True)
    p.add_argument("--labels1", type=str, required=True)
    p.add_argument("--labels2", type=str, required=True)
    p.add_argument("--label_id_col", type=str, default="cell_id")
    p.add_argument("--batch_col", type=str, default="batch")

    # Preprocessing
    p.add_argument("--pca_dim_mod1", type=int, default=100)
    p.add_argument("--pca_dim_mod2", type=int, default=100)

    # Architecture
    p.add_argument("--emb_dim", type=int, default=48)
    p.add_argument("--mlp_hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--dropout", type=float, default=0.10)

    # Loss weights (always-on)
    p.add_argument("--w_recon", type=float, default=1.0)
    p.add_argument("--w_mmd", type=float, default=1.0)
    p.add_argument("--w_anchor", type=float, default=1.5)
    p.add_argument("--w_mnn", type=float, default=1.0)
    p.add_argument("--w_var", type=float, default=0.10)
    p.add_argument("--w_cov", type=float, default=0.04)
    p.add_argument("--var_target", type=float, default=1.0)


    # model20 guarded mixing losses (Bladder defaults)
    p.add_argument("--pseudo_k", type=int, default=3,
                   help="Number of anchor-derived pseudo-clusters; Bladder default is 3.")
    p.add_argument("--cluster_warmup_epochs", type=int, default=40)
    p.add_argument("--w_cluster_centroid_align", type=float, default=1.0)
    p.add_argument("--w_cluster_mmd", type=float, default=0.5)
    p.add_argument("--cluster_min_per_modality", type=int, default=8)
    p.add_argument("--cluster_mmd_n_sample", type=int, default=128)
    p.add_argument("--w_anchor_consolidate", type=float, default=0.0,
                   help="model20_v2_1: label-free pull toward detached centroids of anchor-derived pseudo-clusters.")
    p.add_argument("--anchor_consolidate_warmup_epochs", type=int, default=40)
    p.add_argument("--anchor_consolidate_min_cluster_size", type=int, default=10)

    p.add_argument("--w_dann", type=float, default=0.25,
                   help="Class-balanced modality-adversarial loss weight.")
    p.add_argument("--dann_warmup_epochs", type=int, default=80)
    p.add_argument("--dann_lambda_max", type=float, default=0.5)
    p.add_argument("--domain_hidden", type=int, default=64)

    p.add_argument("--w_gamma_pairs", type=float, default=0.35,
                   help="Weak attraction on initial PCA-space reciprocal Γ pairs.")
    p.add_argument("--gamma_pair_warmup_epochs", type=int, default=40)
    p.add_argument("--gamma_pair_top_q", type=float, default=0.40)
    p.add_argument("--gamma_pair_max_pairs", type=int, default=6000)

    p.add_argument("--w_dynamic_gamma", type=float, default=0.20,
                   help="Weak attraction on dynamic z-space Γ pairs; never rebuilds anchor.")
    p.add_argument("--dynamic_gamma_warmup_epochs", type=int, default=80)
    p.add_argument("--dynamic_gamma_every", type=int, default=20)
    p.add_argument("--dynamic_gamma_epsilon", type=float, default=0.08)
    p.add_argument("--dynamic_gamma_top_q", type=float, default=0.60)
    p.add_argument("--dynamic_gamma_max_pairs", type=int, default=5000)

    # Loss weights (conditional)
    # batch_mmd: only active when >=2 batches detected per modality
    p.add_argument("--w_batch_mmd", type=float, default=1.0,
                   help="Weight for multi-batch MMD. Auto-skipped if "
                        "fewer than 2 batches per modality.")
    p.add_argument("--batch_mmd_n_sample", type=int, default=128)
    p.add_argument("--batch_mmd_min_per_batch", type=int, default=8)

    # distill: only active for paired data (barcode-matched)
    p.add_argument("--w_distill", type=float, default=1.0,
                   help="Weight for asymmetric barcode-matched "
                        "distillation. Auto-skipped on cross-protocol data.")
    p.add_argument("--teacher_modality", type=int, default=0,
                   help="0 = auto (larger modality is teacher), "
                        "1 = mod1, 2 = mod2.")
    p.add_argument("--distill_warmup_epochs", type=int, default=20)

    # MMD / MNN
    p.add_argument("--mmd_n_sample", type=int, default=512)
    p.add_argument("--mnn_k", type=int, default=10)
    p.add_argument("--mnn_warmup_epochs", type=int, default=20)
    p.add_argument("--mnn_refresh_every", type=int, default=20)

    # Anchor build
    p.add_argument("--anchor_k_intra", type=int, default=10)
    p.add_argument("--anchor_k_inter", type=int, default=5)

    # Γ (cross-protocol only)
    p.add_argument("--gamma_diff_knn_k", type=int, default=30)
    p.add_argument("--gamma_diff_components", type=int, default=30)
    p.add_argument("--gamma_multiscale_times", type=str, default="1,2,4")
    p.add_argument("--gamma_epsilon", type=float, default=0.1)
    p.add_argument("--gamma_sinkhorn_iters", type=int, default=500)
    p.add_argument("--gamma_k_partners", type=int, default=5)
    p.add_argument("--gamma_rel_threshold", type=float, default=2.0)

    # Training
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--external_gamma", type=str, default="", help="path to a precomputed coupling (e.g. v12) to use instead of the internal diffusion gamma")

    return p


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Main
# =============================================================================

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    log("=" * 70)
    log(f"model20 :: {args.dataset_id}  (device={device})")
    log("=" * 70)

    # ---- Load data ----
    X1 = load_features(args.x1)
    X2 = load_features(args.x2)
    bc1 = load_barcodes(args.barcodes1)
    bc2 = load_barcodes(args.barcodes2)
    n1, n2 = X1.shape[0], X2.shape[0]
    if X1.shape[0] != bc1.shape[0]:
        raise SystemExit(f"X1 rows ({X1.shape[0]}) != barcodes1 ({bc1.shape[0]})")
    if X2.shape[0] != bc2.shape[0]:
        raise SystemExit(f"X2 rows ({X2.shape[0]}) != barcodes2 ({bc2.shape[0]})")
    log(f"[DATA] X1={X1.shape}  X2={X2.shape}")

    # ---- Detect paired vs cross-protocol ----
    bc1_set = set(bc1.tolist())
    n_overlap = sum(1 for b in bc2 if b in bc1_set)
    is_paired = n_overlap >= 0.5 * min(n1, n2)
    log(f"[MODE] barcode overlap = {n_overlap}/{min(n1, n2)}  -> "
        f"is_paired={is_paired}")

    # ---- Batch labels (for multi-batch MMD) ----
    batches1 = load_batch_labels(args.labels1, bc1,
                                 args.label_id_col, args.batch_col)
    batches2 = load_batch_labels(args.labels2, bc2,
                                 args.label_id_col, args.batch_col)
    batches_all = np.concatenate([batches1, batches2])
    uniq_b = sorted(set(batches_all.tolist()) - {"__unknown__", "default"})
    n_batches_global = len(uniq_b)
    # Multi-batch MMD only makes sense with multiple batches WITHIN a modality
    n_batches_mod1 = len(set(batches1.tolist())
                        - {"__unknown__", "default"})
    n_batches_mod2 = len(set(batches2.tolist())
                        - {"__unknown__", "default"})
    multi_batch_active = (
        is_paired
        and args.w_batch_mmd > 0
        and (n_batches_mod1 >= 2 or n_batches_mod2 >= 2)
    )
    log(f"[BATCH] mod1 batches={n_batches_mod1}  mod2 batches={n_batches_mod2}"
        f"  multi_batch_mmd={'ON' if multi_batch_active else 'OFF'}")

    # Encode batches to integers (combined across modalities)
    batch_to_int = {b: i for i, b in enumerate(uniq_b)}
    batch_int_np = np.array(
        [batch_to_int.get(str(b), -1) for b in batches_all],
        dtype=np.int64,
    )
    batch_int_t = torch.from_numpy(batch_int_np).to(device)

    # ---- PCA ----
    X1p = safe_pca(X1, args.pca_dim_mod1, args.seed, "X1")
    X2p = safe_pca(X2, args.pca_dim_mod2, args.seed, "X2")
    X1_t = torch.from_numpy(X1p).to(device)
    X2_t = torch.from_numpy(X2p).to(device)

    # ---- Combined-PCA for Γ input (cross-protocol only) ----
    X1_combined = X2_combined = None
    if not is_paired and X1.shape[1] == X2.shape[1] and X1.shape[1] > 0:
        log(f"[PCA] X1 and X2 share feature space ({X1.shape[1]} features) "
            f"-- computing combined PCA for Γ input")
        X1_combined, X2_combined = safe_combined_pca_centered(
            X1, X2, n_components=args.pca_dim_mod1, seed=args.seed,
        )

    # ---- Γ (cross-protocol only, once at init, NEVER refreshed for anchor) ----
    gamma_np = None
    gamma_p_1to2 = gamma_p_2to1 = None
    if not is_paired and args.w_anchor > 0:
        gamma_input_1 = X1_combined if X1_combined is not None else X1p
        gamma_input_2 = X2_combined if X2_combined is not None else X2p
        log(f"[GAMMA] computing joint-diffusion Γ for cross-protocol data")
        t0 = time.time()
        times_tuple = tuple(int(t) for t in args.gamma_multiscale_times.split(","))
        ext = getattr(args, "external_gamma", "")
        if ext and os.path.exists(ext):
            gamma_np = np.load(ext).astype(np.float32)
            if gamma_np.shape != (gamma_input_1.shape[0], gamma_input_2.shape[0]):
                log(f"[GAMMA] WARN external {gamma_np.shape} != expected {(gamma_input_1.shape[0], gamma_input_2.shape[0])}; using diffusion")
                gamma_np = compute_gamma_diffusion(gamma_input_1, gamma_input_2, knn_k=args.gamma_diff_knn_k,
                    n_components=min(args.gamma_diff_components, args.emb_dim - 1), multiscale_times=times_tuple,
                    epsilon=args.gamma_epsilon, sinkhorn_iters=args.gamma_sinkhorn_iters, seed=args.seed)
            else:
                log(f"[GAMMA] loaded EXTERNAL v12 coupling {gamma_np.shape} from {ext}")
        else:
            gamma_np = compute_gamma_diffusion(
                gamma_input_1, gamma_input_2,
                knn_k=args.gamma_diff_knn_k,
                n_components=min(args.gamma_diff_components, args.emb_dim - 1),
                multiscale_times=times_tuple,
                epsilon=args.gamma_epsilon,
                sinkhorn_iters=args.gamma_sinkhorn_iters,
                seed=args.seed,
            )
        gamma_p_1to2, gamma_p_2to1 = gamma_top_partners(
            gamma_np, k=args.gamma_k_partners,
            rel_threshold=args.gamma_rel_threshold,
        )
        log(f"[GAMMA] done in {time.time() - t0:.1f}s")

    # ---- Anchor (ONCE, never refreshed) ----
    anchor_t = None
    anchor_mode = "off"
    if args.w_anchor > 0:
        anchor_np, anchor_mode = precompute_anchor(
            X1p, X2p, bc1, bc2, args.emb_dim,
            k_intra=args.anchor_k_intra, k_inter=args.anchor_k_inter,
            seed=args.seed,
            gamma_partners_1to2=gamma_p_1to2,
            gamma_partners_2to1=gamma_p_2to1,
        )
        anchor_t = torch.from_numpy(anchor_np).to(device)

    # ---- model20: pseudo-clusters from the fixed anchor; no labels used ----
    cluster_labels_np = None
    cluster_labels_t = None
    n_pseudo_clusters = 0
    if anchor_t is not None and args.pseudo_k > 0:
        cluster_labels_np = kmeans_labels_from_anchor(anchor_np, args.pseudo_k, args.seed)
        n_pseudo_clusters = int(cluster_labels_np.max()) + 1
        cluster_labels_t = torch.from_numpy(cluster_labels_np).to(device)
        counts = np.bincount(cluster_labels_np, minlength=n_pseudo_clusters)
        log(f"[PSEUDO] anchor k-means clusters: k={n_pseudo_clusters} counts={counts.tolist()}")

    # ---- model20: initial high-confidence reciprocal Γ pairs from PCA space ----
    gamma_pair_a = gamma_pair_b = None
    n_gamma_pairs_init = 0
    if (gamma_np is not None and args.w_gamma_pairs > 0 and cluster_labels_np is not None):
        gamma_pair_a, gamma_pair_b, n_gamma_pairs_init = gamma_reciprocal_pairs(
            gamma_np, cluster_labels_np=cluster_labels_np, n1=n1,
            top_q=args.gamma_pair_top_q, max_pairs=args.gamma_pair_max_pairs,
            seed=args.seed, device=device,
        )
        log(f"[GAMMA-PAIR] initial reciprocal same-pseudo-cluster pairs = {n_gamma_pairs_init}")

    # ---- Distillation correspondence (paired only, fixed, never refreshed) ----
    distill_active = is_paired and args.w_distill > 0
    corr_i = corr_j = None
    n_corr = 0
    teacher_mod = 0
    if distill_active:
        corr_i, corr_j, n_corr = get_barcode_correspondence(bc1, bc2, device)
        # Auto: larger modality teaches
        if args.teacher_modality == 0:
            teacher_mod = 1 if n1 >= n2 else 2
        else:
            teacher_mod = args.teacher_modality
        log(f"[DISTILL] paired distillation: {n_corr} barcode pairs, "
            f"teacher=mod{teacher_mod}")
    else:
        if args.w_distill > 0:
            log(f"[DISTILL] OFF (cross-protocol; would require z-MNN "
                f"correspondence which causes feedback loops -- removed)")

    # ---- Encoders / decoders / optimizer ----
    enc1 = ModalityEncoder(X1p.shape[1], args.mlp_hidden,
                           args.emb_dim, args.dropout).to(device)
    enc2 = ModalityEncoder(X2p.shape[1], args.mlp_hidden,
                           args.emb_dim, args.dropout).to(device)
    dec1 = ModalityDecoder(args.emb_dim, list(reversed(args.mlp_hidden)),
                           X1p.shape[1], args.dropout).to(device)
    dec2 = ModalityDecoder(args.emb_dim, list(reversed(args.mlp_hidden)),
                           X2p.shape[1], args.dropout).to(device)

    # model20: modality domain classifier (supervised by modality only)
    domain_clf = None
    domain_labels_t = None
    domain_weights_t = None
    if args.w_dann > 0:
        domain_clf = DomainClassifier(args.emb_dim, args.domain_hidden, args.dropout).to(device)
        domain_labels_t = torch.cat([
            torch.zeros(n1, dtype=torch.long),
            torch.ones(n2, dtype=torch.long),
        ]).to(device)
        # class-balanced CE, important because Bladder has 2375 vs 1287 cells
        w0 = (n1 + n2) / (2.0 * max(1, n1))
        w1 = (n1 + n2) / (2.0 * max(1, n2))
        domain_weights_t = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        log(f"[DANN] ON: class weights mod1={w0:.3f}, mod2={w1:.3f}")

    params = (list(enc1.parameters()) + list(enc2.parameters())
              + list(dec1.parameters()) + list(dec2.parameters()))
    if domain_clf is not None:
        params = params + list(domain_clf.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    log("=" * 70)
    log(f"[CONFIG] mode={'PAIRED' if is_paired else 'CROSS-PROTOCOL'}  "
        f"anchor_mode={anchor_mode}  distill={'ON' if distill_active else 'OFF'}  "
        f"batch_mmd={'ON' if multi_batch_active else 'OFF'}")
    log(f"[CONFIG] weights: recon={args.w_recon}  mmd={args.w_mmd}  "
        f"anchor={args.w_anchor}  mnn={args.w_mnn}  "
        f"var={args.w_var}  cov={args.w_cov}  "
        f"batch_mmd={args.w_batch_mmd if multi_batch_active else 0.0}  "
        f"distill={args.w_distill if distill_active else 0.0}  "
        f"cluster_centroid={args.w_cluster_centroid_align}  "
        f"cluster_mmd={args.w_cluster_mmd}  anchor_cons={args.w_anchor_consolidate}  "
        f"dann={args.w_dann}  "
        f"gamma_pairs={args.w_gamma_pairs}  dyn_gamma={args.w_dynamic_gamma}")
    log("=" * 70)

    # ---- Training loop (single-stage, no refresh) ----
    history: List[dict] = []
    mnn_a = mnn_b = None
    mnn_last_refresh = -1
    dyn_gamma_a = dyn_gamma_b = None
    dyn_gamma_last_refresh = -1
    n_dyn_gamma = 0
    t_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        enc1.train(); enc2.train(); dec1.train(); dec2.train()
        z1 = enc1(X1_t)
        z2 = enc2(X2_t)
        z = torch.cat([z1, z2], dim=0)

        # 1. Reconstruction
        x1_hat = dec1(z1)
        x2_hat = dec2(z2)
        l_recon = 0.5 * (F.mse_loss(x1_hat, X1_t) + F.mse_loss(x2_hat, X2_t))

        # 2. Cross-modal MMD
        l_mmd = mmd_loss(z1, z2, n_sample=args.mmd_n_sample)

        # 3. MNN attraction (refresh periodically; warmup before activating)
        if (args.w_mnn > 0
                and epoch > args.mnn_warmup_epochs
                and (mnn_last_refresh < 0
                     or (epoch - mnn_last_refresh) >= args.mnn_refresh_every)):
            with torch.no_grad():
                mnn_a, mnn_b, n_pairs = compute_mnn_pairs(
                    z.detach(), n1, args.mnn_k,
                )
            mnn_last_refresh = epoch
            log(f"[MNN] ep {epoch}: refreshed pairs = {n_pairs}")
        l_mnn = z.new_tensor(0.0)
        n_mnn = 0
        if (mnn_a is not None and args.w_mnn > 0 and mnn_a.numel() > 0):
            l_mnn = mnn_attraction_loss(z, mnn_a, mnn_b)
            n_mnn = int(mnn_a.numel())

        # 4 & 5. VICReg anti-collapse
        l_var = variance_loss(z, args.var_target)
        l_cov = covariance_loss(z)

        # 6. Anchor MSE (target is FIXED, never refreshed)
        l_anchor = z.new_tensor(0.0)
        if anchor_t is not None and args.w_anchor > 0:
            l_anchor = F.mse_loss(z, anchor_t)

        # 7. Multi-batch MMD (paired multi-batch only)
        l_batch_mmd = z.new_tensor(0.0)
        n_batch_pairs = 0
        if multi_batch_active:
            l_batch_mmd, n_batch_pairs = multi_batch_mmd(
                z, batch_int_t,
                min_per_batch=args.batch_mmd_min_per_batch,
                n_sample=args.batch_mmd_n_sample,
            )

        # 8. Distillation (paired only, fixed barcode correspondence)
        l_distill = z.new_tensor(0.0)
        if distill_active and epoch > args.distill_warmup_epochs:
            l_distill = barcode_distillation_loss(
                z, n1, corr_i, corr_j, teacher_modality=teacher_mod,
            )

        # 9. model20 pseudo-cluster conditional centroid alignment
        l_cluster_centroid = z.new_tensor(0.0)
        n_cluster_centroid = 0
        if (cluster_labels_t is not None and args.w_cluster_centroid_align > 0
                and epoch > args.cluster_warmup_epochs):
            l_cluster_centroid, n_cluster_centroid = cluster_centroid_alignment_loss(
                z, n1, cluster_labels_t, n_pseudo_clusters,
                min_per_modality=args.cluster_min_per_modality,
            )

        # 10. model20 pseudo-cluster conditional MMD
        l_cluster_mmd = z.new_tensor(0.0)
        n_cluster_mmd = 0
        if (cluster_labels_t is not None and args.w_cluster_mmd > 0
                and epoch > args.cluster_warmup_epochs):
            l_cluster_mmd, n_cluster_mmd = cluster_conditional_mmd_loss(
                z, n1, cluster_labels_t, n_pseudo_clusters,
                min_per_modality=args.cluster_min_per_modality,
                n_sample=args.cluster_mmd_n_sample,
            )

        # 11. model20_v2_1 anchor-consolidation: label-free cluster tightening
        l_anchor_consolidate = z.new_tensor(0.0)
        n_anchor_consolidate = 0
        if (cluster_labels_t is not None and args.w_anchor_consolidate > 0
                and epoch > args.anchor_consolidate_warmup_epochs):
            l_anchor_consolidate, n_anchor_consolidate = anchor_consolidation_loss(
                z, cluster_labels_t, n_pseudo_clusters,
                min_cluster_size=args.anchor_consolidate_min_cluster_size,
            )

        # 12. model20 initial PCA-space reciprocal Γ pair attraction
        l_gamma_pairs = z.new_tensor(0.0)
        n_gamma_pairs = 0
        if (gamma_pair_a is not None and gamma_pair_a.numel() > 0
                and args.w_gamma_pairs > 0
                and epoch > args.gamma_pair_warmup_epochs):
            l_gamma_pairs = mnn_attraction_loss(z, gamma_pair_a, gamma_pair_b)
            n_gamma_pairs = int(gamma_pair_a.numel())

        # 13. model20 dynamic z-space Γ pair attraction; weak and guarded
        if (args.w_dynamic_gamma > 0 and cluster_labels_np is not None
                and epoch > args.dynamic_gamma_warmup_epochs
                and (dyn_gamma_last_refresh < 0
                     or (epoch - dyn_gamma_last_refresh) >= args.dynamic_gamma_every)):
            with torch.no_grad():
                dyn_gamma_a, dyn_gamma_b, n_dyn_gamma = dynamic_gamma_pairs_from_z(
                    z.detach(), n1, cluster_labels_np=cluster_labels_np,
                    epsilon=args.dynamic_gamma_epsilon,
                    top_q=args.dynamic_gamma_top_q,
                    max_pairs=args.dynamic_gamma_max_pairs,
                    seed=args.seed + epoch,
                )
            dyn_gamma_last_refresh = epoch
            log(f"[DYN-GAMMA] ep {epoch}: reciprocal same-pseudo-cluster pairs = {n_dyn_gamma}")
        l_dynamic_gamma = z.new_tensor(0.0)
        if dyn_gamma_a is not None and dyn_gamma_a.numel() > 0 and args.w_dynamic_gamma > 0:
            l_dynamic_gamma = mnn_attraction_loss(z, dyn_gamma_a, dyn_gamma_b)

        # 14. model20 modality adversarial loss (supervised by modality only)
        l_dann = z.new_tensor(0.0)
        adv_alpha = 0.0
        if domain_clf is not None and args.w_dann > 0:
            adv_alpha = dann_alpha(epoch, args.dann_warmup_epochs,
                                   args.max_epochs, args.dann_lambda_max)
            if adv_alpha > 0:
                logits_domain = domain_clf(grad_reverse(z, adv_alpha))
                l_dann = F.cross_entropy(logits_domain, domain_labels_t,
                                         weight=domain_weights_t)

        # Assemble total loss; conditional weights gated at runtime
        w_bm = args.w_batch_mmd if multi_batch_active else 0.0
        w_d = args.w_distill if distill_active else 0.0
        loss = (args.w_recon * l_recon
                + args.w_mmd * l_mmd
                + args.w_mnn * l_mnn
                + args.w_var * l_var
                + args.w_cov * l_cov
                + args.w_anchor * l_anchor
                + w_bm * l_batch_mmd
                + w_d * l_distill
                + args.w_cluster_centroid_align * l_cluster_centroid
                + args.w_cluster_mmd * l_cluster_mmd
                + args.w_anchor_consolidate * l_anchor_consolidate
                + args.w_gamma_pairs * l_gamma_pairs
                + args.w_dynamic_gamma * l_dynamic_gamma
                + args.w_dann * l_dann)

        opt.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        opt.step()

        z_std = z.detach().std(dim=0)
        z_std_min = float(z_std.min().item())
        z_std_mean = float(z_std.mean().item())

        history.append({
            "epoch": epoch,
            "total": float(loss.item()),
            "recon": float(l_recon.item()),
            "mmd": float(l_mmd.item()),
            "mnn": float(l_mnn.item()),
            "var": float(l_var.item()),
            "cov": float(l_cov.item()),
            "anchor": float(l_anchor.item()),
            "batch_mmd": float(l_batch_mmd.item()),
            "distill": float(l_distill.item()),
            "cluster_centroid": float(l_cluster_centroid.item()),
            "cluster_mmd": float(l_cluster_mmd.item()),
            "anchor_consolidate": float(l_anchor_consolidate.item()),
            "gamma_pairs": float(l_gamma_pairs.item()),
            "dynamic_gamma": float(l_dynamic_gamma.item()),
            "dann": float(l_dann.item()),
            "adv_alpha": float(adv_alpha),
            "n_cluster_centroid": n_cluster_centroid,
            "n_cluster_mmd": n_cluster_mmd,
            "n_anchor_consolidate": n_anchor_consolidate,
            "n_gamma_pairs": n_gamma_pairs,
            "n_dynamic_gamma_pairs": n_dyn_gamma,
            "mnn_n_pairs": n_mnn,
            "n_batch_pairs": n_batch_pairs,
            "z_std_min": z_std_min,
            "z_std_mean": z_std_mean,
        })

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.max_epochs:
            log(f"[ep {epoch:4d}] total={loss.item():.4f}  "
                f"recon={l_recon.item():.4f}  mmd={l_mmd.item():.4f}  "
                f"anchor={l_anchor.item():.4f}  mnn={l_mnn.item():.4f}  "
                f"cc={l_cluster_centroid.item():.4f}  cmmd={l_cluster_mmd.item():.4f}  "
                f"acon={l_anchor_consolidate.item():.4f}  "
                f"gp={l_gamma_pairs.item():.4f}  dgp={l_dynamic_gamma.item():.4f}  "
                f"dann={l_dann.item():.4f}  a={adv_alpha:.2f}  "
                f"z_std_min={z_std_min:.3f}")

    log(f"[TRAIN] done in {time.time() - t_start:.1f}s")

    # ---- Save embeddings ----
    enc1.eval(); enc2.eval()
    with torch.no_grad():
        z1 = enc1(X1_t).cpu().numpy()
        z2 = enc2(X2_t).cpu().numpy()
    z_all = np.vstack([z1, z2])
    df_emb = pd.DataFrame(
        z_all, columns=[f"emb_{i}" for i in range(z_all.shape[1])],
    )
    df_emb.insert(0, "cell_id", np.concatenate([bc1, bc2]))
    df_emb.insert(1, "modality",
                  np.concatenate([np.full(n1, "mod1"), np.full(n2, "mod2")]))
    df_emb.insert(2, "reliable", 1)
    if cluster_labels_np is not None and len(cluster_labels_np) == z_all.shape[0]:
        df_emb.insert(3, "pseudo_cluster", cluster_labels_np)
    emb_path = args.out_prefix + "_embeddings.csv"
    df_emb.to_csv(emb_path, index=False)
    log(f"[SAVE] {emb_path}")

    # ---- Save history ----
    df_hist = pd.DataFrame(history)
    hist_path = args.out_prefix + "_history.csv"
    df_hist.to_csv(hist_path, index=False)
    log(f"[SAVE] {hist_path}")

    # ---- Save model ----
    model_path = args.out_prefix + "_model.pt"
    torch.save({
        "enc1": enc1.state_dict(),
        "enc2": enc2.state_dict(),
        "dec1": dec1.state_dict(),
        "dec2": dec2.state_dict(),
        "args": vars(args),
        "anchor_mode": anchor_mode,
        "is_paired": bool(is_paired),
        "distill_active": bool(distill_active),
        "multi_batch_active": bool(multi_batch_active),
        "n_pseudo_clusters": int(n_pseudo_clusters),
        "n_gamma_pairs_init": int(n_gamma_pairs_init),
        "model20_note": "model20_v2_1: fixed anchor + pseudo-cluster conditional mixing + anchor consolidation + weak dynamic gamma",
    }, model_path)
    log(f"[SAVE] {model_path}")
    log(f"[DONE] {args.dataset_id}")


if __name__ == "__main__":
    main()
