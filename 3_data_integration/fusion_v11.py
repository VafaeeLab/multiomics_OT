#!/usr/bin/env python3
"""DI-ElasticFlow (v11): coupling-guided elastic displacement.

WHAT THE DIAGNOSTICS ESTABLISHED

  1. The couplings are accurate. Top-1 same-cell-type rate is 0.70-0.80 on every
     diagonal modality pair (lift 2.3-5.3x over chance). The information needed to
     align the blocks is present.
  2. No rigid or piecewise-rigid map can use it. Sweeping 8 transforms x 4 alphas
     with the guards switched OFF, the best achievable mixing gain on rna-rna under
     ensemble/diffusion was +0.008 against ~0.35 of headroom.
  3. Every previous version chose alpha by MIXING ALONE. That is why NMI/ARI fell
     each time: mixing rises whenever the blocks overlap more, whether or not the
     right cells overlap.
  4. Loosening the transform without fixing (3) just buys iLISI and sells NMI --
     v5 affine (iLISI +0.103, NMI -0.039), v10 piecewise (rna-rna iLISI +0.025,
     NMI -0.021).

THE MECHANISM THAT WAS MISSING

  Moving the query block as a body can only ever trade mixing against structure.
  What raises all four metrics at once is moving each query cell toward reference
  cells OF ITS OWN TYPE: same-type cells from the two modalities co-locate, so the
  joint clusters get both purer (NMI/ARI/silhouette up) and better mixed (iLISI up).
  The coupling estimates that, label-free, at ~80% accuracy.

  Doing it per cell -- z_q <- barycentre of its partners -- collapses the query
  block onto the reference, which is why every preservation floor vetoed it.

  The resolution is to require the displacement field to be SMOOTH: neighbouring
  query cells must move together. Then each cell's displacement is effectively an
  average over its neighbourhood's targets, so the coupling's ~20% wrong partners
  are averaged away (spatial denoising), and local structure survives because
  neighbours share a motion.

THE OPTIMISATION

  With U the displacement of the query block, B the coupling barycentres and L the
  graph Laplacian of the query kNN graph,

      min_U  || (Z + U) - B ||^2  +  lambda * tr( U^T L U )

  which is quadratic, so it is solved exactly by one sparse SPD system

      (I + lambda L) U = B - Z

  lambda sets the stiffness across the whole spectrum:
      lambda -> 0    U = B - Z         pure barycentric projection (collapse)
      lambda -> inf  U -> constant     global translation (constants span ker L)
  so the useful middle -- a smooth warp that is locally near-rigid but globally
  flexible -- is reachable by sweeping one scalar, not by guessing a transform.

  Note sum_r w_r ||z - z_r||^2 = ||z - b||^2 + const, so pulling toward the
  barycentre IS the coupling-weighted attraction, exactly.

SELECTION (label-free, two terms)

  A candidate is scored on BOTH objectives, each as the fraction of its available
  headroom captured:

      J = d_mixing / (ideal_mixing - base_mixing)  +  d_structure / (1 - base_structure)

  structure is a simplified silhouette against centroids of an unsupervised
  clustering of the REFERENCE block -- it rewards query cells landing decisively
  inside a reference cluster rather than between clusters. It is a label-free
  stand-in for NMI/ARI/silhouette, and its presence is what stops the method from
  selecting an iLISI-for-NMI trade.

  Guards: mixing must improve, and query-neighbourhood preservation must clear a
  LOOSE floor (0.5) -- loose on purpose, because a smooth field preserves
  structure by construction and a tight floor is what made v10 decline moves that
  v6 accepted.

  v6's rigid transform is included as a candidate, so the search space is a strict
  superset of v6's.

No cell-type labels are read anywhere: clustering is unsupervised, and the
barycentres come from the coupling alone.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, cg
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

LAMBDAS = (1.0, 10.0, 100.0, 1000.0)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
K_SMOOTH = 15           # query kNN graph for the roughness penalty
K_COUP = 10             # coupling partners per query cell
K_MIX = 15              # neighbourhood size for the mixing measure
N_REF_CLUSTERS = 15     # unsupervised reference clusters for the structure proxy
PRES_FLOOR = 0.50       # loose: a smooth field preserves structure by construction
EVAL_CAP = 4000
MIN_ANCHORS = 10
CHUNK = 512


# ----------------------------------------------------------------- coupling
def barycentres(Z, i1, G, topk=K_COUP):
    """Coupling-weighted mean position of each query cell's reference partners."""
    n1, n2 = G.shape
    k = int(min(topk, n1))
    Zr = Z[i1]
    B = np.zeros((n2, Z.shape[1]))
    for s in range(0, n2, CHUNK):
        e = min(s + CHUNK, n2)
        C = np.asarray(G[:, s:e], dtype=np.float64)
        top = np.argpartition(-C, k - 1, axis=0)[:k]          # k x chunk
        W = np.take_along_axis(C, top, axis=0)
        ws = W.sum(0)
        bad = ws <= 0
        if bad.any():
            W[:, bad] = 1.0
            ws[bad] = float(k)
        W /= ws
        # (k x chunk x d) contraction without materialising the cube
        for j in range(e - s):
            B[s + j] = W[:, j] @ Zr[top[:, j]]
    return B


# ----------------------------------------------------------------- v6 candidate
def rigid_target(Z, i1, i2, G, k=5, cap=4000, seed=0):
    """v6's transform, so the candidate set is a superset of v6."""
    n1, n2 = G.shape
    r = np.random.RandomState(seed)
    s1 = np.arange(n1) if n1 <= cap else np.sort(r.choice(n1, cap, replace=False))
    s2 = np.arange(n2) if n2 <= cap else np.sort(r.choice(n2, cap, replace=False))
    S = np.asarray(G[np.ix_(s1, s2)], float)
    if S.size == 0:
        return None
    kk = int(max(1, min(k, S.shape[0] - 1, S.shape[1] - 1)))
    top_c = np.argpartition(-S, kk, axis=0)[:kk]
    best_r = S.argmax(1)
    pairs = [(s1[r_], s2[c]) for c in range(S.shape[1]) for r_ in top_c[:, c]
             if best_r[r_] == c]
    if len(pairs) < MIN_ANCHORS:
        a = S.argmax(0)
        pairs = [(s1[a[c]], s2[c]) for c in range(S.shape[1])]
    if len(pairs) < MIN_ANCHORS:
        return None
    p = np.array(pairs)
    A, Bm = Z[i1][p[:, 0]], Z[i2][p[:, 1]]
    ca, cb = A.mean(0), Bm.mean(0)
    H = (Bm - cb).T @ (A - ca)
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    D = np.eye(H.shape[0])
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    return (Z[i2] - cb) @ (U @ D @ Vt) + ca


# ----------------------------------------------------------------- elastic solve
def query_laplacian(Zq, k=K_SMOOTH):
    n = len(Zq)
    k = int(min(k, n - 1))
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Zq).kneighbors(
        return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)                                   # symmetrise
    d = np.asarray(W.sum(1)).ravel()
    return sp.diags(d) - W


def elastic_displacement(T, L, lam, direct_max=3000):
    """Solve (I + lam L) U = T.

    SPD system -> conjugate gradient. A sparse direct factorisation of a kNN
    Laplacian suffers severe fill-in: 76.5s per solve at 8.6k cells versus 0.68s
    for CG (max|diff| 3e-08). Direct is retained for small blocks only.
    """
    n = T.shape[0]
    A = (sp.eye(n, format="csc") + lam * L).tocsc()
    if n <= direct_max:
        solve = factorized(A)
        return np.column_stack([solve(T[:, j]) for j in range(T.shape[1])])
    out = np.zeros_like(T)
    for j in range(T.shape[1]):
        x, _ = cg(A, T[:, j], rtol=1e-6, maxiter=500)
        out[:, j] = x
    return out


# ----------------------------------------------------------------- objectives
def mixing(Z, is2, k=K_MIX):
    k = int(min(k, len(Z) - 1))
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(
        Z, return_distance=False)[:, 1:]
    return float(1 - (is2[idx] == is2[:, None]).mean())


def structure(Zq, C):
    """Simplified silhouette of query cells against fixed reference centroids.

    High when a query cell sits decisively inside one reference cluster rather
    than between clusters -- a label-free stand-in for NMI/ARI/silhouette.
    """
    d = np.linalg.norm(Zq[:, None, :] - C[None, :, :], axis=2)
    if d.shape[1] < 2:
        return 0.0
    part = np.partition(d, 1, axis=1)
    a, b = part[:, 0], part[:, 1]
    return float(np.mean((b - a) / np.maximum(np.maximum(a, b), 1e-12)))


def preservation(Zb, Zf, k=K_MIX):
    if len(Zb) < k + 2:
        return 1.0
    a = NearestNeighbors(n_neighbors=k + 1).fit(Zb).kneighbors(
        return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Zf).kneighbors(
        return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


# ----------------------------------------------------------------- driver
def fuse(Z, i1, i2, G, alphas=ALPHAS, lambdas=LAMBDAS, k=K_MIX,
         pres_floor=PRES_FLOOR, **kw):
    Z = np.asarray(Z, float)
    i1 = np.asarray(i1); i2 = np.asarray(i2)
    info = {}
    n1, n2 = G.shape
    if n1 < MIN_ANCHORS or n2 < MIN_ANCHORS or len(i1) != n1 or len(i2) != n2:
        info["reason"] = "block/coupling size mismatch"
        return Z, info

    Zq = Z[i2]

    # evaluation subsample (fixed across candidates so scores are comparable)
    rs = np.random.RandomState(0)
    e1 = i1 if len(i1) <= EVAL_CAP // 2 else np.sort(rs.choice(i1, EVAL_CAP // 2, replace=False))
    e2 = i2 if len(i2) <= EVAL_CAP // 2 else np.sort(rs.choice(i2, EVAL_CAP // 2, replace=False))
    ev = np.concatenate([e1, e2])
    is2 = np.concatenate([np.zeros(len(e1), int), np.ones(len(e2), int)])
    q_ev = np.searchsorted(i2, e2)
    Zev = Z[ev]

    # reference clusters -> fixed centroids (reference cells never move)
    nk = int(max(2, min(N_REF_CLUSTERS, len(e1) // 20)))
    try:
        km = KMeans(n_clusters=nk, n_init=4, random_state=0).fit(Z[e1])
        C = km.cluster_centers_
    except Exception as e:
        info["cluster_failed"] = str(e)[:40]
        C = Z[e1][:nk]

    base_mix = mixing(Zev, is2, k)
    base_str = structure(Z[e2], C)
    ideal_mix = 2.0 * len(e1) * len(e2) / (len(e1) + len(e2)) ** 2
    mix_head = max(ideal_mix - base_mix, 1e-6)
    str_head = max(1.0 - base_str, 1e-6)
    info.update(base_mix=base_mix, base_structure=base_str)

    candidates = []          # (name, query_target_positions)

    # candidate: v6 rigid
    rt = rigid_target(Z, i1, i2, G)
    if rt is not None:
        candidates.append(("v6_rigid", rt))

    # candidates: elastic flow at each stiffness, from the baseline AND from the
    # rigid pre-alignment (coarse-to-fine)
    try:
        L = query_laplacian(Zq)
        starts = [("", Zq)]
        if rt is not None:
            starts.append(("rigid+", rt))
        for tag, Z0 in starts:
            B = barycentres(np.concatenate([Z[i1], Z0]),
                            np.arange(len(i1)), G)
            T = B - Z0
            for lam in lambdas:
                try:
                    U = elastic_displacement(T, L, lam)
                except Exception as e:
                    info[f"solve_failed_{tag}{lam}"] = str(e)[:30]
                    continue
                candidates.append((f"{tag}elastic_lam{lam:g}", Z0 + U))
    except Exception as e:
        info["elastic_failed"] = str(e)[:60]

    if not candidates:
        info["reason"] = "no candidate"
        return Z, info
    info["n_candidates"] = len(candidates)

    best_Z, best_J, best = Z, 0.0, ("none", 0.0, base_mix, base_str, 1.0)
    for name, tgt in candidates:
        for al in alphas:
            Qe = (1 - al) * Z[e2] + al * tgt[q_ev]
            Zf_ev = Zev.copy()
            Zf_ev[len(e1):] = Qe
            m = mixing(Zf_ev, is2, k)
            if m <= base_mix:
                continue
            p = preservation(Z[e2], Qe, k)
            if p < pres_floor:
                continue
            s = structure(Qe, C)
            J = (m - base_mix) / mix_head + (s - base_str) / str_head
            if J > best_J:
                Zf = Z.copy()
                Zf[i2] = (1 - al) * Zq + al * tgt
                best_Z, best_J, best = Zf, J, (name, al, m, s, p)
    info.update(candidate=best[0], alpha=best[1], mix_after=best[2],
                structure_after=best[3], preservation=best[4], score=best_J)
    return best_Z, info
