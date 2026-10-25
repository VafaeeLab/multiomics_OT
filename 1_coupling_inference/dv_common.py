#!/usr/bin/env python3
"""
dv_common.py — SELF-CONTAINED shared utilities for diffusion correspondence models.
Embeds the matrix loaders (no dependency on any model file, to avoid clobber/circular
imports). Shared-gene bridge (ADT remap + HVG/XINF/TF-IDF/rank), within-modality
diffusion transition + joint-graph helpers, dense cosine affinity, (unbalanced) entropic
OT, dense mid-rank FOSCTTM scorer, and label-transfer scoring. numpy/scipy/sklearn only.
"""
from __future__ import annotations
import importlib.util, sys, subprocess, tempfile, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

NF = Path(__file__).resolve().parent
PROJ = NF.parent
LT = PROJ / "label_transfer_models"
if not (LT / "remap_adt_to_gene.py").exists():
    LT = PROJ / "4_label_transfer"  # dir renamed; remap lives here
PSF_PATH = PROJ / "1_coupling_inference" / "plot_sorted_foscttm.py"


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

PSF = _imp("plot_sorted_foscttm", PSF_PATH)
try:
    REMAP = _imp("remap_adt_to_gene", LT / "remap_adt_to_gene.py")
except Exception:
    REMAP = None


# ---------------- matrix loaders (embedded; self-contained) ----------------
def read_names(path):
    names = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                names.append(line.split("\t")[0])
    return np.array(names, dtype=str)


def auto_sidecars(base, candidates):
    for c in candidates:
        p = Path(str(base) + c)
        if p.exists():
            return str(p)
    return None


def load_npz(path):
    path = Path(path); d = np.load(path, allow_pickle=True); keys = set(d.files)
    if {"data", "indices", "indptr", "shape"}.issubset(keys):
        shape = tuple(np.asarray(d["shape"], dtype=np.int64).tolist())
        data, indices, indptr = d["data"], d["indices"], d["indptr"]
        if len(indptr) == shape[0] + 1:
            X = sp.csr_matrix((data, indices, indptr), shape=shape)
        elif len(indptr) == shape[1] + 1:
            X = sp.csc_matrix((data, indices, indptr), shape=shape).tocsr()
        else:
            raise ValueError(f"Cannot infer sparse format for {path}")
    elif "X" in keys:
        X = sp.csr_matrix(d["X"])
    elif "matrix" in keys:
        X = sp.csr_matrix(d["matrix"])
    else:
        raise ValueError(f"Unsupported NPZ keys in {path}: {list(d.files)}")
    rn = d["row_names"].astype(str) if "row_names" in keys else np.arange(X.shape[0]).astype(str)
    cn = d["col_names"].astype(str) if "col_names" in keys else np.arange(X.shape[1]).astype(str)
    return X, rn, cn


def load_mtx(path, barcodes=None, features=None):
    path = Path(path); X = sio.mmread(str(path)).tocsr(); n_rows, n_cols = X.shape
    if barcodes is None:
        barcodes = auto_sidecars(path.with_suffix(""),
            ["_barcodes.tsv", "_barcodes.tsv.gz", ".barcodes.tsv", ".barcodes.tsv.gz"])
    if features is None:
        features = auto_sidecars(path.with_suffix(""),
            ["_features.tsv", "_features.tsv.gz", ".features.tsv", ".features.tsv.gz", "_genes.tsv", ".genes.tsv"])
    rn = read_names(barcodes) if barcodes and Path(barcodes).exists() else np.arange(n_rows).astype(str)
    cn = read_names(features) if features and Path(features).exists() else np.arange(n_cols).astype(str)
    return X, rn, cn


def load_rds(path, log_fn=None):
    path = Path(path)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td); x_bin = td_p / "x.bin"; i_bin = td_p / "i.bin"; p_bin = td_p / "p.bin"
        meta_f = td_p / "meta.txt"; rn_f = td_p / "rownames.txt"; cn_f = td_p / "colnames.txt"
        rscript = f"""
suppressMessages(library(Matrix))
m <- readRDS('{path}')
if (!inherits(m, 'sparseMatrix')) m <- as(m, 'CsparseMatrix')
writeBin(as.double(m@x),  '{x_bin}', size = 4, endian = 'little')
writeBin(as.integer(m@i), '{i_bin}', size = 4, endian = 'little')
writeBin(as.integer(m@p), '{p_bin}', size = 4, endian = 'little')
cat(m@Dim[1], m@Dim[2], length(m@x), file = '{meta_f}', sep = ' ')
rn <- rownames(m); if (is.null(rn)) rn <- as.character(seq_len(m@Dim[1]))
cn <- colnames(m); if (is.null(cn)) cn <- as.character(seq_len(m@Dim[2]))
writeLines(rn, '{rn_f}'); writeLines(cn, '{cn_f}')
rm(m); invisible(gc(verbose = FALSE))
"""
        res = subprocess.run(["Rscript", "--vanilla", "-"], input=rscript, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Rscript failed for {path}\nSTDERR:\n{res.stderr}")
        meta = meta_f.read_text().strip().split()
        nrow, ncol, nnz = int(meta[0]), int(meta[1]), int(meta[2])
        x = np.fromfile(str(x_bin), dtype=np.float32, count=nnz)
        i = np.fromfile(str(i_bin), dtype=np.int32, count=nnz)
        p = np.fromfile(str(p_bin), dtype=np.int32, count=ncol + 1)
        X_csc = sp.csc_matrix((x, i, p), shape=(nrow, ncol), copy=False); del x, i, p; gc.collect()
        X = X_csc.tocsr(); del X_csc; gc.collect()
        rn = np.array(rn_f.read_text().splitlines(), dtype=str)
        cn = np.array(cn_f.read_text().splitlines(), dtype=str)
    return X, rn, cn


def load_matrix(path, barcodes=None, features=None, log_fn=None):
    path = Path(path)
    if path.name.endswith(".npz"):
        X, rn, cn = load_npz(path)
    elif path.name.endswith(".mtx") or path.name.endswith(".mtx.gz"):
        X, rn, cn = load_mtx(path, barcodes=barcodes, features=features)
    elif path.name.endswith(".rds"):
        X, rn, cn = load_rds(path, log_fn=log_fn)
    else:
        raise ValueError(f"Unsupported format: {path}")
    if log_fn:
        log_fn(f"  loaded {path.name}: shape={X.shape}")
    return {"path": str(path), "X": X, "row_names": rn, "col_names": cn}


def get_oriented(record, orientation):
    X, rn, cn = record["X"], record["row_names"], record["col_names"]
    if orientation == "cell_by_feature":
        return X.tocsr(), rn, cn
    if orientation == "feature_by_cell":
        return X.T.tocsr(), cn, rn
    raise ValueError("orientation must be cell_by_feature or feature_by_cell")


def align_common_cells(X1, cells1, X2, cells2):
    pos2 = {c: i for i, c in enumerate(cells2)}; pos1 = {c: i for i, c in enumerate(cells1)}
    common = [c for c in cells1 if c in pos2]
    if not common:
        raise ValueError("No common cells found between modalities.")
    idx1 = np.array([pos1[c] for c in common]); idx2 = np.array([pos2[c] for c in common])
    return X1[idx1, :].tocsr(), X2[idx2, :].tocsr(), np.array(common, dtype=str)


def sparse_log1p(X):
    X = X.copy().tocsr()
    if X.nnz > 0 and np.nanmin(X.data) >= 0:
        X.data = np.log1p(X.data)
    return X


def row_l1_normalize(X):
    X = X.tocsr(copy=True); s = np.asarray(X.sum(axis=1)).ravel(); s[s == 0] = 1.0
    return sp.diags(1.0 / s).dot(X).tocsr()


# ---------------- shared-gene bridge + reps ----------------
def to_dense(X):
    return np.asarray(X.todense(), np.float64) if sp.issparse(X) else np.asarray(X, np.float64)


def zscore_cols(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-8)


def l2_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0; return M / n


def load_align_unpaired(rna, atac, ori1, ori2, log=print):
    """Load and orient both modalities WITHOUT intersecting cells.

    For a diagonal pair the two modalities profile different cells, so there is
    no shared barcode set to align on. Returns the cell lists separately; the
    caller must not assume X1 and X2 have the same number of rows."""
    r1 = load_matrix(rna, log_fn=log); r2 = load_matrix(atac, log_fn=log)
    X1, c1, f1 = get_oriented(r1, ori1); X2, c2, f2 = get_oriented(r2, ori2)
    return (X1.tocsr(), X2.tocsr(), np.array([str(x) for x in c1]),
            np.array([str(x) for x in c2]),
            np.array([str(x) for x in f1]), np.array([str(x) for x in f2]))


def load_align(rna, atac, ori1, ori2, log=print):
    r1 = load_matrix(rna, log_fn=log); r2 = load_matrix(atac, log_fn=log)
    X1, c1, f1 = get_oriented(r1, ori1); X2, c2, f2 = get_oriented(r2, ori2)
    X1, X2, cells = align_common_cells(X1, c1, X2, c2)
    return X1, X2, cells, np.array([str(x) for x in f1]), np.array([str(x) for x in f2])


def map_mod2_to_genes(f2, f1, mod2_kind):
    if mod2_kind == "adt" and REMAP is not None:
        recs, stats = REMAP.remap_dataset(list(f2), list(f1))
        f2m = np.array([(r["assigned"] if r.get("action") in ("alias", "gene_exact") else "__DROP__")
                        for r in recs], dtype=object)
        return f2m, stats
    return f2.astype(object), {}


def shared_block(X1, f1, X2, f2, mod2_kind, do_log1p=True,
                 gene_mode="all", tfidf=False, n_hvg=2000, rank=False, log=print):
    f2m, _ = map_mod2_to_genes(f2, f1, mod2_kind)
    set2 = set(map(str, f2m))
    common = [g for g in dict.fromkeys(f1) if str(g) in set2 and g not in ("", "nan", "None", "__DROP__")]
    i1 = {g: k for k, g in enumerate(f1)}; i2 = {}
    for k, g in enumerate(f2m):
        g = str(g)
        if g in set(common) and g not in i2:
            i2[g] = k
    common = [g for g in common if g in i2]
    if len(common) < 5:
        return None, None, common
    idx1 = np.array([i1[g] for g in common]); idx2 = np.array([i2[g] for g in common])
    S1 = to_dense(X1[:, idx1]); S2 = to_dense(X2[:, idx2])
    if do_log1p:
        S1 = np.log1p(np.clip(S1, 0, None)); S2 = np.log1p(np.clip(S2, 0, None))
    if gene_mode == "hvg" and S1.shape[1] > n_hvg:
        disp = S1.var(0) / (np.abs(S1.mean(0)) + 1e-8) + S2.var(0) / (np.abs(S2.mean(0)) + 1e-8)
        keep = np.argsort(disp)[::-1][:n_hvg]
        S1 = S1[:, keep]; S2 = S2[:, keep]; common = [common[k] for k in keep]
        log(f"  HVG: kept {len(common)} genes")
    elif gene_mode == "xinf" and S1.shape[1] > n_hvg:
        d1 = S1.var(0) / (np.abs(S1.mean(0)) + 1e-8); d2 = S2.var(0) / (np.abs(S2.mean(0)) + 1e-8)
        r1 = np.argsort(np.argsort(d1)); r2 = np.argsort(np.argsort(d2))
        keep = np.argsort(np.minimum(r1, r2))[::-1][:n_hvg]
        S1 = S1[:, keep]; S2 = S2[:, keep]; common = [common[k] for k in keep]
        log(f"  XINF: kept {len(common)} genes")
    if tfidf:
        det = 0.5 * ((S1 > 0).mean(0) + (S2 > 0).mean(0)) + 1e-6
        idf = np.log(1.0 + 1.0 / det); S1 = S1 * idf[None, :]; S2 = S2 * idf[None, :]
        log("  TF-IDF applied")
    if rank:
        from scipy.stats import rankdata
        S1 = np.apply_along_axis(rankdata, 0, S1); S2 = np.apply_along_axis(rankdata, 0, S2)
        log("  RANK-transform per gene")
    S1 = l2_rows(zscore_cols(S1)); S2 = l2_rows(zscore_cols(S2))
    log(f"  shared block: {len(common)} genes (gene_mode={gene_mode}, tfidf={tfidf}, rank={rank})")
    return S1, S2, common


def cosine_affinity(S1n, S2n):
    return S1n @ S2n.T


def within_transition(X, n_pcs, k, seed, do_log1p=True):
    Z = pca_dense(X, n_pcs, seed, do_log1p=do_log1p)
    W = knn_adjacency(Z, k)
    deg = np.asarray(W.sum(1)).ravel(); deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ W


def smooth_affinity(K, P1, P2, a):
    if a <= 0:
        return K
    Kl = K.copy()
    for _ in range(int(a)):
        Kl = P1 @ Kl
    Kr = Kl
    for _ in range(int(a)):
        Kr = Kr @ P2.T
    return Kr


def sinkhorn_plan(C, eps, iters=400):
    C = np.asarray(C, np.float64); scale = np.median(C) + 1e-12
    K = np.exp(-C / (scale * max(eps, 1e-3)))
    n1, n2 = K.shape; u = np.ones(n1); v = np.ones(n2)
    a = np.full(n1, 1.0 / n1); b = np.full(n2, 1.0 / n2)
    for _ in range(iters):
        u = a / (K @ v + 1e-300); v = b / (K.T @ u + 1e-300)
    return u[:, None] * K * v[None, :]


def pca_dense(X, n_comp, seed, do_log1p=True):
    Xp = sparse_log1p(X) if (do_log1p and sp.issparse(X)) else (X.tocsr() if sp.issparse(X) else X)
    if sp.issparse(Xp):
        Xp = row_l1_normalize(Xp)
    Xc = Xp if sp.issparse(Xp) else sp.csr_matrix(Xp)
    nc = int(max(2, min(n_comp, min(Xc.shape) - 1)))
    Z = TruncatedSVD(n_components=nc, random_state=seed).fit_transform(Xc)
    return ((Z - Z.mean(0)) / (Z.std(0) + 1e-8)).astype(np.float64)


def knn_adjacency(Z, k):
    n = Z.shape[0]; k = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z); dist = dist[:, 1:]; idx = idx[:, 1:]
    sigma = np.maximum(dist[:, -1], 1e-8)
    w = np.exp(-(dist ** 2) / (sigma[:, None] * sigma[idx] + 1e-12))
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    return W.maximum(W.T)


def mutual_inter(Za, Zb, k, alpha=1.0):
    na, nb = Za.shape[0], Zb.shape[0]; k = int(min(k, min(na, nb) - 1))
    dab, iab = NearestNeighbors(n_neighbors=k).fit(Zb).kneighbors(Za)
    dba, iba = NearestNeighbors(n_neighbors=k).fit(Za).kneighbors(Zb)
    setba = [set(r) for r in iba]; sig = np.median(dab) + 1e-8
    rows = []; cols = []; vals = []
    for i in range(na):
        for jj, j in enumerate(iab[i]):
            if i in setba[j]:
                rows.append(i); cols.append(j); vals.append(alpha * np.exp(-(dab[i, jj] ** 2) / (sig ** 2)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(na, nb))


def unbalanced_sinkhorn(C, eps, tau=1.0, iters=400):
    C = np.asarray(C, np.float64); scale = np.median(C) + 1e-12
    K = np.exp(-C / (scale * max(eps, 1e-3)))
    n1, n2 = K.shape; a = np.full(n1, 1.0 / n1); b = np.full(n2, 1.0 / n2)
    u = np.ones(n1); v = np.ones(n2); f = tau / (tau + 1.0)
    for _ in range(iters):
        u = (a / (K @ v + 1e-300)) ** f
        v = (b / (K.T @ u + 1e-300)) ** f
    G = u[:, None] * K * v[None, :]
    return G / (G.sum(1, keepdims=True) + 1e-12)


def score_dense(D, label=""):
    D = np.asarray(D, np.float64); n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        # Rectangular plan => diagonal pair => no true match to rank against.
        _nan = float("nan")
        out = {"label": label, "n": n, "foscttm_midrank": _nan,
               "foscttm_strict": _nan, "mrr": _nan, "median_rank": _nan,
               "hungarian": _nan, "paired": False}
        out.update({f"recall_at_{k}": _nan for k in (1, 5, 10, 50)})
        return out
    mid = PSF.metrics_from_matrix(D)["foscttm_sym"]
    foscttm_midrank = float(np.mean(mid))
    diag = np.diag(D)
    less_r = (D < diag[:, None]).sum(1); eq_r = (D == diag[:, None]).sum(1) - 1
    less_c = (D < diag[None, :]).sum(0); eq_c = (D == diag[None, :]).sum(0) - 1
    strict = float(0.5 * ((less_r / max(1, n - 1)).mean() + (less_c / max(1, n - 1)).mean()))
    rank_r = less_r + 0.5 * eq_r; rank_c = less_c + 0.5 * eq_c
    def rec(r, k): return float(np.mean(r < k))
    recall = {f"recall_at_{k}": 0.5 * (rec(rank_r, k) + rec(rank_c, k)) for k in (1, 5, 10, 50)}
    mrr = float(0.5 * (np.mean(1.0 / (rank_r + 1.0)) + np.mean(1.0 / (rank_c + 1.0))))
    median_rank = float(0.5 * (np.median(rank_r) + np.median(rank_c)))
    if n <= 5000:
        _, col = linear_sum_assignment(D); hungarian = float(np.mean(col == np.arange(n)))
    else:
        hungarian = float("nan")
    out = {"label": label, "n": n, "foscttm_midrank": foscttm_midrank, "foscttm_strict": strict,
           "mrr": mrr, "median_rank": median_rank, "hungarian": hungarian}
    out.update(recall)
    return out


def write_outputs(outdir, eval_D, cells, dataset, model, extra=None):
    import json
    import pandas as pd
    outdir = Path(outdir)
    _sq = (np.asarray(eval_D).ndim == 2 and
           np.asarray(eval_D).shape[0] == np.asarray(eval_D).shape[1])
    md = PSF.metrics_from_matrix(np.asarray(eval_D, np.float64)) if _sq else None
    if _sq:
        pd.DataFrame({"cell_id": list(map(str, cells)),
                      "foscttm_sym": md["foscttm_sym"], "nfoscttm_sym": md["nfoscttm_sym"]}).to_csv(
            outdir / "per_cell_nfoscttm_metrics.csv", index=False)
    else:
        # diagonal pair: record the row labels so the coupling stays interpretable
        pd.DataFrame({"cell_id": list(map(str, cells))}).to_csv(
            outdir / "cells_mod1.csv", index=False)
    np.save(outdir / "pairwise_matrix_common.npy", np.asarray(eval_D, np.float64))  # float64: float32 collapsed near-uniform plans (FOSCTTM=0.5)
    s = score_dense(eval_D)
    summ = {"dataset": dataset, "model": model, "Mean_FOSCTTM_Sym": s["foscttm_midrank"],
            "Mean_nFOSCTTM_Sym": (float(np.mean(md["nfoscttm_sym"])) if md else float("nan")), "foscttm_strict": s["foscttm_strict"]}
    for k in ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "median_rank", "hungarian", "mrr"):
        summ[k] = s[k]
    summ.update(extra or {})
    (outdir / "summary_nfoscttm.json").write_text(json.dumps(summ, indent=2, default=float))


def load_celltype_labels(dataset_id, cells):
    import glob, pandas as pd
    files = glob.glob(f"{PROJ}/runs_data_integration/_coassay_labels/{dataset_id}/cells_with_labels.csv") \
            or glob.glob(f"{PROJ}/runs_integration/benchmarking_{dataset_id}/*/{dataset_id}/cells_with_labels.csv")
    if not files:
        return None
    d = pd.read_csv(files[0]).drop_duplicates("cell_id")
    m = dict(zip(d["cell_id"].astype(str), d["cell_type"].astype(str)))
    labs = np.array([m.get(str(c), "__NA__") for c in cells], dtype=object)
    if len(set(labs[labs != "__NA__"])) < 2 or (labs == "__NA__").mean() > 0.5:
        return None
    return labs


def _transfer_one_dir(D, labels, k):
    n = D.shape[1]; preds = np.empty(n, dtype=object)
    for j in range(n):
        col = D[:, j].copy(); col[j] = np.inf
        idx = np.argpartition(col, k)[:k]; idx = idx[np.argsort(col[idx])]
        w = 1.0 / (np.arange(len(idx)) + 1.0); tally = {}
        for r, i in enumerate(idx):
            tally[labels[i]] = tally.get(labels[i], 0.0) + w[r]
        preds[j] = max(tally, key=tally.get)
    return preds


def label_transfer_score(D, labels, k_list=(10, 30)):
    from sklearn.metrics import f1_score
    D = np.asarray(D, np.float64); labels = np.asarray(labels, dtype=object); out = {}
    for k in k_list:
        p12 = _transfer_one_dir(D, labels, k); p21 = _transfer_one_dir(D.T, labels, k)
        out[f"lt_acc_k{k}"] = 0.5 * (float(np.mean(p12 == labels)) + float(np.mean(p21 == labels)))
        out[f"lt_macrof1_k{k}"] = 0.5 * (f1_score(labels, p12, average="macro", zero_division=0) +
                                         f1_score(labels, p21, average="macro", zero_division=0))
    return out


# ---------------- v11 helpers ----------------
def csls_similarity(K, k):
    kk = int(min(k, K.shape[1] - 1, K.shape[0] - 1))
    rk1 = np.sort(K, axis=1)[:, -kk:].mean(1)
    rk2 = np.sort(K, axis=0)[-kk:, :].mean(0)
    return 2.0 * K - rk1[:, None] - rk2[None, :]


def rank_rows(D):
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 1, np.asarray(D, np.float64))


def joint_row_stochastic(W):
    deg = np.asarray(W.sum(1)).ravel(); deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ W


def label_propagation_score(P, n, labels, alpha=0.9, iters=50):
    """RWR label propagation on a joint row-stochastic P (sparse, 2n x 2n; nodes 0..n-1
    = mod1, n..2n-1 = mod2). Seed one modality's cell-type one-hots, propagate, read the
    other modality's argmax. Symmetric. labels aligned to both modalities (paired)."""
    from sklearn.metrics import f1_score
    labels = np.asarray(labels, dtype=object); cats = sorted(set(labels.tolist()))
    cix = {c: i for i, c in enumerate(cats)}; C = len(cats)
    def prop(seed_side):
        Y = np.zeros((2 * n, C))
        base = 0 if seed_side == 1 else n
        for idx in range(n):
            Y[base + idx, cix[labels[idx]]] = 1.0
        F = Y.copy()
        for _ in range(iters):
            F = alpha * (P @ F) + (1.0 - alpha) * Y
        pred = F[n:2 * n] if seed_side == 1 else F[0:n]
        pl = np.array([cats[a] for a in pred.argmax(1)], dtype=object)
        return float(np.mean(pl == labels)), f1_score(labels, pl, average="macro", zero_division=0)
    a12, f12 = prop(1); a21, f21 = prop(2)
    return {"lt_prop_acc": 0.5 * (a12 + a21), "lt_prop_macrof1": 0.5 * (f12 + f21)}
