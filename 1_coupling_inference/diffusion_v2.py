#!/usr/bin/env python3
"""diffusion_v2 (refactored, ALL-HVG variant) — two-cost entropic Gromov-Wasserstein.
Combines former 1_coupling_inference/diffusion_v2.py + dv_common.py into ONE script.
  Core: within-modality PCA -> diffusion coordinates; build INTRA-modality structural
  distances (diffusion-euclidean + feature-cosine); entropic Gromov-Wasserstein
  (alpha*diffusion + beta*feature) over refine_rounds -> probability gamma (the GW OT plan).

Refinements:
  (1) No log1p by default (use --log1p to enable).
  (2) Modality A/B assignment: RNA-seq -> modalityA. If no RNA, or same type, random pick.
      Modality codes: rna, atac, adt, methylome (user specifies --modality1/--modality2);
      anything else -> "other".
  (2b) --cell_id_alignment T/F (default T). T keeps cells common to both modalities in the
      same order (co-assay / horizontal); ANY cell-id mismatch -> error. F = diagonal
      integration: matrices kept as-is, no FOSCTTM.
  (3) --hvg 5000 default. Per-modality HVG: a modality with <hvg features uses ALL its
      features; a modality with >=hvg features uses its top-hvg variable features. Then the
      COMMON features are intersected. ADT features are remapped to gene symbols (human/mouse).
      Errors: no common feature -> error; both modalities >hvg but common HVG < min_common_hvg
      (default 400) -> error.
  (4) No grid params. Single hvg=5000, eps=0.0001.
  (5) Output = the entropic-OT plan saved as the probability gamma matrix (gamma.npy). If
      cell_id_alignment=T, FOSCTTM is computed by treating (-gamma) as a distance (distance
      matrix is NOT saved).
numpy/scipy/sklearn only (+ optional Rscript for .rds, optional ADT remap util).
"""
import argparse, json, sys, time, random, subprocess, tempfile, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

NF = Path(__file__).resolve().parent
PROJ = NF.parent
# optional ADT->gene remap utility (kept external; large curated alias table)
_REMAP = None
for _cand in (PROJ / "4_label_transfer" / "remap_adt_to_gene.py",
              PROJ / "label_transfer_models" / "remap_adt_to_gene.py"):
    if _cand.exists():
        import importlib.util as _il
        _spec = _il.spec_from_file_location("remap_adt_to_gene", str(_cand))
        _REMAP = _il.module_from_spec(_spec); sys.modules["remap_adt_to_gene"] = _REMAP
        _spec.loader.exec_module(_REMAP); break

MOD_CODES = {"rna", "atac", "adt", "methylome"}

def log(m): print(f"[v2 {time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------------- matrix loaders (embedded) ----------------
def read_names(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line: out.append(line.split("\t")[0])
    return np.array(out, dtype=str)

def auto_sidecars(base, cands):
    for c in cands:
        p = Path(str(base) + c)
        if p.exists(): return str(p)
    return None

def load_npz(path):
    d = np.load(path, allow_pickle=True); keys = set(d.files)
    if {"data", "indices", "indptr", "shape"}.issubset(keys):
        shape = tuple(np.asarray(d["shape"], np.int64).tolist())
        if len(d["indptr"]) == shape[0] + 1:
            X = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape)
        else:
            X = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=shape).tocsr()
    elif "X" in keys: X = sp.csr_matrix(d["X"])
    elif "matrix" in keys: X = sp.csr_matrix(d["matrix"])
    else: raise ValueError(f"Unsupported NPZ keys: {list(d.files)}")
    rn = d["row_names"].astype(str) if "row_names" in keys else np.arange(X.shape[0]).astype(str)
    cn = d["col_names"].astype(str) if "col_names" in keys else np.arange(X.shape[1]).astype(str)
    return X, rn, cn

def load_mtx(path, barcodes=None, features=None):
    path = Path(path); X = sio.mmread(str(path)).tocsr()
    if barcodes is None:
        barcodes = auto_sidecars(path.with_suffix(""), ["_barcodes.tsv", "_barcodes.tsv.gz", ".barcodes.tsv"])
    if features is None:
        features = auto_sidecars(path.with_suffix(""), ["_features.tsv", "_features.tsv.gz", ".features.tsv", "_genes.tsv"])
    rn = read_names(barcodes) if barcodes and Path(barcodes).exists() else np.arange(X.shape[0]).astype(str)
    cn = read_names(features) if features and Path(features).exists() else np.arange(X.shape[1]).astype(str)
    return X, rn, cn

def load_rds(path):
    path = Path(path)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); xb, ib, pb = td/"x.bin", td/"i.bin", td/"p.bin"
        mf, rf, cf = td/"meta.txt", td/"rn.txt", td/"cn.txt"
        rs = f"""suppressMessages(library(Matrix))
m <- readRDS('{path}'); if(!inherits(m,'sparseMatrix')) m <- as(m,'CsparseMatrix')
writeBin(as.double(m@x),'{xb}',size=4,endian='little'); writeBin(as.integer(m@i),'{ib}',size=4,endian='little')
writeBin(as.integer(m@p),'{pb}',size=4,endian='little'); cat(m@Dim[1],m@Dim[2],length(m@x),file='{mf}',sep=' ')
rn<-rownames(m); if(is.null(rn)) rn<-as.character(seq_len(m@Dim[1])); cn<-colnames(m); if(is.null(cn)) cn<-as.character(seq_len(m@Dim[2]))
writeLines(rn,'{rf}'); writeLines(cn,'{cf}')"""
        r = subprocess.run(["Rscript", "--vanilla", "-"], input=rs, capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError(f"Rscript failed for {path}\n{r.stderr}")
        nr, nc, nnz = map(int, mf.read_text().split())
        x = np.fromfile(str(xb), np.float32, nnz); i = np.fromfile(str(ib), np.int32, nnz); p = np.fromfile(str(pb), np.int32, nc+1)
        X = sp.csc_matrix((x, i, p), shape=(nr, nc)).tocsr()
        rn = np.array(rf.read_text().splitlines(), str); cn = np.array(cf.read_text().splitlines(), str)
    return X, rn, cn

def load_matrix(path):
    path = Path(path)
    if path.name.endswith(".npz"): X, rn, cn = load_npz(path)
    elif path.name.endswith((".mtx", ".mtx.gz")): X, rn, cn = load_mtx(path)
    elif path.name.endswith(".rds"): X, rn, cn = load_rds(path)
    else: raise ValueError(f"Unsupported matrix format: {path}")
    log(f"  loaded {path.name}: {X.shape}")
    return X, rn, cn

def get_oriented(X, rn, cn, orientation):
    if orientation == "cell_by_feature": return X.tocsr(), rn, cn
    if orientation == "feature_by_cell": return X.T.tocsr(), cn, rn
    raise ValueError("orientation must be cell_by_feature or feature_by_cell")

# ---------------- ADT -> gene remap (human/mouse) ----------------
def remap_adt(adt_names, gene_ref):
    """Map ADT protein/antibody symbols to gene symbols, case-insensitive vs the reference
    gene set (handles human UPPER + mouse Title-case), via the curated alias table + exact
    match. Returns array aligned to adt_names ('__DROP__' where unmapped)."""
    out = np.array([str(x) for x in adt_names], dtype=object)
    ref_ci = {str(g).upper(): str(g) for g in gene_ref}
    if _REMAP is not None:
        try:
            recs, _ = _REMAP.remap_dataset(list(adt_names), list(gene_ref))
            out = np.array([(r["assigned"] if r.get("action") in ("alias", "gene_exact") else "__DROP__")
                            for r in recs], dtype=object)
        except Exception as e:
            log(f"  [warn] ADT remap util failed ({e}); falling back to case-insensitive exact match")
    # case-insensitive exact match to the reference (recovers mouse Title-case symbols)
    for k, v in enumerate(out):
        if v == "__DROP__" or v == "" or str(v).upper() not in ref_ci:
            u = str(adt_names[k]).upper().lstrip("ANTI-").strip()
            out[k] = ref_ci.get(u, ref_ci.get(str(adt_names[k]).upper(), "__DROP__"))
        else:
            out[k] = ref_ci[str(v).upper()]
    return out

# ---------------- preprocessing + HVG ----------------
def to_dense(X): return np.asarray(X.todense(), np.float64) if sp.issparse(X) else np.asarray(X, np.float64)
def zscore_cols(M): return (M - M.mean(0)) / (M.std(0) + 1e-8)
def l2_rows(M):
    nrm = np.linalg.norm(M, axis=1, keepdims=True); nrm[nrm == 0] = 1.0; return M / nrm
def dispersion(M): return M.var(0) / (np.abs(M.mean(0)) + 1e-8)

def top_hvg_names(S, names, k):
    """top-k variable features within one modality; k>=n_feat -> all."""
    if S.shape[1] <= k: return list(map(str, names))
    keep = np.argsort(dispersion(S))[::-1][:k]
    return [str(names[i]) for i in keep]

# ---------------- entropic-OT (gamma) ----------------
def sparse_log1p(X):
    X = X.copy().tocsr()
    if X.nnz > 0 and np.nanmin(X.data) >= 0: X.data = np.log1p(X.data)
    return X

def row_l1_normalize(X):
    X = X.tocsr(copy=True); ssum = np.asarray(X.sum(1)).ravel(); ssum[ssum == 0] = 1.0
    return sp.diags(1.0 / ssum).dot(X).tocsr()

def pca_embed(X, n_components, seed, do_log1p=False):
    Xp = sparse_log1p(X) if (do_log1p and sp.issparse(X)) else (X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X, np.float64)))
    Xp = row_l1_normalize(Xp)
    nc = int(max(2, min(n_components, min(Xp.shape) - 1)))
    Z = TruncatedSVD(n_components=nc, random_state=seed).fit_transform(Xp)
    return np.asarray(Z, np.float64)

def diffusion_coordinates(Z, knn_k, n_components, diffusion_time, multiscale_times, mutual_knn, seed):
    n = Z.shape[0]
    if n <= 3:
        return np.asarray(Z, np.float32)
    k = int(min(knn_k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Z)
    dist, ind = nn.kneighbors(Z); dist = dist[:, 1:]; ind = ind[:, 1:]
    sigma = np.median(dist[:, -1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    weights = np.exp(-(dist ** 2) / (sigma ** 2 + 1e-12))
    rows = np.repeat(np.arange(n), k); cols = ind.ravel(); vals = weights.ravel()
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.minimum(W.T) if mutual_knn else 0.5 * (W + W.T)
    W.setdiag(0); W.eliminate_zeros()
    deg = np.asarray(W.sum(1)).ravel(); deg[deg == 0] = 1.0
    Dis = sp.diags(1.0 / np.sqrt(deg)); S = Dis @ W @ Dis
    k_eig = int(min(n_components + 1, n - 1))
    try:
        ev, vec = eigsh(S, k=k_eig, which="LA")
        order = np.argsort(ev)[::-1]; ev = ev[order][1:]; vec = vec[:, order][:, 1:]
        times = sorted(set(int(x) for x in str(multiscale_times).split(",") if str(x).strip() != "")) or [int(diffusion_time)]
        coords = np.concatenate([vec * (ev ** t)[None, :] for t in times], axis=1)
    except Exception:
        coords = Z[:, :min(n_components, Z.shape[1])]
    coords = np.asarray(coords, np.float32)
    return (coords - coords.mean(0, keepdims=True)) / (coords.std(0, keepdims=True) + 1e-8)

def minmax01(C):
    C = np.asarray(C, np.float64); mn = np.nanmin(C); mx = np.nanmax(C)
    return np.zeros_like(C) if mx <= mn else (C - mn) / (mx - mn)

def pairwise_l2(A, B):
    return pairwise_distances(A, B, metric="euclidean").astype(np.float64)

def gw_cost(Dx, Dy, G):
    """Entropic-GW structural cost (squared loss) for a transport plan G -> n_x x n_y gradient cost."""
    p = G.sum(1); q = G.sum(0)
    return (Dx ** 2 @ p)[:, None] + (Dy ** 2 @ q)[None, :] - 2.0 * (Dx @ (G @ Dy.T))

def sinkhorn_from_cost(C, epsilon, n_iter):
    """Entropic Sinkhorn from a (non-OT) cost matrix; returns the OT plan."""
    C = np.asarray(C, np.float64); C = C - np.nanmin(C)
    eps = float(epsilon) if epsilon > 0 else 1e-3
    K = np.exp(-C / eps); K[K < 1e-300] = 1e-300
    n, m = K.shape; a = np.full(n, 1.0 / n); b = np.full(m, 1.0 / m)
    u = np.ones(n); v = np.ones(m)
    for _ in range(int(n_iter)):
        u = a / (K @ v + 1e-300); v = b / (K.T @ u + 1e-300)
    return (u[:, None] * K) * v[None, :]

def sinkhorn_plan(C, eps, iters=400):
    """Entropic-OT (matvec Sinkhorn, median-scaled). Returns the OT plan."""
    C = np.asarray(C, np.float64); scale = float(np.median(C)) + 1e-12
    K = np.exp(-C / (scale * max(eps, 1e-12)))
    n1, n2 = K.shape; u = np.ones(n1); v = np.ones(n2)
    a = np.full(n1, 1.0 / n1); b = np.full(n2, 1.0 / n2)
    for _ in range(int(iters)):
        u = a / (K @ v + 1e-300); v = b / (K.T @ u + 1e-300)
    return u[:, None] * K * v[None, :]

# ---------------- FOSCTTM (only when aligned) ----------------
def foscttm_from_distance(D, chunk=2048):
    D = np.asarray(D, np.float64); n = D.shape[0]
    mn = np.nanmin(D[np.isfinite(D)]) if np.isfinite(D).any() else 0.0
    if mn != 0.0: D = D - mn
    diag = np.diag(D).copy(); fr = np.empty(n); fc = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s+chunk, n); blk = D[s:e]; dv = diag[s:e][:, None]
        fr[s:e] = ((blk < dv).sum(1) + 0.5*((blk == dv).sum(1)-1)) / max(1, n-1)
    for s in range(0, n, chunk):
        e = min(s+chunk, n); blk = D[:, s:e]; dv = diag[s:e][None, :]
        fc[s:e] = ((blk < dv).sum(0) + 0.5*((blk == dv).sum(0)-1)) / max(1, n-1)
    return 0.5 * (fr + fc)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="diffusion_v2 self-contained correspondence")
    ap.add_argument("--matrix1", required=True); ap.add_argument("--matrix2", required=True)
    ap.add_argument("--modality1", default="other"); ap.add_argument("--modality2", default="other")
    ap.add_argument("--outdir", required=True); ap.add_argument("--dataset_id", required=True)
    ap.add_argument("--orientation1", default="cell_by_feature")
    ap.add_argument("--orientation2", default="cell_by_feature")
    ap.add_argument("--cell_id_alignment", default="True")
    ap.add_argument("--hvg", type=int, default=20000)
    ap.add_argument("--min_common_hvg", type=int, default=400)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--sinkhorn_iters", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--refine_rounds", type=int, default=3)
    ap.add_argument("--rna_pca_dim", type=int, default=50)
    ap.add_argument("--atac_pca_dim", type=int, default=50)
    ap.add_argument("--knn_k", type=int, default=50)
    ap.add_argument("--diffusion_components", type=int, default=20)
    ap.add_argument("--diffusion_time", type=int, default=3)
    ap.add_argument("--multiscale_times", default="1,3,5")
    ap.add_argument("--mutual_knn", default="True")
    ap.add_argument("--log1p", action="store_true")          # default OFF
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    align = str(args.cell_id_alignment).lower() in ("t", "true", "1", "yes")
    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True); t0 = time.time()

    # load
    X1, c1, f1 = get_oriented(*load_matrix(args.matrix1), args.orientation1)
    X2, c2, f2 = get_oriented(*load_matrix(args.matrix2), args.orientation2)
    m1 = args.modality1.lower() if args.modality1.lower() in MOD_CODES else "other"
    m2 = args.modality2.lower() if args.modality2.lower() in MOD_CODES else "other"

    # (2a) modality A/B assignment
    d1 = (X1, c1, f1, m1); d2 = (X2, c2, f2, m2)
    if m1 == "rna" and m2 != "rna": A, B = d1, d2
    elif m2 == "rna" and m1 != "rna": A, B = d2, d1
    else: A, B = (d1, d2) if random.random() < 0.5 else (d2, d1)   # no rna, or same type -> random
    XA, cA, fA, modA = A; XB, cB, fB, modB = B
    log(f"modalityA={modA} (n={XA.shape[0]}, feat={XA.shape[1]})  modalityB={modB} (n={XB.shape[0]}, feat={XB.shape[1]})  cell_id_alignment={align}")

    # (2b) cell-id alignment
    if align:
        sA, sB = set(map(str, cA)), set(map(str, cB))
        if sA != sB:
            raise SystemExit(
                f"[ERROR] cell_id_alignment=True but cell ids differ "
                f"(|A|={len(sA)} |B|={len(sB)} common={len(sA & sB)}). "
                f"Set --cell_id_alignment False for diagonal integration, "
                f"or provide two matrices with EXACTLY the same cell ids for horizontal integration.")
        pos = {str(c): i for i, c in enumerate(cB)}
        order = np.array([pos[str(c)] for c in cA])
        XB = XB[order, :]; cells = np.array([str(c) for c in cA])
        log(f"  aligned co-assay cells: n={len(cells)} (same order)")
    else:
        cells = None
        log(f"  diagonal integration: matrices kept as-is (A n={XA.shape[0]}, B n={XB.shape[0]}); no FOSCTTM")

    # (3) ADT -> gene remap (use the other modality's names as gene reference)
    if modA == "adt": fA = remap_adt(fA, fB)
    if modB == "adt": fB = remap_adt(fB, fA)
    fA = np.array([str(x) for x in fA]); fB = np.array([str(x) for x in fB])

    # dense + optional log1p (default OFF)
    SA = to_dense(XA); SB = to_dense(XB)
    if args.log1p:
        SA = np.log1p(np.clip(SA, 0, None)); SB = np.log1p(np.clip(SB, 0, None)); log("  log1p applied")

    # (3) per-modality HVG then common intersection
    nA0, nB0 = SA.shape[1], SB.shape[1]
    hvgA = top_hvg_names(SA, fA, args.hvg); hvgB = set(top_hvg_names(SB, fB, args.hvg))
    bad = {"", "nan", "none", "__drop__"}
    common = [g for g in dict.fromkeys(hvgA) if g in hvgB and g.lower() not in bad]
    log(f"  HVG: A used {min(args.hvg, nA0)}/{nA0}, B used {min(args.hvg, nB0)}/{nB0} -> common={len(common)}")
    if len(common) == 0:
        raise SystemExit("[ERROR] No common feature between the two modalities. "
                         "Use datasets with common feature names (e.g. shared gene symbols).")
    if (nA0 > args.hvg and nB0 > args.hvg) and len(common) < args.min_common_hvg:
        raise SystemExit(f"[ERROR] Only {len(common)} common highly-variable features "
                         f"(< min_common_hvg={args.min_common_hvg}). Use datasets with more shared "
                         f"features, or lower --min_common_hvg.")
    iA = {g: k for k, g in enumerate(fA)}; iB = {g: k for k, g in enumerate(fB)}
    idxA = np.array([iA[g] for g in common]); idxB = np.array([iB[g] for g in common])

    # (5) v2 core: two-cost entropic Gromov-Wasserstein on INTRA-modality structural distances
    mutual = str(args.mutual_knn).lower() in ("t", "true", "1", "yes")
    XAh = XA[:, idxA].tocsr(); XBh = XB[:, idxB].tocsr()   # all steps now run on the common-HVG block (like v7)
    log("  ALL-HVG mode: PCA/diffusion/structural costs computed on %d common HVG features" % len(common))
    ZA = pca_embed(XAh, args.rna_pca_dim, args.seed, do_log1p=args.log1p)
    ZB = pca_embed(XBh, args.atac_pca_dim, args.seed, do_log1p=args.log1p)
    PsiA = diffusion_coordinates(ZA, args.knn_k, args.diffusion_components, args.diffusion_time, args.multiscale_times, mutual, args.seed)
    PsiB = diffusion_coordinates(ZB, args.knn_k, args.diffusion_components, args.diffusion_time, args.multiscale_times, mutual, args.seed)
    md = min(PsiA.shape[1], PsiB.shape[1])
    Ddx = pairwise_l2(PsiA[:, :md], PsiA[:, :md]); Ddy = pairwise_l2(PsiB[:, :md], PsiB[:, :md])   # intra-modality diffusion distances
    Dfx = pairwise_distances(ZA, ZA, metric="cosine").astype(np.float64)                            # intra-modality feature distances
    Dfy = pairwise_distances(ZB, ZB, metric="cosine").astype(np.float64)
    nA, nB = ZA.shape[0], ZB.shape[0]
    G = np.full((nA, nB), 1.0 / (nA * nB))
    log("  v2 GW: alpha=%.2f beta=%.2f refine_rounds=%d (common features validated: %d)" % (
        args.alpha, args.beta, args.refine_rounds, len(common)))
    for r in range(int(args.refine_rounds)):
        Cd = minmax01(gw_cost(Ddx, Ddy, G)); Cf = minmax01(gw_cost(Dfx, Dfy, G))
        G = sinkhorn_from_cost(args.alpha * Cd + args.beta * Cf, args.eps, args.sinkhorn_iters)
        log("    GW round %d/%d done" % (r + 1, int(args.refine_rounds)))
    gamma = G / (G.sum() + 1e-300)                                 # probability gamma (the GW transport plan, sum=1)
    np.save(out / "gamma.npy", np.asarray(gamma, np.float64))
    rn = cells if align else np.array([str(c) for c in cA])
    cn = cells if align else np.array([str(c) for c in cB])
    np.savez_compressed(out / "gamma_metadata.npz", row_names=rn.astype(str), col_names=cn.astype(str),
                        modalityA=modA, modalityB=modB, n_common_features=len(common))
    (out / "run_settings.json").write_text(json.dumps({**vars(args), "cell_id_alignment": align,
        "modalityA": modA, "modalityB": modB, "n_common_features": len(common),
        "n_featuresA": int(nA0), "n_featuresB": int(nB0)}, indent=2))

    summary = {"dataset": args.dataset_id, "model": "diffusion_v2", "modalityA": modA, "modalityB": modB,
               "cell_id_alignment": align, "n_common_features": len(common),
               "gamma_shape": list(gamma.shape), "gamma_sum": float(gamma.sum())}
    # (5) FOSCTTM only if aligned: distance = -gamma (NOT saved)
    if align:
        fos = foscttm_from_distance(-gamma)
        import pandas as pd
        pd.DataFrame({"cell_id": list(map(str, cells)), "foscttm_sym": fos}).to_csv(
            out / "per_cell_foscttm.csv", index=False)
        summary["Mean_FOSCTTM"] = float(np.mean(fos))
        log(f"  FOSCTTM (from -gamma distance) = {np.mean(fos):.4f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"DONE {args.dataset_id} {time.time()-t0:.1f}s | gamma{gamma.shape} saved | "
        f"{'FOSCTTM=%.4f' % summary['Mean_FOSCTTM'] if align else 'diagonal (no FOSCTTM)'}")

if __name__ == "__main__":
    main()
