#!/usr/bin/env python3
"""Run UnionCom on a co-assay dataset with given params, compute FOSCTTM, save matching gamma.
Usage: unioncom_tune.py <ds> <kmax> <epsilon> <beta> <rho> <outdir>"""
import sys, os, json, time, numpy as np
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
ROOT = P
sys.path.insert(0, f"{ROOT}/1_coupling_inference")
sys.path.insert(0, "/g/data/yr31/mp8713/tools/UnionCom")
import dv_common as DC
from UnionCom import UnionCom

ds, kmax, eps, beta, rho, outdir = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
os.makedirs(outdir, exist_ok=True); t0 = time.time()
m = json.load(open(f"{ROOT}/1_coupling_inference/foscttm_input_map.json"))[ds]
mod2 = "adt" if m["kind"] == "adt" else ("atac" if m["kind"] == "atac" else "gene_activity")
# A co-assay pair shares barcodes across modalities; a diagonal pair does
# not, so align_common_cells raises. Try paired first and fall back --
# the data decides, not a flag that could disagree with it.
try:
    X1, X2, cells, f1, f2 = DC.load_align(m["rna"], m["atac"], m["ori1"], m["ori2"], log=lambda *a: None)
    cells1 = cells2 = cells; PAIRED = True
except ValueError:
    X1, X2, cells1, cells2, f1, f2 = DC.load_align_unpaired(m["rna"], m["atac"], m["ori1"], m["ori2"], log=lambda *a: None)
    cells = cells1; PAIRED = False   # FOSCTTM undefined without pairing
# UnionCom matches via intra-modal geodesic distances -> use each modality's OWN features (PCA of log1p),
# NOT a shared common-feature block (that impoverished input collapses the matching).
import scipy.sparse as _sp
from sklearn.decomposition import PCA as _PCA
# Measured, not assumed: reducing to 50 components made scnmtRNAmC WORSE,
# 0.3489 -> 0.4323, so the missing reduction did NOT explain UnionCom's
# poor scores. Off by default; UC_PCA=<n> re-enables it for testing.
_PCA_DIM = int(os.environ.get("UC_PCA", "0"))           # 0 = no reduction


def _prep(_X):
    _X = np.log1p(np.clip(_X.toarray() if _sp.issparse(_X) else np.asarray(_X, float), 0, None))
    _thr = np.percentile(_X, 99.9); _X = np.clip(_X, 0.0, _thr)   # original: --clip_percentile 99.9 (winsorize extremes)
    _n = np.linalg.norm(_X, axis=1, keepdims=True)
    _X = (_X / np.clip(_n, 1e-8, None)).astype(np.float32)        # log1p + clip + row-l2
    # UnionCom's matching runs on a k-NN graph with geodesic distances. At full
    # feature width those distances concentrate and the graph carries almost no
    # signal, which is why the integrated embedding recovered the true partner
    # at chance rate. Reduce first, as the method intends.
    if _PCA_DIM and _X.shape[1] > _PCA_DIM:
        _k = min(_PCA_DIM, _X.shape[0] - 1, _X.shape[1])
        _X = _PCA(n_components=_k, random_state=0).fit_transform(_X).astype(np.float32)
    return _X
S1 = _prep(X1); S2 = _prep(X2); common = list(range(S1.shape[1]))
S1 = np.asarray(S1, np.float32); S2 = np.asarray(S2, np.float32)
print(f"[unioncom] {ds} n={len(cells)} common={len(common)} kmax={kmax} eps={eps} beta={beta} rho={rho}", flush=True)

uc = UnionCom(epoch_pd=int(os.environ.get("UC_EPOCH_PD", "2000")), epoch_DNN=100, epsilon=eps, lr=0.001, batch_size=100, beta=beta,
              rho=rho, kmax=kmax,
              output_dim=int(os.environ.get("UC_OUTPUT_DIM", "8")),
              distance_mode="geodesic", project_mode="barycentric", manual_seed=666)
cap = {}
_orig = uc.match
def match_capture(dataset):
    r = _orig(dataset); cap["F"] = np.asarray(r[0], np.float64); return r
uc.match = match_capture
try:
    integrated = uc.fit_transform([S1.copy(), S2.copy()])
except Exception as e:
    json.dump({"ds": ds, "kmax": kmax, "eps": eps, "beta": beta, "rho": rho, "status": "error", "err": str(e)[:200], "foscttm": None}, open(f"{outdir}/result.json", "w"))
    print(f"[unioncom] {ds} ERROR {str(e)[:120]}", flush=True); sys.exit(0)

def foscttm(A, B):  # row+col fraction-of-samples-closer-than-true-match (paired)
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(A, B); n = D.shape[0]; dg = np.diag(D)
    fr = ((D < dg[:, None]).sum(1)) / (n - 1); fc = ((D < dg[None, :]).sum(0)) / (n - 1)
    return float(0.5 * (fr + fc).mean())
fos = foscttm(np.asarray(integrated[0]), np.asarray(integrated[1]))
# save per-cell FOSCTTM + pairwise from the ALIGNED EMBEDDINGS (the matching matrix F can be degenerate/uniform)
from sklearn.metrics import pairwise_distances as _pwd
import pandas as _pd
_D = _pwd(np.asarray(integrated[0], float), np.asarray(integrated[1], float)); _n = _D.shape[0]; _dg = np.diag(_D)
_fr = (_D < _dg[:, None]).sum(1) / (_n - 1); _fc = (_D < _dg[None, :]).sum(0) / (_n - 1)
_pd.DataFrame({"cell_id": list(map(str, cells)), "foscttm_row": _fr, "foscttm_col": _fc,
              "foscttm_symmetric": 0.5 * (_fr + _fc)}).to_csv(f"{outdir}/per_cell_nfoscttm_metrics.csv", index=False)
np.save(f"{outdir}/pairwise_matrix_common.npy", _D.astype(np.float64))
# The prime-dual matching matrix is kept for reference, but it is NOT the
# coupling: the script's own comment above notes it can be degenerate, and the
# FOSCTTM reported here comes from the embeddings, not from F. Saving F as
# gamma.npy meant the score and the coupling described different objects.
# gamma.npy is now a similarity built from the embedding distances - the same
# quantity the score is computed on - so downstream consumers and the metric
# agree.
_Fm = cap.get("F")
if _Fm is not None:
    np.save(f"{outdir}/F_matching.npy", _Fm)
_R = np.apply_along_axis(lambda r: np.argsort(np.argsort(r)) + 1, 1, _D)
G = (1.0 / (_R.astype(np.float64) + 1.0))
G = G / G.sum()
np.save(f"{outdir}/gamma.npy", G); open(f"{outdir}/cells.txt", "w").write("\n".join(map(str, cells)))
# A collapsed embedding makes every pairwise distance equal, which drives
# FOSCTTM to ~0 and looks like a perfect result. Detect it and refuse to report
# a score, so no sweep can select it and no table can average it.
_spread = float(_D.max() - _D.min())
_rel = _spread / (abs(float(_D.mean())) + 1e-12)
_collapsed = _rel < 1e-4
if _collapsed:
    print(f"[unioncom] {ds} COLLAPSED: distance spread {_spread:.3g} "
          f"(relative {_rel:.3g}); FOSCTTM {fos:.4f} is an artefact, not a score",
          flush=True)
json.dump({"ds": ds, "kmax": kmax, "eps": eps, "beta": beta, "rho": rho,
           "status": "collapsed" if _collapsed else "ok",
           "foscttm": (None if _collapsed else fos),
           "foscttm_raw": fos, "dist_spread": _spread, "dist_spread_rel": _rel,
           "n": len(cells), "n_common": len(common), "secs": time.time()-t0}, open(f"{outdir}/result.json", "w"), indent=2)
print(f"[unioncom] {ds} FOSCTTM={fos:.4f} ({time.time()-t0:.0f}s)", flush=True)
