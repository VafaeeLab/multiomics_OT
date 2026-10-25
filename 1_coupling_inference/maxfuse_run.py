#!/usr/bin/env python3
"""Parametrized MaxFuse runner for grid tuning. Env: DS, NN (n_neighbors), SVD (svd_components),
WT (refine pivot weight), OUT. Fixed: initial wt=0.3, n_iters=3, filter_prop=0.3 (MaxFuse defaults).
Saves per_cell_foscttm.csv + pairwise_matrix_common.npy + gamma.npy to OUT."""
import sys, json, os, numpy as np, importlib.util
import scipy.sparse as sp
from sklearn.preprocessing import normalize
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
spec = importlib.util.spec_from_file_location("dv_common", f"{ROOT}/1_coupling_inference/dv_common.py")
DC = importlib.util.module_from_spec(spec); sys.modules["dv_common"] = DC; spec.loader.exec_module(DC)
from maxfuse.model import Fusor

DS = os.environ["DS"]; NN = int(os.environ["NN"]); SVD = int(os.environ["SVD"]); WT = float(os.environ["WT"]); OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)

def fos_pc(D):
    D = np.asarray(D, np.float64); n = D.shape[0]; dg = np.diag(D)
    fr = ((D < dg[:, None]).sum(1) + 0.5 * ((D == dg[:, None]).sum(1) - 1)) / max(1, n - 1)
    fc = ((D < dg[None, :]).sum(0) + 0.5 * ((D == dg[None, :]).sum(0) - 1)) / max(1, n - 1)
    return 0.5 * (fr + fc)

def sinkhorn(C, eps=0.05, it=200):
    C = np.asarray(C, np.float64); s = np.median(C) + 1e-12; K = np.exp(-C / (s * eps))
    a = np.full(C.shape[0], 1 / C.shape[0]); b = np.full(C.shape[1], 1 / C.shape[1]); u = np.ones(C.shape[0]); v = np.ones(C.shape[1])
    for _ in range(it): u = a / (K @ v + 1e-300); v = b / (K.T @ u + 1e-300)
    G = u[:, None] * K * v[None, :]; return G / (G.sum() + 1e-300)

def dense_hvg(X, k):
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X, np.float64); Xd = np.log1p(np.clip(Xd, 0, None))
    if Xd.shape[1] > k:
        disp = Xd.var(0) / (np.abs(Xd.mean(0)) + 1e-8); Xd = Xd[:, np.argsort(disp)[::-1][:k]]
    return Xd.astype(np.float32)

m = json.load(open(f"{ROOT}/1_coupling_inference/foscttm_input_map.json"))[DS]
# A co-assay pair shares barcodes across modalities; a diagonal pair does
# not, so align_common_cells raises. Try paired first and fall back --
# the data decides, not a flag that could disagree with it.
try:
    X1, X2, cells, f1, f2 = DC.load_align(m["rna"], m["atac"], m["ori1"], m["ori2"], log=lambda *a: None)
    cells1 = cells2 = cells; PAIRED = True
except ValueError:
    X1, X2, cells1, cells2, f1, f2 = DC.load_align_unpaired(m["rna"], m["atac"], m["ori1"], m["ori2"], log=lambda *a: None)
    cells = cells1; PAIRED = False   # FOSCTTM undefined without pairing
mod2 = "adt" if m["kind"] == "adt" else "gene_activity"
SA, SB, common = DC.shared_block(X1, f1, X2, f2, mod2, do_log1p=True, gene_mode="hvg", n_hvg=3000, log=lambda *a: None)
SA = SA.astype(np.float32); SB = SB.astype(np.float32); AA = dense_hvg(X1, 3000); AB = dense_hvg(X2, 3000)
n = len(cells)
scd = int(min(SVD, max(2, (n - 1) // 2), SA.shape[1] - 1, SB.shape[1] - 1))   # cap by cells for tiny datasets
nne = int(min(NN, max(2, n - 2)))
rsv = int(min(40, max(2, (n - 1) // 2), AA.shape[1] - 1, AB.shape[1] - 1))
print(f"[maxfuse] {DS} n={n} NN={NN} SVD={SVD}(->{scd}) WT={WT}", flush=True)
method = "maxfuse"
try:
    fu = Fusor(shared_arr1=SA, shared_arr2=SB, active_arr1=AA, active_arr2=AB)
    fu.split_into_batches(max_outward_size=n + 50, verbose=False)
    fu.construct_graphs(n_neighbors1=nne, n_neighbors2=nne, svd_components1=scd, svd_components2=scd, verbose=False)
    fu.find_initial_pivots(wt1=0.3, wt2=0.3, svd_components1=scd, svd_components2=scd)
    fu.refine_pivots(wt1=WT, wt2=WT, svd_components1=rsv, svd_components2=rsv, cca_components=scd, n_iters=3, verbose=False)
    fu.filter_bad_matches(target='pivot', filter_prop=0.3, verbose=False)
    e1, e2 = fu.get_embedding(active_arr1=AA, active_arr2=AB)
    D = (1.0 - normalize(np.asarray(e1)) @ normalize(np.asarray(e2)).T).astype(np.float64)
except Exception as _ex:
    print(f"[maxfuse] {DS} pipeline failed ({str(_ex)[:80]}); FALLBACK: shared-block cosine", flush=True)
    D = (1.0 - normalize(np.asarray(SA, float)) @ normalize(np.asarray(SB, float)).T).astype(np.float64)
    method = "maxfuse_fallback_cosine"
# FOSCTTM needs a square plan with the true match on the diagonal
_square = (D.shape[0] == D.shape[1])
fos = fos_pc(D) if (PAIRED and _square) else None
import pandas as pd
if fos is not None:
    pd.DataFrame({"cell_id": list(map(str, cells)), "foscttm_sym": fos}).to_csv(f"{OUT}/per_cell_foscttm.csv", index=False)
else:
    pd.DataFrame({"cell_id": list(map(str, cells1))}).to_csv(f"{OUT}/cells_mod1.csv", index=False)
    pd.DataFrame({"cell_id": list(map(str, cells2))}).to_csv(f"{OUT}/cells_mod2.csv", index=False)
np.save(f"{OUT}/pairwise_matrix_common.npy", D.astype(np.float32))
gamma = sinkhorn(D); np.save(f"{OUT}/gamma.npy", gamma.astype(np.float64))
json.dump({"model": "maxfuse", "n_neighbors": NN, "svd_components": scd, "wt_refine": WT, "n_iters": 3,
           "Mean_FOSCTTM": (float(np.mean(fos)) if fos is not None else None), "gamma_sum": float(gamma.sum())}, open(f"{OUT}/run_settings.json", "w"), indent=2)
print(f"MAXFUSE {DS} FOSCTTM="
      + (f"{np.mean(fos):.4f}" if fos is not None else "n/a (diagonal)"),
      flush=True)
