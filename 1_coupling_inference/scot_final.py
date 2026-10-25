#!/usr/bin/env python3
"""Run SCOT on a co-assay dataset with given (k, epsilon), compute FOSCTTM, save coupling.
Usage: scot_tune.py <ds> <k> <epsilon> <outdir>"""
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
sys.path.insert(0, "/g/data/yr31/mp8713/tools/SCOT/src")
import dv_common as DC
from scotv1 import SCOT
import evals

ds, k, eps, outdir = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
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
S1, S2, common = DC.shared_block(X1, f1, X2, f2, mod2, do_log1p=True, gene_mode="hvg", n_hvg=3000, log=lambda *a: None)
S1 = np.asarray(S1, np.float64); S2 = np.asarray(S2, np.float64)
print(f"[scot] {ds} n={len(cells)} common={len(common)} k={k} eps={eps}", flush=True)
sc = SCOT(S1, S2)
Xa, ya = sc.align(k=k, e=eps, mode="connectivity", metric="correlation", normalize=True, norm="l2", verbose=False)
ok = sc.flag and Xa is not None
if not ok:
    json.dump({"ds": ds, "k": k, "eps": eps, "status": "convergence_error", "foscttm": None}, open(f"{outdir}/result.json", "w"))
    print(f"[scot] {ds} k={k} eps={eps} CONVERGENCE_ERROR", flush=True); sys.exit(0)
fos = float(np.mean(evals.calc_domainAveraged_FOSCTTM(Xa, ya)))
G = np.asarray(sc.coupling, np.float64)            # GW transport plan (gamma)
np.save(f"{outdir}/gamma.npy", G)
open(f"{outdir}/cells.txt", "w").write("\n".join(map(str, cells)))
json.dump({"ds": ds, "k": k, "eps": eps, "status": "ok", "foscttm": fos, "n": len(cells),
           "n_common": len(common), "gamma_sum": float(G.sum()), "secs": time.time()-t0},
          open(f"{outdir}/result.json", "w"), indent=2)
print(f"[scot] {ds} k={k} eps={eps} FOSCTTM={fos:.4f} ({time.time()-t0:.0f}s)", flush=True)
