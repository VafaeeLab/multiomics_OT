#!/usr/bin/env python3
"""Faithful MMD-MA (kernel, Nystrom-scaled). Learns alpha1,alpha2 s.t. Z_i = K_i @ alpha_i, minimising
MMD(Z1,Z2) + lam_orth*||Z_i^T Z_i/n - I||^2 (orthonormal embed) + lam_dist*||G_i - Z_i Z_i^T||^2
(within-domain structure preservation -> this is what aligns cells across domains). Data-scaled RBF.
Usage: mmdma_run_v3.py <ds> <d> <lam_orth> <lam_dist> <lr> <outdir> [--epochs N]"""
import sys, os, json, time, numpy as np, torch
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
import dv_common as DC
ds, d, lo, ld, lr, outdir = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
epochs = int(sys.argv[sys.argv.index("--epochs")+1]) if "--epochs" in sys.argv else 2000
os.makedirs(outdir, exist_ok=True); t0 = time.time()
m = json.load(open(f"{ROOT}/1_coupling_inference/foscttm_input_map.json"))[ds]
mod2 = "adt" if m["kind"]=="adt" else ("atac" if m["kind"]=="atac" else "gene_activity")
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
def zscore(M): M=np.asarray(M,np.float32); return (M-M.mean(0))/(M.std(0)+1e-6)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = torch.tensor(zscore(S1), device=dev); B = torch.tensor(zscore(S2), device=dev); n = A.shape[0]
L = min(n, 1000); torch.manual_seed(0)
lm = torch.randperm(n, device=dev)[:L]                                    # Nystrom landmarks
def med_bw(Xr, Xc):
    with torch.no_grad():
        dd = torch.cdist(Xr[:800], Xc)**2; return torch.sqrt(0.5*torch.median(dd[dd>0]).clamp_min(1e-6))
sA = med_bw(A, A[lm]); sB = med_bw(B, B[lm])
def rbf(Xr, Xc, s): return torch.exp(-(torch.cdist(Xr, Xc)**2)/(2*s*s))
K1 = rbf(A, A[lm], sA); K2 = rbf(B, B[lm], sB)                            # (n x L) Nystrom features
print(f"[mmdma3] {ds} n={n} L={L} d={d} lo={lo} ld={ld} lr={lr} ep={epochs}", flush=True)
a1 = (torch.randn(L, d, device=dev)/np.sqrt(L)).requires_grad_(True)
a2 = (torch.randn(L, d, device=dev)/np.sqrt(L)).requires_grad_(True)
opt = torch.optim.Adam([a1, a2], lr=lr); I = torch.eye(d, device=dev)
sub = torch.randperm(n, device=dev)[:min(n, 1200)]
G1 = (K1[sub] @ K1[sub].t()); G2 = (K2[sub] @ K2[sub].t())               # within-domain structure (Nystrom Gram)
G1 = G1/G1.norm(); G2 = G2/G2.norm()
def mmd(X, Y, s):
    def k(U,V): d2=(U*U).sum(1)[:,None]+(V*V).sum(1)[None,:]-2*U@V.t(); return sum(torch.exp(-d2/(2*(s*f)**2)) for f in (0.5,1.,2.))
    return k(X,X).mean()+k(Y,Y).mean()-2*k(X,Y).mean()
for it in range(epochs):
    opt.zero_grad(); Z1 = K1@a1; Z2 = K2@a2; z1s, z2s = Z1[sub], Z2[sub]
    with torch.no_grad(): sz = (torch.sqrt(0.5*torch.median((torch.cdist(z1s,z1s)**2).clamp_min(1e-9)))).clamp_min(1e-4)
    Lm = mmd(z1s, z2s, sz)
    orth = ((z1s.t()@z1s/len(sub)-I)**2).sum() + ((z2s.t()@z2s/len(sub)-I)**2).sum()
    P1 = z1s@z1s.t(); P2 = z2s@z2s.t(); dist = ((G1-P1/P1.norm())**2).sum() + ((G2-P2/P2.norm())**2).sum()
    (Lm + lo*orth + ld*dist).backward(); opt.step()
Z1 = (K1@a1).detach().cpu().numpy(); Z2 = (K2@a2).detach().cpu().numpy()
from sklearn.preprocessing import normalize; Z1, Z2 = normalize(Z1), normalize(Z2)
from sklearn.metrics import pairwise_distances
D = pairwise_distances(Z1, Z2); dg = np.diag(D)
pc = 0.5*(((D<dg[:,None]).sum(1))/(n-1) + ((D<dg[None,:]).sum(0))/(n-1)); fos = float(pc.mean())
def sinkhorn(C, eps=0.05, it=150):
    C=np.asarray(C,np.float64); sc=np.median(C)+1e-12; K=np.exp(-C/(sc*eps)); a=np.full(K.shape[0],1/K.shape[0]);b=np.full(K.shape[1],1/K.shape[1]);u=np.ones(K.shape[0]);v=np.ones(K.shape[1])
    for _ in range(it): u=a/(K@v+1e-300); v=b/(K.T@u+1e-300)
    G=u[:,None]*K*v[None,:]; return G/(G.sum()+1e-300)
np.save(f"{outdir}/gamma.npy", sinkhorn(D)); np.save(f"{outdir}/pairwise_matrix_common.npy", D.astype(np.float32))
import pandas as pd; pd.DataFrame({"cell_id":list(map(str,cells)),"foscttm_sym":pc}).to_csv(f"{outdir}/per_cell_foscttm.csv", index=False)
json.dump({"ds":ds,"d":d,"lam_orth":lo,"lam_dist":ld,"lr":lr,"epochs":epochs,"foscttm":fos,"n":n}, open(f"{outdir}/result.json","w"), indent=2)
print(f"[mmdma3] {ds} FOSCTTM={fos:.4f} ({time.time()-t0:.0f}s)", flush=True)
