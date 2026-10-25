#!/usr/bin/env python3
"""Unified custom-model label-transfer benchmark (gamma-based).
- Coassay datasets (datasets11/12): use the ALREADY-ESTIMATED gamma in runs_foscttm (no re-estimation).
- Tabula-muris closed/open (cross-batch, different cells): estimate the coupling from the two batches
  via the inline diffusion backends (lt_diffusion_backends: v1,v2,v7,v8) + fusion recipes for v9/v10/v11/v12,
  using UNSUPERVISED weight selection; save the coupling under runs_label_transfer/gammas/.
Label transfer = gamma-weighted label vote. Metrics: overall_accuracy, balanced_accuracy, macro_f1,
weighted_f1, coverage (closed-set); open-set: macro_class_accuracy, unknown_class_accuracy, macro_f1
(unknown = query cell type absent from reference; abstain via confidence threshold maximising macro-F1).
Writes one JSON row per (model,dataset). Usage: lt_bench.py --model <m> --dataset <key> --out <json>
"""
import argparse, json, os, sys
from pathlib import Path
import os
import numpy as np, pandas as pd
import scipy.io as sio, scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score, f1_score
PROJ=Path("/g/data/yr31/mp8713/project_data_integration"); TINY=1e-30
sys.path.insert(0, str(PROJ/"4_label_transfer")); sys.path.insert(0, str(PROJ/"1_coupling_inference"))
def log(m): print(m,flush=True)

# ---------------- dataset registry ----------------
COASSAY=["datasets11_CITE_BMMC_10pct","datasets11_MULTIOME_BMMC_10pct",
         "datasets12_BF21","datasets12_BM27","datasets12_WF26","datasets12_WM34","scnmtRNAmC","snmc2tRNAmC"]
TM_CLOSED=["Bladder","Heart","Kidney","liver","marrow"]     # closed-set eval (restrict query to ref types)
TM_OPEN=["mammary","muscle","spleen","thymus","Kidney"]      # open-set eval (keep unknown query types)
# ---- NEW diagonal (cross-modality) datasets: dataset-name -> modality-pair group (=> regime folder suffix) ----
DIAG_GROUP={"breastRNARNA":"rna_rna","crcRNARNA":"rna_rna","prostateRNARNA":"rna_rna",
            "liverRNAATAC":"rna_atac","kidneyRNAATAC":"rna_atac","cerebrumRNAATAC":"rna_atac","heartRNAATAC":"rna_atac",
            "mouseRNAmC":"rna_methylome","humanRNAmC":"rna_methylome",
            "brainRNAmC":"rna_methylome",
            "fallopianD3D4RNARNA":"rna_rna",
            "fallopianD3D5RNARNA":"rna_rna",
            "fallopianD3D6RNARNA":"rna_rna",
            "fallopianD3RNAATAC":"rna_atac",
            "fallopianD4D5RNARNA":"rna_rna",
            "fallopianD4D6RNARNA":"rna_rna",
            "fallopianD4RNAATAC":"rna_atac",
            "fallopianD5D6RNARNA":"rna_rna",
            "fallopianD5RNAATAC":"rna_atac",
            "fallopianD6RNAATAC":"rna_atac",
            "mopF003x2CRNAmC":"rna_methylome",
            "mopF004x3CRNAmC":"rna_methylome",
            "mopF006x4BRNAmC":"rna_methylome",
            "mopF008x5DRNAmC":"rna_methylome",
            "mopP1P2RNARNA":"rna_rna",
            "mopP1P3RNARNA":"rna_rna",
            "mopP1P4RNARNA":"rna_rna",
            "mopP1RNAATAC":"rna_atac",
            "mopP2P3RNARNA":"rna_rna",
            "mopP2P4RNARNA":"rna_rna",
            "mopP2RNAATAC":"rna_atac",
            "mopP3P4RNARNA":"rna_rna",
            "mopP3RNAATAC":"rna_atac",
            "mopP4RNAATAC":"rna_atac",
            "mutoD1D2RNARNA":"rna_rna",
            "mutoD1D3RNARNA":"rna_rna",
            "mutoD1D4RNARNA":"rna_rna",
            "mutoD1RNAATAC":"rna_atac",
            "mutoD2D3RNARNA":"rna_rna",
            "mutoD2D4RNARNA":"rna_rna",
            "mutoD2RNAATAC":"rna_atac",
            "mutoD3D4RNARNA":"rna_rna",
            "mutoD3RNAATAC":"rna_atac",
            "mutoD4RNAATAC":"rna_atac",
            "scicarR1RNAATAC":"rna_atac",
            "scicarR2RNAATAC":"rna_atac",
            "shareP5RNAATAC":"rna_atac",
            "shareP6RNAATAC":"rna_atac",
            "shareP7RNAATAC":"rna_atac",
            "shareP8RNAATAC":"rna_atac",
            "trevinoR1RNAATAC":"rna_atac",
            "trevinoR2RNAATAC":"rna_atac",
            "trevinoR3RNAATAC":"rna_atac"}
def ds_kind(ds):
    if ds in COASSAY: return "coassay"
    if ds.startswith("closed_"): return "tm_closed"   # diagonal closed sets also match here (closed_<DATASET>)
    if ds.startswith("open_"): return "tm_open"        # diagonal open sets also match here (open_<DATASET>)
    raise SystemExit(f"unknown dataset {ds}")
def cc_regime(ds):
    """runs/consensus regime folder for ds. TM -> closedset/openset (no suffix); diagonal -> <regime>_<group>."""
    dname=ds.split("_",1)[1]; pre="closedset" if ds.startswith("closed_") else "openset"
    return pre+("_"+DIAG_GROUP[dname] if dname in DIAG_GROUP else "")
def load_cc_coupling(ds,bc1,bc2):
    """Consensus Coupling (runs/consensus) aligned to (bc1,bc2) order -> G (n_mod1 x n_mod2), zeros where a
    requested cell is absent from the CC. Same orientation as the old v12 gamma (mod1 x mod2)."""
    cdir=PROJ/"runs/consensus"/cc_regime(ds)/ds
    if not (cdir/"coupling.npy").exists():
        # diagonal couplings live in the FLAT regime folder (runs/consensus/closedset),
        # not the group-suffixed one cc_regime() builds
        _flat="closedset" if ds.startswith("closed_") else "openset"
        _alt=PROJ/"runs/consensus"/_flat/ds
        if (_alt/"coupling.npy").exists():
            cdir=_alt
    Gm=np.load(str(cdir/"coupling.npy")).astype(float)
    c1=[x for x in open(str(cdir/"cells_mod1.txt")).read().split("\n") if x]
    c2=[x for x in open(str(cdir/"cells_mod2.txt")).read().split("\n") if x]
    p1={str(c):i for i,c in enumerate(bc1)}; p2={str(c):i for i,c in enumerate(bc2)}
    ri=np.array([p1.get(str(c),-1) for c in c1]); cj=np.array([p2.get(str(c),-1) for c in c2])
    vr=ri>=0; vc=cj>=0; G=np.zeros((len(bc1),len(bc2)))
    G[np.ix_(ri[vr],cj[vc])]=Gm[np.ix_(np.where(vr)[0],np.where(vc)[0])]
    return G

# ---------------- coassay: saved gamma + labels ----------------
GDIR={"v1":"diffusion_v1_hvg50000_eps1","v2":"diffusion_v2_hvg20000_eps1","v7":"diffusion_v7",
      "v8":"diffusion_v8_cos","v9":"diffusion_v9_orig","v10":"diffusion_v10_orig","v11":"diffusion_v11",
      "v12":"diffusion_v12","maxfuse":"diffusion_maxfuse"}
def load_saved_coupling(model,ds):
    dd=PROJ/"runs_foscttm"/GDIR[model]/ds
    if not (dd/"gamma.npy").exists():
        for alt in [PROJ/"runs/gamma_matrix"/ds, PROJ/"runs/ensemble"/ds]:
            if (alt/"gamma.npy").exists(): dd=alt; break   # runs_foscttm removed -> use current ensemble coupling
    if (dd/"gamma.npy").exists():
        G=np.load(dd/"gamma.npy").astype(np.float64); kind="prob"      # higher=better
    elif (dd/"pairwise_matrix_common.npy").exists():
        D=np.load(dd/"pairwise_matrix_common.npy").astype(np.float64)   # distance, lower=better
        pos=D[D>0]; s=float(np.median(pos)) if pos.size else 1.0
        G=np.exp(-(D-D.min(1,keepdims=True))/(s+TINY)); kind="prob"
    else: return None,None,None
    cells=None
    for fn in ("per_cell_foscttm.csv","per_cell_nfoscttm_metrics.csv"):
        fp=dd/fn
        if fp.exists(): cells=pd.read_csv(fp)["cell_id"].astype(str).to_numpy(); break
    if cells is None: cells=np.array([f"c{i}" for i in range(G.shape[0])])
    return G,cells,kind
def _labels_from_inventory(ds, cells):
    """Authoritative label file recorded in 3_data_integration/dataset_inventory.csv."""
    import numpy as np
    inv = PROJ/"3_data_integration"/"dataset_inventory.csv"
    if not inv.exists(): return None
    try:
        d = pd.read_csv(inv)
    except Exception:
        return None
    alias = [ds]
    if ds.startswith("datasets12_"): alias.append(ds.split("_",1)[1])
    if ds.startswith("datasets11_"):
        alias.append(ds.replace("datasets11_","").replace("_10pct","").replace("_3pct",""))
    row = d[d.dataset_id.astype(str).isin(alias)]
    if not len(row): return None
    lp = str(row.labels1_path.iloc[0])
    if not lp or lp == "nan" or not os.path.exists(lp): return None
    t = pd.read_csv(lp)
    if "cell_type" not in t.columns: return None
    idcol = "cell_id" if "cell_id" in t.columns else t.columns[0]
    m = dict(zip(t[idcol].astype(str), t["cell_type"].astype(str)))
    labs = np.array([m.get(str(c), "__NA__") for c in cells], dtype=object)
    ok = labs != "__NA__"
    if ok.mean() > 0.5 and len(set(labs[ok])) >= 2:
        print(f"[lt_bench] {ds}: labels from inventory ({ok.sum()} matched, "
              f"{len(set(labs[ok]))} types)", flush=True)
        return labs
    # names differ but the counts line up -> same prepared bundle order
    if len(t) == len(cells) and t["cell_type"].nunique() >= 2:
        labs = t["cell_type"].astype(str).to_numpy(dtype=object)
        print(f"[lt_bench] {ds}: inventory labels matched positionally "
              f"({len(labs)} cells, {len(set(labs))} types)", flush=True)
        return labs
    return None


def coassay_labels(ds,cells):
    import glob
    _inv = _labels_from_inventory(ds, cells)
    if _inv is not None: return _inv
    cands=[ds]
    if ds.startswith("datasets12_"): cands.append(ds.split("_",1)[1])
    # datasets11_CITE_BMMC_10pct lives under benchmarking_CITE_BMMC
    if ds.startswith("datasets11_"):
        cands.append(ds.replace("datasets11_","").replace("_10pct","").replace("_3pct",""))
    for c in cands:
        fs=glob.glob(str(PROJ/f"runs_integration/benchmarking_{c}/*/{c}/cells_with_labels.csv"))
        if not fs: fs=glob.glob(str(PROJ/f"runs_integration/benchmarking_{c}/**/cells_with_labels.csv"),recursive=True)
        if fs:
            d=pd.read_csv(fs[0])[["cell_id","cell_type"]].drop_duplicates("cell_id")
            m=dict(zip(d["cell_id"].astype(str),d["cell_type"].astype(str)))
            labs=np.array([m.get(str(x),"__NA__") for x in cells],dtype=object)
            if (labs!="__NA__").mean()>0.5 and len(set(labs[labs!="__NA__"]))>=2: return labs
    _short=[ds, ds.replace("datasets12_","")]
    if ds.startswith("datasets11_"):
        _short.append(ds.replace("datasets11_","").replace("_10pct","").replace("_3pct",""))
    for src in ["runs_data_integration_backup_20260707","runs_data_integration"]:   # DI-backup coassay label fallback
        for nm in _short:
            # the nested DI tree is <tool>/<regime>/<version>/<dataset>/, so the
            # two-level glob alone never reached it
            fs=(glob.glob(str(PROJ/f"{src}/*/{nm}/cells_with_labels.csv"))
                or glob.glob(str(PROJ/f"{src}/*/coassay/*/{nm}/cells_with_labels.csv")))
            if fs:
                d=pd.read_csv(fs[0])
                if "cell_type" in d.columns and "cell_id" in d.columns:
                    m=dict(zip(d["cell_id"].astype(str),d["cell_type"].astype(str)))
                    labs=np.array([m.get(str(x),"__NA__") for x in cells],dtype=object)
                    if (labs!="__NA__").mean()>0.3 and len(set(labs[labs!="__NA__"]))>=2: return labs
    # Positional fallback: the co-assay bundles label cells `rna_i` while the
    # coupling carries `cell_i`; both come from the same prepared bundle in the
    # same order. Only used when name matching failed AND the counts agree.
    for c in cands:
        fs=glob.glob(str(PROJ/f"runs_integration/benchmarking_{c}/*/{c}/cells_with_labels.csv"))
        if not fs: continue
        d=pd.read_csv(fs[0])
        if "cell_type" not in d.columns: continue
        if "modality" in d.columns:
            d=d[d["modality"].astype(str)=="mod1"]
        if len(d)==len(cells):
            labs=d["cell_type"].astype(str).to_numpy(dtype=object)
            if len(set(labs))<2:
                print(f"[lt_bench] {ds}: positional labels degenerate "
                      f"({len(set(labs))} type) -- rejected", flush=True)
                continue
            print(f"[lt_bench] {ds}: cell names unmatched -> positional labels "
                  f"({len(labs)} cells, {len(set(labs))} types)", flush=True)
            return labs
    return None

# ---------------- tabula-muris: load two batches + estimate cross-batch coupling ----------------
def load_mtx(prefix):
    X=sio.mmread(str(prefix)+".mtx").tocsr().astype(np.float64)
    bc=pd.read_csv(str(prefix)+"_barcodes.tsv",header=None)[0].astype(str).to_numpy()
    if X.shape[0]!=len(bc): X=X.T.tocsr()   # ensure cells x features
    return X,bc
def tm_paths(tissue, ds=None):
    if tissue in DIAG_GROUP:   # diagonal bundle: externals read raw counts + obs labels (barcodes/features == log)
        reg="closedset" if (ds and ds.startswith("closed_")) else "openset"
        base=PROJ/"Results/prepared_input"/reg/tissue
        return (str(base/"mod1_raw"), str(base/"mod2_raw"), str(base/"mod1_raw_obs.csv"), str(base/"mod2_raw_obs.csv"))
    base=PROJ/"Data/datasets/opensetV2/prepared"/tissue
    return (str(base/f"expr_{tissue}_10X_cell_by_feature"), str(base/f"expr_{tissue}_FACS_cell_by_feature"),
            str(base/f"{tissue}_10X_labels.csv"), str(base/f"{tissue}_FACS_labels.csv"))
def _diag_load(tissue, ds):
    """Diagonal dataset loader: mod1(RNA ref) + mod2(query) raw matrices, gene-symbol features, cell_type labels.
    Barcodes/features are identical across raw/log and to the CC coupling cells, so raw is used (single log1p in
    the model featuriser, exactly like TM counts) while alignment to the coupling stays exact."""
    reg="closedset" if (ds and ds.startswith("closed_")) else "openset"
    base=PROJ/"Results/prepared_input"/reg/tissue
    X1,bc1=load_mtx(str(base/"mod1_raw")); X2,bc2=load_mtx(str(base/"mod2_raw"))
    def feats(prefix,n):
        for suf in ("_features.tsv","_genes.tsv"):
            if os.path.exists(prefix+suf): return pd.read_csv(prefix+suf,header=None)[0].astype(str).to_numpy()
        return np.array([f"g{i}" for i in range(n)])
    f1=feats(str(base/"mod1_raw"),X1.shape[1]); f2=feats(str(base/"mod2_raw"),X2.shape[1])
    o1=pd.read_csv(str(base/"mod1_raw_obs.csv")); o2=pd.read_csv(str(base/"mod2_raw_obs.csv"))
    m1=dict(zip(o1["cell_id"].astype(str),o1["cell_type"].astype(str))); m2=dict(zip(o2["cell_id"].astype(str),o2["cell_type"].astype(str)))
    y1=np.array([m1.get(str(b),"__NA__") for b in bc1],dtype=object); y2=np.array([m2.get(str(b),"__NA__") for b in bc2],dtype=object)
    return X1,f1,bc1,y1,X2,f2,bc2,y2
def tm_load(tissue, ds=None):
    if tissue in DIAG_GROUP: return _diag_load(tissue, ds)
    p10,pfa,l10,lfa=tm_paths(tissue)
    X1,bc1=load_mtx(p10); X2,bc2=load_mtx(pfa)
    f1=pd.read_csv(p10+"_genes.tsv",header=None)[0].astype(str).to_numpy() if os.path.exists(p10+"_genes.tsv") else None
    # genes file name may differ; fall back to feature index
    def feats(prefix,n):
        for suf in ("_genes.tsv","_features.tsv"):
            if os.path.exists(prefix+suf): return pd.read_csv(prefix+suf,header=None)[0].astype(str).to_numpy()
        return np.array([f"g{i}" for i in range(n)])
    f1=feats(p10,X1.shape[1]); f2=feats(pfa,X2.shape[1])
    lab1=pd.read_csv(l10); lab2=pd.read_csv(lfa)
    c1=lab1.columns[0]; t1=lab1.columns[1]; c2=lab2.columns[0]; t2=lab2.columns[1]
    m1=dict(zip(lab1[c1].astype(str),lab1[t1].astype(str))); m2=dict(zip(lab2[c2].astype(str),lab2[t2].astype(str)))
    y1=np.array([m1.get(str(b),"__NA__") for b in bc1],dtype=object); y2=np.array([m2.get(str(b),"__NA__") for b in bc2],dtype=object)
    return X1,f1,bc1,y1,X2,f2,bc2,y2
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler, normalize
def _shared(X1,f1,X2,f2,nhvg=2000):
    common,i1,i2=np.intersect1d(f1,f2,return_indices=True)
    if len(common)<50: raise SystemExit(f"shared genes {len(common)}")
    A=X1[:,i1]; B=X2[:,i2]
    A=A.toarray() if sp.issparse(A) else np.asarray(A); B=B.toarray() if sp.issparse(B) else np.asarray(B)
    A=np.log1p(A); B=np.log1p(B)
    v=A.var(0)+B.var(0); idx=np.argsort(v)[-min(nhvg,len(common)):]
    return A[:,idx],B[:,idx]
def _rk(D): return np.apply_along_axis(lambda r: (np.argsort(np.argsort(r))+1), 1, D).astype(np.float64)
def base_v7(Z1,Z2):                                   # shared-HVG cosine (v7)
    a=normalize(Z1); b=normalize(Z2); return 1.0-a@b.T
def base_v1(Z1,Z2,k=50):                              # PCA-Sinkhorn essence (v1)
    Z=np.vstack([Z1,Z2]); Z=StandardScaler().fit_transform(Z)
    P=TruncatedSVD(min(k,Z.shape[1]-1),random_state=0).fit_transform(Z); P1=P[:len(Z1)]; P2=P[len(Z1):]
    P1=normalize(P1); P2=normalize(P2); return 1.0-P1@P2.T
def base_v8(Z1,Z2,k=20):                              # diagonal/Seurat CCA (SVD of cross-covariance; unpaired-safe)
    Z1=StandardScaler().fit_transform(Z1); Z2=StandardScaler().fit_transform(Z2)
    M=Z1@Z2.T                                         # n1 x n2 cell-cell cross-correlation
    k=min(k,min(M.shape)-1)
    U,s,Vt=np.linalg.svd(M,full_matrices=False)
    E1=U[:,:k]*s[:k]; E2=Vt[:k,:].T*s[:k]
    return 1.0-normalize(E1)@normalize(E2).T
def base_v2(Z1,Z2,k=30):                              # structure/correlation map (v2 GW essence, lite)
    P1=TruncatedSVD(min(k,Z1.shape[1]-1),random_state=0).fit_transform(StandardScaler().fit_transform(Z1))
    P2=TruncatedSVD(min(k,Z2.shape[1]-1),random_state=0).fit_transform(StandardScaler().fit_transform(Z2))
    return 1.0-normalize(P1)@normalize(P2).T
def estimate_crossbatch(model,X1,f1,X2,f2):
    """Self-contained cross-batch coupling (the original inline backends were removed with the
    label_transfer_models package). Faithful re-implementation of each model's correspondence recipe."""
    Z1,Z2=_shared(X1,f1,X2,f2)
    need={"v1":["v1"],"v2":["v2"],"v7":["v7"],"v8":["v8"],"v9":["v7","v8"],
          "v10":["v1","v7","v8"],"v11":["v1","v2","v7","v8"],"v12":["v1","v7","v8"],"maxfuse":["v7","v8"]}[model]
    fn={"v1":base_v1,"v2":base_v2,"v7":base_v7,"v8":base_v8}
    base={b:fn[b](Z1,Z2) for b in need}
    rk={b:_rk(base[b]) for b in base}
    if model in ("v1","v2","v7","v8"): D=rk[model]
    elif model in ("v9","maxfuse"): D=0.5*(rk["v7"]+rk["v8"])
    elif model=="v10": D=0.5*rk["v1"]+0.25*rk["v7"]+0.25*rk["v8"]
    elif model=="v11": D=(rk["v1"]+rk["v2"]+0.5*(rk["v7"]+rk["v8"]))/3.0
    elif model=="v12":
        comps={"v1":rk["v1"],"v9":0.5*(rk["v7"]+rk["v8"])}
        def sharp(R): a=(R==R.min(1,keepdims=True)); b=(R==R.min(0,keepdims=True)); return float((a&b).mean())+1e-6
        w={k:sharp(v) for k,v in comps.items()}; sw=sum(w.values()); D=sum((w[k]/sw)*v for k,v in comps.items())
    G=1.0/(_rk(D)+1.0)
    return G

# ---------------- label transfer + metrics ----------------
def transfer(G, ref_lab, query_axis):
    """G rows=A cells, cols=B cells (higher=better). Predict query cells' labels from ref labels via weighted vote.
       query_axis='cols' -> predict B (cols) using A(row) labels ref_lab(len=rows).
       returns pred labels (len=n_query), confidence (max share)."""
    classes=sorted(set(ref_lab[ref_lab!="__NA__"]))
    cidx={c:i for i,c in enumerate(classes)}
    keep=ref_lab!="__NA__"
    if query_axis=="cols":
        Y=np.zeros((G.shape[0],len(classes)));
        for i in np.where(keep)[0]: Y[i,cidx[ref_lab[i]]]=1.0
        S=G.T@Y                          # n_cols x C
    else:
        Y=np.zeros((G.shape[1],len(classes)))
        for j in np.where(keep)[0]: Y[j,cidx[ref_lab[j]]]=1.0
        S=G@Y                            # n_rows x C
    tot=S.sum(1)+TINY; conf=S.max(1)/tot
    pred=np.array([classes[k] for k in S.argmax(1)],dtype=object)
    return pred,conf
def closed_metrics(true,pred):
    m=true!="__NA__"; t=true[m]; p=pred[m]
    return dict(n=int(m.sum()),overall_accuracy=float((t==p).mean()),
                balanced_accuracy=float(balanced_accuracy_score(t,p)),
                macro_f1=float(f1_score(t,p,average="macro",zero_division=0)),
                weighted_f1=float(f1_score(t,p,average="weighted",zero_division=0)),
                coverage=1.0)
def openset_metrics(true,pred,conf,ref_classes):
    m=true!="__NA__"; t=np.array(["__unknown__" if x not in ref_classes else x for x in true[m]],dtype=object)
    p0=pred[m].copy(); c=conf[m]; n_unknown=int((t=="__unknown__").sum())
    best=None
    for q in np.linspace(0,0.99,60):
        thr=np.quantile(c,q); pp=np.where(c<thr,"__unknown__",p0)
        mf=f1_score(t,pp,average="macro",zero_division=0)
        if best is None or mf>best[0]: best=(mf,thr,pp)
    mf,thr,pp=best
    return dict(n_query=int(m.sum()),n_unknown=n_unknown,threshold=float(thr),
                macro_class_accuracy=float(balanced_accuracy_score(t,pp)),
                unknown_class_accuracy=float(((pp=="__unknown__")&(t=="__unknown__")).sum()/max(1,n_unknown)),
                macro_f1=float(mf),weighted_f1=float(f1_score(t,pp,average="weighted",zero_division=0)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--dataset",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); model=a.model; ds=a.dataset; kind=ds_kind(ds); row={"model":model,"dataset":ds,"kind":kind}
    if kind=="coassay":
        G,cells,_=load_saved_coupling(model,ds)
        if G is None: row["status"]="no_gamma"; json.dump(row,open(a.out,"w")); return
        labs=coassay_labels(ds,cells)
        if labs is None: row["status"]="no_labels"; json.dump(row,open(a.out,"w")); return
        # both directions, square paired
        predB,confB=transfer(G,labs,"cols"); predA,confA=transfer(G,labs,"rows")
        true=np.concatenate([labs,labs]); pred=np.concatenate([predA,predB])
        row.update(closed_metrics(true,pred)); row["status"]="ok"
    else:
        tissue=ds.split("_",1)[1]
        X1,f1,bc1,y1,X2,f2,bc2,y2=tm_load(tissue,ds)
        gpath=PROJ/"runs_label_transfer/gammas"/model/ds; gpath.mkdir(parents=True,exist_ok=True)
        if tissue in DIAG_GROUP:                              # diagonal -> use the precomputed CC coupling (no re-estimation)
            G=load_cc_coupling(ds,bc1,bc2)
        elif (gpath/"coupling.npy").exists(): G=np.load(gpath/"coupling.npy")
        else:
            G=estimate_crossbatch(model,X1,f1,X2,f2); np.save(gpath/"coupling.npy",G.astype(np.float32))
            pd.DataFrame({"row_cell":bc1}).to_csv(gpath/"row_cells.csv",index=False); pd.DataFrame({"col_cell":bc2}).to_csv(gpath/"col_cells.csv",index=False)
        # transfer 10X(ref,rows)->FACS(query,cols)
        pred,conf=transfer(G,y1,"cols"); true=y2; ref_classes=set(y1[y1!="__NA__"])
        if kind=="tm_closed":
            mm=np.array([t in ref_classes for t in true])  # closed: keep only query types present in ref
            row.update(closed_metrics(true[mm & (true!="__NA__")],pred[mm & (true!="__NA__")])); row["status"]="ok"
        else:
            row.update(openset_metrics(true,pred,conf,ref_classes)); row["status"]="ok"
    json.dump(row,open(a.out,"w"),indent=2); log(f"DONE {model} {ds}: {row.get('status')}")
if __name__=="__main__": main()
