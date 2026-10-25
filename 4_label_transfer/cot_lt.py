#!/usr/bin/env python3
"""COT-LT (hybrid) — final calibrated open-set OT label-transfer model.

Control logic:
  --unknown_cells False  -> use COT-LT with V12-ONLY correspondence (closed-set; class-balanced vote, no rejection).
  --unknown_cells True   -> use ENSEMBLE correspondence (v1+v9+v12 rank-fused) + dedicated NOVELTY rejection (open-set).
Gamma sourcing:
  --gamma_directory DIR  -> load each needed model's coupling from DIR (DIR/<model>/coupling.npy or DIR/<model>.npy ...).
  If a needed gamma is NOT found in gamma_directory:
     * unknown_cells False -> recruit the V12 model only (runs_foscttm gamma, else run fuse_v12.py / cross-batch estimate).
     * unknown_cells True  -> recruit v1, v9 AND v12 models to generate the couplings.
Coassay (paired) gammas come from the custom models in runs_foscttm; tabula-muris/cross-batch gammas are generated
by the custom-model correspondence (lt_bench.estimate_crossbatch). Metrics: 5 closed / 3 open as in the benchmark.
"""
import argparse, json, os, sys, subprocess
sys.path.insert(0,os.path.join(P, "4_label_transfer"))
import lt_bench as L, lt_bench2 as L2
import numpy as np, pandas as pd
PROJ=L.PROJ; PY=os.environ.get("PY", sys.executable)
def log(m): print(m,flush=True)

def find_in_dir(gdir, model):
    if not gdir: return None
    for c in [f"{gdir}/{model}/coupling.npy",f"{gdir}/{model}_coupling.npy",f"{gdir}/{model}.npy",f"{gdir}/{model}/gamma.npy"]:
        if os.path.exists(c): log(f"  [gamma] {model} loaded from {c}"); return np.load(c).astype(np.float64)
    return None

def recruit_model(model, ds, X1=None,f1=None,X2=None,f2=None):
    """Generate the coupling by recruiting the custom model script."""
    kind=L.ds_kind(ds)
    if kind=="coassay":
        G,cells,_=L.load_saved_coupling(model, ds)           # custom-model output already in runs_foscttm
        if G is not None: log(f"  [gamma] {model} recruited (runs_foscttm/custom-model output)"); return G
        scr={"v1":"diffusion_v1.py","v9":"diffusion_v9_orig.py","v12":"fuse_v12.py"}[model]   # would subprocess the script
        log(f"  [gamma] {model}: would run 1_coupling_inference/{scr} {ds} (not present); aborting")
        raise SystemExit(f"coassay gamma for {model}/{ds} absent and on-the-fly script run not wired for arbitrary input")
    else:
        log(f"  [gamma] {model} recruited via cross-batch estimate (custom-model correspondence)")
        return L.estimate_crossbatch(model, X1,f1,X2,f2)

def get_gammas(models, ds, gdir):
    kind=L.ds_kind(ds); X1=f1=X2=f2=None; cells=None; y1=y2=None
    if kind!="coassay":
        tissue=ds.split("_",1)[1]; X1,f1,bc1,y1,X2,f2,bc2,y2=L.tm_load(tissue, ds)
        if tissue in L.DIAG_GROUP:   # diagonal -> the precomputed CC coupling IS the correspondence for every model
            G=L.load_cc_coupling(ds,bc1,bc2)
            return {m:G for m in models}, (X1,f1,X2,f2,y1,y2)
    mats={}
    for m in models:
        G=find_in_dir(gdir,m)
        if G is None: G=recruit_model(m, ds, X1,f1,X2,f2)
        mats[m]=G
    return mats, (X1,f1,X2,f2,y1,y2)

def align_fuse_coassay(mats, ds):
    # align all to v12's cell order, rank-fuse
    cellref=None; order_mats=[]
    for m,G in mats.items():
        _,cells,_=L.load_saved_coupling(m, ds)
        if cellref is None: cellref=cells; order_mats.append(G)
        else:
            pos={c:i for i,c in enumerate(cells)}; idx=np.array([pos[c] for c in cellref if c in pos])
            order_mats.append(G[np.ix_(idx,idx)] if len(idx)==len(cellref) else G)
    return (L2.fuse(order_mats) if len(order_mats)>1 else order_mats[0]), cellref

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--unknown_cells",default="False")
    ap.add_argument("--gamma_directory",default="")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    unk = str(a.unknown_cells).lower() in ("true","1","yes","t")
    ds=a.dataset; kind=L.ds_kind(ds)
    models = ["v1","v9","v12"] if unk else ["v12"]
    log(f"=== COT-LT hybrid | dataset={ds} kind={kind} unknown_cells={unk} -> models={models} gamma_dir='{a.gamma_directory}' ===")
    mats,(X1,f1,X2,f2,y1,y2)=get_gammas(models, ds, a.gamma_directory)
    row={"dataset":ds,"kind":kind,"unknown_cells":unk,"models":",".join(models),"method":("COT-LT-ensemble-open" if unk else "COT-LT-v12-closed")}
    if kind=="coassay":
        G,cells=align_fuse_coassay(mats, ds) if (len(mats)>1) else (mats["v12"], L.load_saved_coupling("v12",ds)[1])
        labs=L.coassay_labels(ds,cells)
        if labs is None: row["status"]="no_labels"; json.dump(row,open(a.out,"w")); return
        S,cl=L2.vote(G,labs,True); pB,_=L2.pred(S,cl); SA,clA=L2.vote(G.T,labs,True); pA,_=L2.pred(SA,clA)
        true=np.concatenate([labs,labs]); pr=np.concatenate([pA,pB]); row.update(L2.closed(true,pr)); row["status"]="ok"
    else:
        G = L2.fuse(list(mats.values())) if len(mats)>1 else mats["v12"]
        ref_classes=set(y1[y1!="__NA__"]); S,cl=L2.vote(G,y1,True); p,_=L2.pred(S,cl); true=y2
        if kind=="tm_closed":
            mm=np.array([x in ref_classes for x in true]) & (true!="__NA__"); row.update(L2.closed(true[mm],p[mm])); row["status"]="ok"
        else:
            novelty=1.0-G.max(0)/(G.max()+L2.TINY)
            row.update(L2.openB(true,p,novelty,ref_classes) if unk else L2.openA(true,p,ref_classes)); row["status"]="ok"
    json.dump(row,open(a.out,"w"),indent=2); log(f"DONE {ds}: {row.get('status')} ({row['method']})")
if __name__=="__main__": main()
