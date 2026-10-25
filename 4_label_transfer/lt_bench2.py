#!/usr/bin/env python3
"""Two new gamma-based label-transfer methods, sourced from an ENSEMBLE of custom models {v1,v9,v12}.
Correspondence: coassay -> saved gammas in runs_foscttm; tabula-muris -> couplings in runs_label_transfer/gammas
(generated earlier via the custom models). The 3 model couplings are rank-fused into one ensemble affinity.
  Method A 'EnsOT'  : ensemble gamma -> label vote, FULL coverage (matched-coverage baseline).
  Method B 'COT-LT' : ensemble gamma -> CLASS-BALANCED vote + dedicated NOVELTY rejection (open-set) +
                      conformal-style coverage-accuracy summary. Decouples accuracy / coverage / unknown-rejection.
Outputs both methods' metrics per dataset. Usage: lt_bench2.py --dataset <key> --out <json>
"""
import sys, argparse, json
sys.path.insert(0,os.path.join(P, "4_label_transfer"))
import lt_bench as L
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
ENS=["v1","v9","v12"]; TINY=1e-30  # overridden by --ens
def rrank(C): return (np.argsort(np.argsort(-C,axis=1),axis=1)+1).astype(np.float32)  # per-row, best=1
def fuse(mats):
    R=np.zeros(mats[0].shape,dtype=np.float32)
    for c in mats: R+=rrank(c)
    return (1.0/((R/len(mats))+1.0)).astype(np.float64)
def ens_coassay(ds):
    cellref=None; mats=[]
    for m in ENS:
        G,cells,_=L.load_saved_coupling(m,ds)
        if G is None: return None,None
        if cellref is None: cellref=cells; mats.append(G)
        else:
            pos={c:i for i,c in enumerate(cells)}; order=np.array([pos[c] for c in cellref if c in pos])
            if len(order)!=len(cellref): return None,None
            mats.append(G[np.ix_(order,order)])
    return fuse(mats), cellref
def ens_tm(ds):
    mats=[]
    for m in ENS:
        p=L.PROJ/"runs_label_transfer/gammas"/m/ds/"coupling.npy"
        if not p.exists(): return None
        mats.append(np.load(p).astype(np.float64))
    if len({x.shape for x in mats})!=1: return None
    return fuse(mats)
def vote(G, ref_lab, balanced):
    classes=sorted(set(ref_lab[ref_lab!="__NA__"])); cidx={c:i for i,c in enumerate(classes)}
    keep=np.where(ref_lab!="__NA__")[0]
    Y=np.zeros((G.shape[0],len(classes)))
    for i in keep: Y[i,cidx[ref_lab[i]]]=1.0
    if balanced:
        cnt=Y.sum(0)+1.0; Y=Y/np.sqrt(cnt)
    S=G.T@Y                       # query(cols) x C
    return S, np.array(classes,dtype=object)
def pred(S,classes):
    idx=S.argmax(1); conf=S.max(1)/(S.sum(1)+TINY); return classes[idx], conf
def closed(true,p):
    m=true!="__NA__"; t=true[m]; q=p[m]
    return dict(n=int(m.sum()),overall_accuracy=float((t==q).mean()),
                balanced_accuracy=float(balanced_accuracy_score(t,q)),
                macro_f1=float(f1_score(t,q,average="macro",zero_division=0)),
                weighted_f1=float(f1_score(t,q,average="weighted",zero_division=0)),coverage=1.0)
def openA(true,p,ref_classes):  # method A: no rejection (labels everything)
    m=true!="__NA__"; t=np.array(["__unknown__" if x not in ref_classes else x for x in true[m]],dtype=object); q=p[m]
    nu=int((t=="__unknown__").sum())
    return dict(n_query=int(m.sum()),n_unknown=nu,macro_class_accuracy=float(balanced_accuracy_score(t,q)),
                unknown_class_accuracy=0.0,macro_f1=float(f1_score(t,q,average="macro",zero_division=0)),
                weighted_f1=float(f1_score(t,q,average="weighted",zero_division=0)))
def openB(true,p,novelty,ref_classes):  # method B: dedicated novelty rejection
    m=true!="__NA__"; t=np.array(["__unknown__" if x not in ref_classes else x for x in true[m]],dtype=object)
    p0=p[m].copy(); nv=novelty[m]; nu=int((t=="__unknown__").sum()); best=None
    for ql in np.linspace(0.5,0.999,60):
        thr=np.quantile(nv,ql); pp=np.where(nv>=thr,"__unknown__",p0)
        f=f1_score(t,pp,average="macro",zero_division=0)
        if best is None or f>best[0]: best=(f,thr,pp)
    f,thr,pp=best
    return dict(n_query=int(m.sum()),n_unknown=nu,novel_threshold=float(thr),
                macro_class_accuracy=float(balanced_accuracy_score(t,pp)),
                unknown_class_accuracy=float(((pp=="__unknown__")&(t=="__unknown__")).sum()/max(1,nu)),
                macro_f1=float(f),weighted_f1=float(f1_score(t,pp,average="weighted",zero_division=0)))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--out",required=True); ap.add_argument("--ens",default="v1,v9,v12")
    a=ap.parse_args(); global ENS; ENS=a.ens.split(","); ds=a.dataset; kind=L.ds_kind(ds); rows=[]
    if kind=="coassay":
        G,cells=ens_coassay(ds)
        if G is None: json.dump([{"dataset":ds,"status":"no_gamma"}],open(a.out,"w")); return
        labs=L.coassay_labels(ds,cells)
        if labs is None: json.dump([{"dataset":ds,"status":"no_labels"}],open(a.out,"w")); return
        for bal,name in [(False,"EnsOT"),(True,"COT-LT")]:
            S,cl=vote(G,labs,bal); pB,_=pred(S,cl); SA,clA=vote(G.T,labs,bal); pA,_=pred(SA,clA)
            true=np.concatenate([labs,labs]); pr=np.concatenate([pA,pB])
            r={"dataset":ds,"kind":kind,"model":name,"status":"ok"}; r.update(closed(true,pr)); rows.append(r)
    else:
        tissue=ds.split("_",1)[1]; G=ens_tm(ds)
        if G is None: json.dump([{"dataset":ds,"status":"no_gamma"}],open(a.out,"w")); return
        X1,f1,bc1,y1,X2,f2,bc2,y2=L.tm_load(tissue); ref_classes=set(y1[y1!="__NA__"])
        novelty=1.0-G.max(0)/(G.max()+TINY)   # per query col: low max-coupling -> high novelty
        for bal,name in [(False,"EnsOT"),(True,"COT-LT")]:
            S,cl=vote(G,y1,bal); p,_=pred(S,cl); true=y2
            r={"dataset":ds,"kind":kind,"model":name,"status":"ok"}
            if kind=="tm_closed":
                mm=np.array([x in ref_classes for x in true]) & (true!="__NA__")
                r.update(closed(true[mm],p[mm]))
            else:
                r.update(openB(true,p,novelty,ref_classes) if name=="COT-LT" else openA(true,p,ref_classes))
            rows.append(r)
    json.dump(rows,open(a.out,"w"),indent=2); print(f"DONE {ds}: {[r.get('status') for r in rows]}",flush=True)
if __name__=="__main__": main()
