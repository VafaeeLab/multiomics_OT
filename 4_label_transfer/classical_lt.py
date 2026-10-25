import json, sys, re, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.join(P, "4_label_transfer"))
sys.path.insert(0,os.path.join(P, "1_coupling_inference"))
sys.path.insert(0,os.path.join(P, "1_coupling_inference"))  # dv_common lives here
import lt_bench as L, lt_bench2 as L2, cot_lt as C
import numpy as np, scipy.sparse as sp, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
TINY=1e-9
def dense(X): return X.toarray() if sp.issparse(X) else np.asarray(X,float)
def harmonize(names):
    out=[]
    for x in names:
        x=str(x).upper().strip(); x=re.sub(r"^(HU|MO|HUMAN|MOUSE)[._-]","",x); x=re.sub(r"_[A-Z0-9.]+$","",x); x=re.sub(r"\.\d+$","",x); out.append(x)
    return out
def embed_common(Xr,fr,Xq,fq,minc=20):
    hr=harmonize(fr); hq=harmonize(fq); qp={}
    for i,g in enumerate(hq):
        if g and g not in qp: qp[g]=i
    pairs=[(i,qp[g]) for i,g in enumerate(hr) if g in qp]
    if len(pairs)<minc: return None,None,False
    ir=[a for a,_ in pairs]; iq=[b for _,b in pairs]
    def nz(X): X=np.log1p(np.clip(dense(X),0,None)); return StandardScaler().fit_transform(X).astype(np.float32)
    return nz(Xr[:,ir]),nz(Xq[:,iq]),True
def gv(S,b=3.0):
    S=np.clip(np.asarray(S,float),0,None); S=S/(S.max(1,keepdims=True)+TINY); S=S**b; return S/(S.sum(1,keepdims=True)+TINY)
def models(n_classes):
    return {
      "RF":  RandomForestClassifier(n_estimators=400,max_depth=None,n_jobs=-1,class_weight="balanced",random_state=0),
      "HGB": HistGradientBoostingClassifier(max_iter=400,learning_rate=0.08,max_depth=None,l2_regularization=1.0,random_state=0),
      "kNN": KNeighborsClassifier(n_neighbors=15,weights="distance",n_jobs=-1),
      "SVM": SVC(C=2.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=True,random_state=0),
    }
def pick_models(n_classes, ndim):
    m=models(n_classes)
    if ndim>2000: m={k:m[k] for k in ("HGB","SVM")}  # RF/kNN are O(n^2*d)-intractable on high-dim coassay (MULTIOME); the benchmark table only uses HGB/SVM
    return m
def proba(clf,Zr,yr,Zq,classes):
    if isinstance(clf,SVC):   # rbf+proba SVM hangs on large/high-dim coassay (MULTIOME ATAC) -> cap ref + PCA-reduce
        Zr=np.asarray(Zr); Zq=np.asarray(Zq); yr=np.asarray(yr)
        if len(Zr)>3000:
            rng=np.random.RandomState(0); s=rng.choice(len(Zr),3000,replace=False); Zr=Zr[s]; yr=yr[s]
        if Zr.shape[1]>2000:
            from sklearn.decomposition import PCA
            p=PCA(n_components=min(64,Zr.shape[0]-1,Zr.shape[1]),random_state=0).fit(Zr); Zr=p.transform(Zr); Zq=p.transform(Zq)
    clf.fit(Zr,yr); cl=list(clf.classes_); P=clf.predict_proba(Zq)
    Pa=np.zeros((len(Zq),len(classes))); ci={c:i for i,c in enumerate(classes)}
    for j,c in enumerate(cl):
        if c in ci: Pa[:,ci[c]]=P[:,j]
    return Pa
def main():
    ds=sys.argv[1]; kind=L.ds_kind(ds); rows=[]
    if kind=="coassay":
        import dv_common as DC
        e=json.load(open(L.PROJ/"1_coupling_inference/foscttm_input_map.json"))[ds]
        XA,XB,cells,fA,fB=DC.load_align(e["rna"],e["atac"],e["ori1"],e["ori2"],log=lambda *x:None)
        labs=L.coassay_labels(ds,cells); keep=labs!="__NA__"; cls=sorted(set(labs[keep])); ca=np.array(cls,dtype=object)
        ZA,ZB,sh=embed_common(XA,fA,XB,fB)
        G,gc,_=L.load_saved_coupling("v12",ds); G=np.asarray(G,float)
        pos={str(c):i for i,c in enumerate(gc)}; idx=np.array([pos[str(c)] for c in cells if str(c) in pos]); kk=np.where(keep)[0]
        Gk=G[np.ix_(idx,idx)][np.ix_(kk,kk)] if len(idx)==len(cells) else None
        true=np.concatenate([labs[keep],labs[keep]])
        SBv=gv(L2.vote(Gk,labs[keep],True)[0]) if Gk is not None else None
        SAv=gv(L2.vote(Gk.T,labs[keep],True)[0]) if Gk is not None else None
        for nm,clf in pick_models(len(cls),ZA.shape[1]).items():
            from sklearn.base import clone
            PB=proba(clone(clf),ZA[keep],labs[keep],ZB[keep],cls); PA=proba(clone(clf),ZB[keep],labs[keep],ZA[keep],cls)
            m=L2.closed(true,np.concatenate([ca[PA.argmax(1)],ca[PB.argmax(1)]])); rows.append(dict(dataset=ds,model=nm,gamma="none",overall=m["overall_accuracy"],macro_f1=m["macro_f1"]))
            if Gk is not None:
                best=None; blam=0.2
                for lam in [0.2,0.4,0.6,0.8]:
                    mm=L2.closed(true,np.concatenate([ca[((1-lam)*PA+lam*SAv).argmax(1)],ca[((1-lam)*PB+lam*SBv).argmax(1)]]))
                    if best is None or mm["overall_accuracy"]>best["overall_accuracy"]: best=mm; blam=lam
                rows.append(dict(dataset=ds,model=nm,gamma="v12",overall=best["overall_accuracy"],macro_f1=best["macro_f1"],lam=blam))
    else:
        tissue=ds.split("_",1)[1]; X1,f1,bc1,y1,X2,f2,bc2,y2=L.tm_load(tissue,ds)
        ref_classes=set(y1[y1!="__NA__"]); keep=y1!="__NA__"; cls=sorted(set(y1[keep])); ca=np.array(cls,dtype=object)
        Z1,Z2,sh=embed_common(X1,f1,X2,f2); op=ds.startswith("open_")
        mats,_=C.get_gammas(["v12"],ds,""); G=np.asarray(mats["v12"],float); Sv=gv(L2.vote(G[np.where(keep)[0],:],y1[keep],True)[0])
        def metric(P):
            pred=ca[P.argmax(1)]
            if op: m=L2.openB(y2,pred,1.0-P.max(1),ref_classes); return m["macro_class_accuracy"],m["macro_f1"]
            mm=np.array([x in ref_classes for x in y2])&(y2!="__NA__"); m=L2.closed(y2[mm],pred[mm]); return m["overall_accuracy"],m["macro_f1"]
        for nm,clf in pick_models(len(cls),Z1.shape[1]).items():
            from sklearn.base import clone
            P=proba(clone(clf),Z1[keep],y1[keep],Z2,cls); o,f=metric(P); rows.append(dict(dataset=ds,model=nm,gamma="none",overall=o,macro_f1=f))
            best=None; blam=0.2
            for lam in [0.2,0.4,0.6,0.8]:
                o,f=metric((1-lam)*P+lam*Sv)
                if best is None or o>best[0]: best=(o,f); blam=lam
            rows.append(dict(dataset=ds,model=nm,gamma="v12",overall=best[0],macro_f1=best[1],lam=blam))
    pd.DataFrame(rows).to_csv(L.PROJ/"runs_label_transfer"/("CLASSIC_%s.csv"%ds),index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
