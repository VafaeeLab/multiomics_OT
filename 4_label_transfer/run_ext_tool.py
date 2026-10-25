import sys, os, json, subprocess, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,os.path.join(P, "4_label_transfer"))
sys.path.insert(0,os.path.join(P, "1_coupling_inference"))
sys.path.insert(0,os.path.join(P, "1_coupling_inference"))  # dv_common lives here
import lt_bench as L, lt_bench2 as L2, cot_lt as C
import gfuse as GF
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
P = os.environ.get("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration")
OT_ROOT = os.environ.get("OT_ROOT", P + "/runs")
AMB_ROOT = os.environ.get("AMB_ROOT", P + "/runs_ambiguity")
DI_ROOT = os.environ.get("DI_ROOT", P + "/runs_data_integration")
LT_ROOT = os.environ.get("LT_ROOT", P + "/runs_label_transfer")
PROJ = P
RLIB=PROJ+"/R_analysis/R_libs_4.3"; TINY=1e-9
LAM_MAX=0.8
SCR={"SingleR":("R","run_singler_label_transfer.R"),"Symphony":("R","run_symphony_label_transfer.R"),
     "scmap":("R","run_scmap_label_transfer.R"),"Seurat":("R","run_seurat_transferdata_label_transfer.R"),
     "ItClust":("PY","run_itclust_label_transfer.py"),"Concerto":("PY","run_concerto_label_transfer.py"),"Harmony":("PY","run_harmony_label_transfer.py")}
def _harm(path,outp,adt=False):
    import pandas as pd, re
    try:
        from adt_map import ADT2GENE
    except Exception:
        ADT2GENE={}
    f=pd.read_csv(path,header=None)[0].astype(str).tolist()
    out=[]
    for x in f:
        # strip a single make.unique ".N" first (CD86.1 -> CD86), then species prefix, then clone suffix
        h=re.sub(r"\.\d+$","",x.strip()); h=re.sub(r"^(HU|MO|HUMAN|MOUSE)[._-]","",h.upper()); h=re.sub(r"_[A-Z0-9.]+$","",h)
        if adt and h in ADT2GENE: h=ADT2GENE[h]
        out.append(h)
    pd.Series(out).to_csv(outp,header=False,index=False); return outp
def to_mtx(p):
    if p.endswith(".mtx"): return p
    return p.replace("_cell_by_feature/","_cell_by_feature_mtx/").replace(".rds",".mtx")
def write_labels(cellids, labs, path):
    pd.DataFrame({"cell_id":cellids,"cell_type":labs}).to_csv(path,index=False)

CITE_ALIAS = {"BF21": "datasets12_BF21", "BM27": "datasets12_BM27",
              "WF26": "datasets12_WF26", "WM34": "datasets12_WM34"}


def _reconcile_roster():
    """Register the current co-assay roster into lt_bench, and return the name
    that foscttm_input_map is keyed by."""
    try:
        m = json.load(open(PROJ + "/1_coupling_inference/foscttm_input_map.json"))
    except Exception:
        m = {}
    co = set(getattr(L, "COASSAY", set())) | set(m.keys())
    for short, legacy in CITE_ALIAS.items():
        if legacy in co:
            co.add(short)
    L.COASSAY = co
    return m


def _resolve_ds(ds, m):
    """Roster name -> the name lt_bench understands."""
    if ds in getattr(L, "COASSAY", set()) or ds in m:
        return ds
    if ds.startswith(("closed_", "open_")):
        return ds
    for probe, pre in (
            (PROJ + "/Results/prepared_input/closedset/" + ds, "closed_"),
            (PROJ + "/Results/prepared_input/closedset_tabula_muris/" + ds, "closed_"),
            (PROJ + "/runs/non-coassay/closedset/closed_" + ds, "closed_"),
            (PROJ + "/runs/non-coassay/openset/open_" + ds, "open_")):
        if os.path.isdir(probe):
            return pre + ds
    return "closed_" + ds


def _map_key(ds, m):
    if ds in m:
        return ds
    return CITE_ALIAS.get(ds, ds)


def main():
    tool=sys.argv[1]; ds=sys.argv[2]
    _FMAP=_reconcile_roster()
    _ds_in=ds
    ds=_resolve_ds(ds,_FMAP)
    if ds!=_ds_in:
        print('resolved %s -> %s' % (_ds_in, ds), flush=True)
    rest=[a for a in sys.argv[3:]]
    refuse="--refuse" in rest; extra=[a for a in rest if a!="--refuse"]
    kind=L.ds_kind(ds); lang,script=SCR[tool]; out=f"{PROJ}/runs_label_transfer/ext_run/{tool}_{_ds_in}"; os.makedirs(out,exist_ok=True)
    env=dict(os.environ); env["R_LIBS"]=RLIB
    pf=os.path.join(out,"standardized_predictions_raw.csv"); res={"tool":tool,"dataset":ds,"kind":kind}
    if not refuse:
        if kind!="coassay":
            t=ds.split("_",1)[1]; p10,pfa,l10,lfa=L.tm_paths(t, ds)
            args=["--dataset_id",t,"--x1",p10+".mtx","--barcodes1",p10+"_barcodes.tsv","--features1",p10+"_features.tsv",
                  "--x2",pfa+".mtx","--barcodes2",pfa+"_barcodes.tsv","--features2",pfa+"_features.tsv","--labels1",l10,"--labels2",lfa,
                  "--out_dir",out,"--label_id_col","cell_id","--label_col","cell_type"]
        else:
            e=_FMAP[_map_key(ds,_FMAP)]
            x1=to_mtx(e["rna"]); x2=to_mtx(e["atac"]); b1=x1.replace(".mtx","_barcodes.tsv"); f1=x1.replace(".mtx","_features.tsv")
            b2=x2.replace(".mtx","_barcodes.tsv"); f2=x2.replace(".mtx","_features.tsv")
            import dv_common as DC
            XA,XB,cells,fA,fB=DC.load_align(e["rna"],e["atac"],e["ori1"],e["ori2"],log=lambda *x:None)
            labs=L.coassay_labels(_map_key(ds,_FMAP),cells)
            # --- disjoint reference/query split (co-assay only) -------------
            # reference = half A, query = half B, in BOTH directions. B's labels
            # are masked so no query cell's twin can be trained on.
            _cn=[str(c) for c in cells]
            _rs=np.random.RandomState(0); _perm=_rs.permutation(len(_cn))
            _A=np.zeros(len(_cn),bool); _A[_perm[:len(_cn)//2]]=True
            pd.DataFrame({"cell_id":_cn,"is_reference":_A.astype(int)}).to_csv(
                os.path.join(out,"split.csv"),index=False)
            labs=np.asarray(labs,object).copy()
            # No sentinel: lab1 holds the reference half's rows, lab2 the query
            # half's, and each omits the other half entirely. A row that is
            # simply absent is what every wrapper's own filter already handles;
            # a literal "__NA__" was being learned as a cell type by six of the
            # seven and predicted back out.
            lmA={str(c):l for c,l,a in zip(cells,labs,_A) if a}
            lmB={str(c):l for c,l,a in zip(cells,labs,_A) if not a}
            print("split: %d reference / %d query (co-assay, seed 0); "
                  "lab1=reference rows only, lab2=query rows only"
                  %(int(_A.sum()),int((~_A).sum())),flush=True)
            bc1=pd.read_csv(b1,header=None)[0].astype(str).tolist()
            lab1p=os.path.join(out,"lab1.csv")
            k1=[c for c in bc1 if c in lmA]
            write_labels(k1,[lmA[c] for c in k1],lab1p)
            bc2=pd.read_csv(b2,header=None)[0].astype(str).tolist(); lab2p=os.path.join(out,"lab2.csv")
            k2=[c for c in bc2 if c in lmB]
            write_labels(k2,[lmB[c] for c in k2],lab2p)
            if not k1 or not k2:
                raise SystemExit("empty reference or query label file for %s"%ds)
            f1h=_harm(f1,os.path.join(out,"f1h.tsv")); f2h=_harm(f2,os.path.join(out,"f2h.tsv"),adt=True)
            args=["--dataset_id",ds,"--x1",x1,"--barcodes1",b1,"--features1",f1h,"--x2",x2,"--barcodes2",b2,"--features2",f2h,
                  "--labels1",lab1p,"--labels2",lab2p,"--out_dir",out,"--label_id_col","cell_id","--label_col","cell_type"]
        cmd=(["Rscript",f"{PROJ}/4_label_transfer/archive/{script}"] if lang=="R" else [os.environ.get("PY", sys.executable),f"{PROJ}/4_label_transfer/archive/{script}"])+args+extra
        r=subprocess.run(cmd,env=env,cwd=PROJ,capture_output=True,text=True,timeout=20000)
        # keep the tool's own output: the fallback notices live in here, and
        # discarding it is why two silent substitutions went unnoticed
        _log=(r.stdout or "")+"\n----- stderr -----\n"+(r.stderr or "")
        with open(os.path.join(out,"tool_stdout.log"),"w") as _fh: _fh.write(_log)
        _MARK=[("anchor/TransferData failed","seurat_knn_pca_fallback"),
               ("buildReference/irlba failed","symphony_base_pca_fallback"),
               ("dropout selection gave","scmap_variance_feature_fallback"),
               ("knn_classifier failed","concerto_knn_embedding_fallback")]
        _used=[nm for pat,nm in _MARK if pat in _log] or ["primary"]
        with open(os.path.join(out,"path_used.txt"),"w") as _fh:
            _fh.write("\n".join(_used)+"\n")
        if _used!=["primary"]:
            print("PATH_USED %s %s: %s"%(tool,ds,",".join(_used)),flush=True)
        if not os.path.exists(pf):
            res["status"]="no_pred"; res["err"]=(r.stderr or r.stdout)[-300:]; pd.DataFrame([res]).to_csv(f"{PROJ}/runs_label_transfer/EXT_{tool}_{ds}.csv",index=False); print(res); return
    if not os.path.exists(pf):
        res["status"]="no_pred_refuse"; pd.DataFrame([res]).to_csv(f"{PROJ}/runs_label_transfer/EXT_{tool}_{ds}.csv",index=False); print(res); return
    df=pd.read_csv(pf); df=df[df["true_label"].notna()].copy()
    df["true_label"]=df["true_label"].astype(str); df["predicted_label"]=df["predicted_label"].astype(str)
    if "prediction_score" not in df.columns: df["prediction_score"]=np.nan
    accs=[(g["true_label"]==g["predicted_label"]).mean() for _,g in df.groupby("direction")]
    mf1=[f1_score(g["true_label"],g["predicted_label"],average="macro",zero_division=0) for _,g in df.groupby("direction")]
    res["status"]="ok"; res["overall_nogamma"]=float(np.mean(accs)); res["macrof1_nogamma"]=float(np.mean(mf1))
    # ---- gamma-guided: blend soft external base with v12 coupling-vote, lambda swept incl 0 (=base) ----
    try:
        cls=sorted(set(df["true_label"]) | set(df["predicted_label"])); ci={c:i for i,c in enumerate(cls)}; ca=np.array(cls,dtype=object); K=len(cls)
        if kind!="coassay":
            t=ds.split("_",1)[1]; X1,f1b,bc1t,y1,X2,f2b,bc2t,y2=L.tm_load(t, ds); mats,_=C.get_gammas(["v12"],ds,""); G=np.asarray(mats["v12"],float)
            kp1=y1!="__NA__"; S12,v12c=L2.vote(G[np.where(kp1)[0],:],y1[kp1],True)
            kp2=y2!="__NA__"; S21,v21c=L2.vote(G.T[np.where(kp2)[0],:],y2[kp2],True)
            Sv={"mod1_to_mod2": GF.sharpen(GF.align(S12,v12c,cls)), "mod2_to_mod1": GF.sharpen(GF.align(S21,v21c,cls))}
            qids={"mod1_to_mod2": list(map(str,bc2t)), "mod2_to_mod1": list(map(str,bc1t))}
        else:
            G,gc,_=L.load_saved_coupling("v12",ds); G=np.asarray(G,float)
            import dv_common as DC; e=_FMAP[_map_key(ds,_FMAP)]
            XA,XB,cells,fA,fB=DC.load_align(e["rna"],e["atac"],e["ori1"],e["ori2"],log=lambda *x:None)
            labs=L.coassay_labels(_map_key(ds,_FMAP),cells); kp=labs!="__NA__"; kk=np.where(kp)[0]
            pos={str(c):i for i,c in enumerate(gc)}; idx=np.array([pos[str(c)] for c in cells if str(c) in pos])
            Gk=G[np.ix_(idx,idx)][np.ix_(kk,kk)]
            Sb,vb=L2.vote(Gk,labs[kp],True); Sa,va=L2.vote(Gk.T,labs[kp],True)
            Sv={"mod1_to_mod2":GF.sharpen(GF.align(Sb,vb,cls)),"mod2_to_mod1":GF.sharpen(GF.align(Sa,va,cls))}
            cq=[str(c) for c in np.array(cells)[kk]]; qids={"mod1_to_mod2":cq,"mod2_to_mod1":cq}
        # precompute per-direction soft base + aligned vote
        per=[]
        for d,g in df.groupby("direction"):
            g=g.reset_index(drop=True); tl=g["true_label"].values
            if d not in Sv:
                per.append((tl,None,None)); continue
            qid=qids[d]; qmap={c:i for i,c in enumerate(qid)}
            pidx=np.array([ci.get(pl,-1) for pl in g["predicted_label"]]); sc=g["prediction_score"].to_numpy(dtype=float)
            P_base=GF.soft_base(pidx,sc,K); vote=np.zeros((len(g),K))
            for r2 in range(len(g)):
                cid=str(g.loc[r2,"target_cell_id"])
                if cid in qmap: vote[r2]=Sv[d][qmap[cid]]
            per.append((tl,P_base,vote))
        def score_at(lam):
            a=[];f=[]
            for tl,P_base,vote in per:
                if P_base is None:
                    a.append(np.nan); continue
                pred=ca[GF.blend(P_base,vote,lam).argmax(1)]
                a.append((tl==pred).mean()); f.append(f1_score(tl,pred,average="macro",zero_division=0))
            return float(np.nanmean(a)), (float(np.nanmean(f)) if f else float("nan"))
        best=-1; lstar=0.0; bf=float("nan")
        for lam in GF.LAM_GRID:
            o,fm=score_at(lam)
            if o>best: best=o; lstar=lam; bf=fm
        res["overall_gamma"]=best; res["lam_star"]=lstar; res["macrof1_gamma"]=bf
        res["overall_gamma_fix0p5"]=score_at(0.5)[0]
    except Exception as ex:
        res["gamma_err"]=str(ex)[:160]
    pd.DataFrame([res]).to_csv(f"{PROJ}/runs_label_transfer/EXT_{tool}_{ds}.csv",index=False); print(res)
if __name__=="__main__": main()
