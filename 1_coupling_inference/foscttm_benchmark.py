#!/usr/bin/env python3
"""foscttm_benchmark.py — unified FOSCTTM benchmark for SEVEN models.

  custom (solid lines):  v1, v2, v9, v12      external (dashed lines):  unioncom, mmdma, scot

Stage 1  ensure_results : generate a model's FOSCTTM result ONLY if it is missing
                          (custom -> run_benchmark_model.py; external -> copy from runs_foscttm/).
                          Already-completed models are skipped (idempotent).
Stage 2  make_plots     : symmetric benchmark ->  foscttm_summary.csv
                                                  multipanel_foscttm.png / .pdf
                                                  ranking_barplot.png
                          plus per-dataset sorted line plots, split by orientation, into:
                            plots/row_foscttm/<ds>.png
                            plots/column_foscttm/<ds>.png
                            plots/symmetric_foscttm/<ds>.png

Per-cell row/col/sym FOSCTTM is taken from the model's saved metrics when present
(external), else computed from its score matrix (gamma.npy as similarity, or
pairwise_matrix_common.npy as distance) with a chunked, memory-safe pass, and cached
to <model>/<ds>/_rcs_cache.npz so re-plotting is instant.

Usage:
  python foscttm_benchmark.py                # ensure results (skip done) + all plots
  python foscttm_benchmark.py --plots_only   # only (re)build plots from existing results
  python foscttm_benchmark.py --force        # re-run model results even if present
  python foscttm_benchmark.py --recompute_rcs# ignore the cached row/col/sym arrays
"""
import os, sys, json, shutil, subprocess, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

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
CS=f"{ROOT}/1_coupling_inference"; PY=os.environ.get("PY", sys.executable)
BM=f"{OT_ROOT}/foscttm_benchmark"; EXT_SRC=f"{ROOT}/runs_foscttm"
# (label, dir, colour, is_external)
MODELS=[("v1","v1","#4363D8",False),("v2","v2","#3CB44B",False),
        ("v9","v9","#F58231",False),("v12","v12","#E6194B",False),
        ("unioncom","unioncom","#911EB4",True),("mmdma","mmdma","#42D4F4",True),("scot","scot","#808000",True)]
DS=sorted(json.load(open(f"{CS}/foscttm_input_map.json")))
LW=1.6
METRICS=[("row","FOSCTTM row-wise","row_foscttm"),
         ("col","FOSCTTM column-wise","column_foscttm"),
         ("sym","FOSCTTM symmetric","symmetric_foscttm")]

# ----------------------------------------------------------------------- results
def has_result(rd,ds):
    base=f"{BM}/{rd}/{ds}"
    return any(os.path.exists(f"{base}/{f}") for f in
               ("per_cell_foscttm.csv","per_cell_nfoscttm_metrics.csv"))

def ensure_results(force):
    for nm,rd,_,ext in MODELS:
        for ds in DS:
            base=f"{BM}/{rd}/{ds}"
            if has_result(rd,ds) and not force:
                continue
            if ext:                                   # external: copy completed run
                src=f"{EXT_SRC}/{rd}/{ds}"
                if os.path.isdir(src):
                    os.makedirs(os.path.dirname(base),exist_ok=True)
                    if os.path.isdir(base): shutil.rmtree(base)
                    shutil.copytree(src,base); print("COPIED",nm,ds,flush=True)
                else: print("MISSING external source",nm,ds,flush=True)
            else:                                     # custom: regenerate
                print("RUN",nm,ds,flush=True)
                subprocess.run([PY,f"{CS}/run_benchmark_model.py",rd,ds])

# ------------------------------------------------------------------- row/col/sym
def fos_chunked(M,higher_better,chunk=2048):
    n=M.shape[0]; dg=np.asarray(np.diag(M),np.float64); fr=np.zeros(n); fc=np.zeros(n)
    for s in range(0,n,chunk):
        e=min(s+chunk,n); B=np.asarray(M[s:e],np.float64); di=dg[s:e][:,None]
        better=(B>di) if higher_better else (B<di)
        fr[s:e]=(better.sum(1)+0.5*((B==di).sum(1)-1))/max(1,n-1)
    for s in range(0,n,chunk):
        e=min(s+chunk,n); B=np.asarray(M[:,s:e],np.float64); dj=dg[s:e][None,:]
        better=(B>dj) if higher_better else (B<dj)
        fc[s:e]=(better.sum(0)+0.5*((B==dj).sum(0)-1))/max(1,n-1)
    return fr,fc,0.5*(fr+fc)

def get_rcs(rd,ds,recompute):
    base=f"{BM}/{rd}/{ds}"; cache=f"{base}/_rcs_cache.npz"
    if os.path.exists(cache) and not recompute:
        z=np.load(cache); return z["row"],z["col"],z["sym"]
    row=col=sym=None
    pc=f"{base}/per_cell_nfoscttm_metrics.csv"
    if os.path.exists(pc):                              # external: saved row/col/sym
        df=pd.read_csv(pc)
        if {"foscttm_row","foscttm_col","foscttm_symmetric"}<=set(df.columns):
            row,col,sym=(df[c].to_numpy(float) for c in ("foscttm_row","foscttm_col","foscttm_symmetric"))
    if row is None and os.path.exists(f"{base}/gamma.npy"):        # custom v1/v2/v12
        M=np.load(f"{base}/gamma.npy"); row,col,sym=fos_chunked(M,True); del M
    if row is None and os.path.exists(f"{base}/pairwise_matrix_common.npy"):  # v9 / fallback
        M=np.load(f"{base}/pairwise_matrix_common.npy"); row,col,sym=fos_chunked(M,False); del M
    if row is None:                                    # last resort: sym only
        for fn in ("per_cell_foscttm.csv","per_cell_nfoscttm_metrics.csv"):
            p=f"{base}/{fn}"
            if os.path.exists(p):
                df=pd.read_csv(p)
                for c in ("foscttm_sym","foscttm","foscttm_symmetric"):
                    if c in df.columns: sym=pd.to_numeric(df[c],errors="coerce").to_numpy(); break
            if sym is not None: break
        if sym is None: return None
        row=col=sym
    np.savez_compressed(cache,row=row,col=col,sym=sym)
    return row,col,sym

# ------------------------------------------------------------------------- plots
def make_plots(recompute):
    for _,_,sub in METRICS: os.makedirs(f"{BM}/plots/{sub}",exist_ok=True)
    data={}                                            # data[ds][nm] = (row,col,sym)
    for ds in DS:
        data[ds]={}
        for nm,rd,_,_ in MODELS:
            try:
                r=get_rcs(rd,ds,recompute)
                if r is not None: data[ds][nm]=r
            except Exception as e: print("skip",nm,ds,repr(e)[:90],flush=True)
        print("rcs ready",ds,flush=True)
    colmap={m[0]:m[2] for m in MODELS}; extmap={m[0]:m[3] for m in MODELS}
    order=[m[0] for m in MODELS]

    # --- per-dataset sorted line plots, one folder per orientation ---
    mi={"row":0,"col":1,"sym":2}
    for key,title,sub in METRICS:
        for ds in DS:
            if not data[ds]: continue
            fig,a=plt.subplots(figsize=(7,5))
            for nm in order:
                if nm not in data[ds]: continue
                v=np.sort(np.asarray(data[ds][nm][mi[key]],float)); v=v[np.isfinite(v)]
                a.plot(np.arange(len(v)),v,color=colmap[nm],lw=LW,ls="--" if extmap[nm] else "-",
                       label=f"{nm} ({v.mean():.3f})")
            a.set_title(f"{title} — {ds}",fontsize=10); a.set_xlabel("number of cells (sorted)")
            a.set_ylabel(f"{title} (lower=better)"); a.set_ylim(0,1); a.grid(alpha=0.25)
            a.legend(fontsize=7); fig.tight_layout()
            fig.savefig(f"{BM}/plots/{sub}/{ds}.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # --- symmetric benchmark: multipanel + summary + ranking ---
    nc=4; nr=int(np.ceil(len(DS)/nc)); fig,ax=plt.subplots(nr,nc,figsize=(4.2*nc,3.1*nr)); ax=ax.ravel()
    summ={nm:{} for nm in order}
    for i,ds in enumerate(DS):
        a=ax[i]
        for nm in order:
            if nm not in data[ds]: continue
            v=np.sort(np.asarray(data[ds][nm][2],float)); v=v[np.isfinite(v)]
            summ[nm][ds]=float(v.mean())
            a.plot(np.arange(len(v)),v,label=nm,color=colmap[nm],lw=LW,ls="--" if extmap[nm] else "-",alpha=0.95)
        a.set_title(ds,fontsize=8); a.set_ylim(0,1); a.grid(alpha=0.25); a.tick_params(labelsize=6)
        if i==0: a.legend(fontsize=6,ncol=2)
    for j in range(len(DS),len(ax)): ax[j].axis("off")
    fig.suptitle("FOSCTTM benchmark — custom (solid: v1,v2,v9,v12) vs external (dashed: unioncom,mmdma,scot); sorted per-cell, lower=better",fontsize=11,y=1.002)
    fig.tight_layout(); fig.savefig(f"{BM}/multipanel_foscttm.png",dpi=160,bbox_inches="tight")
    fig.savefig(f"{BM}/multipanel_foscttm.pdf",bbox_inches="tight"); plt.close(fig)

    S=pd.DataFrame({nm:summ[nm] for nm in order}).reindex(DS); S.loc["MEAN"]=S.mean()
    S.round(4).to_csv(f"{BM}/foscttm_summary.csv")

    means=S.loc["MEAN"][order]; figb,ab=plt.subplots(figsize=(8,4.5))
    bars=ab.bar(order,means.values,color=[colmap[k] for k in order],edgecolor="k",lw=0.5)
    for k,b in zip(order,bars):
        if extmap[k]: b.set_hatch("//"); b.set_linestyle("--")
        ab.text(b.get_x()+b.get_width()/2,b.get_height()+0.004,"%.3f"%means[k],ha="center",fontsize=8)
    ab.set_ylabel("mean FOSCTTM (lower = better)")
    ab.set_title(f"FOSCTTM ranking over {len(DS)} datasets  (solid bars = custom, hatched = external)")
    ab.grid(axis="y",alpha=0.25); figb.tight_layout()
    figb.savefig(f"{BM}/ranking_barplot.png",dpi=160,bbox_inches="tight"); plt.close(figb)
    print("\nBENCHMARK ->",BM); print(S.round(4).to_string())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--plots_only",action="store_true")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--recompute_rcs",action="store_true")
    a=ap.parse_args()
    os.makedirs(BM,exist_ok=True)
    if not a.plots_only: ensure_results(a.force)
    make_plots(a.recompute_rcs)
    print("foscttm_benchmark DONE")
