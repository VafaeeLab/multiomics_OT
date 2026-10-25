#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import sparse
from scipy.stats import chisquare
from scipy.sparse.csgraph import connected_components
from integration_io import load_embedding_csv, merge_labels


def safe_float(x):
    try:
        if x is None or not np.isfinite(x): return np.nan
        return float(x)
    except Exception: return np.nan


def kmeans_clusters(X, n_clusters, seed):
    n_clusters=max(2, min(int(n_clusters), X.shape[0]-1))
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X).astype(str)


def asw_label(X, labels):
    labels=np.asarray(labels).astype(str)
    if len(np.unique(labels))<2 or len(labels)<3: return np.nan
    try:
        s=silhouette_samples(X, labels, metric='euclidean')
        return float((np.mean(s)+1)/2)
    except Exception: return np.nan


def batch_asw(X, batch, cell_type):
    vals=[]
    for ct in np.unique(cell_type):
        idx=np.where(cell_type==ct)[0]
        if len(idx)<3 or len(np.unique(batch[idx]))<2: continue
        try:
            s=silhouette_samples(X[idx], batch[idx], metric='euclidean')
            vals.append(1 - np.mean(np.abs(s)))
        except Exception: pass
    return float(np.mean(vals)) if vals else np.nan


def lisi_scores(X, labels, k=30, conserve=False):
    labels=np.asarray(labels).astype(str); cats=np.unique(labels); ncat=len(cats)
    if ncat < 2 or X.shape[0] < 3: return np.nan
    k=min(k, X.shape[0]-1)
    nbr=NearestNeighbors(n_neighbors=k+1).fit(X)
    inds=nbr.kneighbors(return_distance=False)[:,1:]
    eff=[]
    for row in inds:
        vals,counts=np.unique(labels[row],return_counts=True)
        p=counts/counts.sum(); eff.append(1/np.sum(p*p))
    eff=np.asarray(eff)
    if conserve:
        return float(1 - np.mean((eff-1)/max(1,ncat-1)))
    return float(np.mean((eff-1)/max(1,ncat-1)))


def kbet_score(X, batch, k=30, alpha=0.05, max_cells=5000, seed=42):
    rng=np.random.default_rng(seed); n=X.shape[0]
    if len(np.unique(batch))<2 or n<3: return np.nan
    idx=np.arange(n)
    if n>max_cells: idx=rng.choice(idx, size=max_cells, replace=False)
    Xs=X[idx]; batch=np.asarray(batch).astype(str); cats,global_counts=np.unique(batch,return_counts=True); expected=global_counts/global_counts.sum()
    k=min(k,n-1)
    nbr=NearestNeighbors(n_neighbors=k+1).fit(X)
    neigh=nbr.kneighbors(Xs, return_distance=False)[:,1:]
    reject=0; valid=0
    for row in neigh:
        obs=np.array([(batch[row]==c).sum() for c in cats],dtype=float)
        exp=expected*obs.sum()
        if np.any(exp<1e-9): continue
        try:
            stat,p=chisquare(obs, exp)
            reject += (p < alpha); valid += 1
        except Exception: pass
    return float(1 - reject/max(1,valid)) if valid else np.nan


def graph_connectivity(X, labels, k=15):
    labels=np.asarray(labels).astype(str)
    if X.shape[0]<3: return np.nan
    k=min(k,X.shape[0]-1)
    inds=NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(return_distance=False)[:,1:]
    rows=np.repeat(np.arange(X.shape[0]), k); cols=inds.ravel()
    G=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(X.shape[0],X.shape[0])); G=G.maximum(G.T)
    vals=[]
    for lab in np.unique(labels):
        idx=np.where(labels==lab)[0]
        if len(idx)<=1: continue
        sub=G[idx][:,idx]
        ncomp, comp=connected_components(sub, directed=False)
        if ncomp==0: continue
        vals.append(np.bincount(comp).max()/len(idx))
    return float(np.mean(vals)) if vals else np.nan


def isolated_label_f1(true_labels, clusters, batch):
    true_labels=np.asarray(true_labels).astype(str); clusters=np.asarray(clusters).astype(str); batch=np.asarray(batch).astype(str)
    labels=np.unique(true_labels)
    # scIB-inspired: labels present in the fewest batches are isolated; if tie includes all, use rarest quartile.
    n_batches={lab:len(np.unique(batch[true_labels==lab])) for lab in labels}
    min_b=min(n_batches.values()) if n_batches else 0
    isolated=[lab for lab,b in n_batches.items() if b==min_b]
    if len(isolated)==len(labels) and len(labels)>4:
        counts=pd.Series(true_labels).value_counts()
        q=np.quantile(counts.values,0.25)
        isolated=[lab for lab in labels if counts.get(lab,0)<=q]
    f1s=[]
    for lab in isolated:
        y=(true_labels==lab).astype(int)
        # best matching cluster for this label
        best=0.0
        for cl in np.unique(clusters):
            pred=(clusters==cl).astype(int)
            best=max(best, f1_score(y,pred,zero_division=0))
        f1s.append(best)
    return float(np.mean(f1s)) if f1s else np.nan


def pcr_batch_score(X, batch):
    batch=np.asarray(batch).astype(str)
    if len(np.unique(batch))<2: return np.nan, np.nan
    enc=OneHotEncoder(drop='first', sparse_output=False)
    B=enc.fit_transform(batch.reshape(-1,1))
    r2=[]
    for j in range(X.shape[1]):
        y=X[:,j]
        try: r2.append(LinearRegression().fit(B,y).score(B,y))
        except Exception: pass
    r2=float(np.mean(r2)) if r2 else np.nan
    return 1-r2 if np.isfinite(r2) else np.nan, r2


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--dataset_id', required=True); ap.add_argument('--method', required=True)
    ap.add_argument('--labels1', default=''); ap.add_argument('--labels2', default='')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--neighbors', type=int, default=30)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df,X=load_embedding_csv(args.embeddings)
    df=merge_labels(df,args.labels1,args.labels2)
    emb_cols=[c for c in df.columns if c.startswith('emb_') or c.startswith('PC_')]
    if emb_cols: X=df[emb_cols].to_numpy(float)
    X=np.nan_to_num(X.astype(float))
    if X.shape[1] > 50: X=X[:,:50]
    X=StandardScaler().fit_transform(X)
    labels=df['cell_type'].astype(str).to_numpy(); batch=df['batch_modality'].astype(str).to_numpy(); modality=df['modality'].astype(str).to_numpy()
    valid=labels!='unknown'
    n_labels=len(np.unique(labels[valid])) if valid.any() else 2
    clusters=kmeans_clusters(X, n_labels, args.seed)
    metrics={}
    metrics['n_cells']=int(X.shape[0]); metrics['n_embedding_dims']=int(X.shape[1]); metrics['n_cell_types']=int(n_labels); metrics['n_batches']=int(len(np.unique(batch)))
    metrics['nmi']=normalized_mutual_info_score(labels[valid], clusters[valid]) if valid.any() else np.nan
    metrics['ami']=adjusted_mutual_info_score(labels[valid], clusters[valid]) if valid.any() else np.nan
    metrics['ari']=adjusted_rand_score(labels[valid], clusters[valid]) if valid.any() else np.nan
    metrics['cell_type_asw']=asw_label(X[valid], labels[valid]) if valid.any() else np.nan
    metrics['isolated_label_f1']=isolated_label_f1(labels[valid], clusters[valid], batch[valid]) if valid.any() else np.nan
    metrics['batch_asw']=batch_asw(X, batch, labels)
    metrics['modality_asw']=batch_asw(X, modality, labels)
    metrics['ilisi']=lisi_scores(X, batch, k=args.neighbors, conserve=False)
    metrics['clisi']=lisi_scores(X, labels, k=args.neighbors, conserve=True) if valid.any() else np.nan
    metrics['kbet']=kbet_score(X, batch, k=args.neighbors, seed=args.seed)
    metrics['graph_connectivity']=graph_connectivity(X, labels, k=min(15,args.neighbors)) if valid.any() else np.nan
    try: metrics['unsupervised_silhouette']=float(silhouette_score(X, clusters)) if len(np.unique(clusters))>1 else np.nan
    except Exception: metrics['unsupervised_silhouette']=np.nan
    try: metrics['davies_bouldin']=float(davies_bouldin_score(X, clusters)) if len(np.unique(clusters))>1 else np.nan
    except Exception: metrics['davies_bouldin']=np.nan
    try: metrics['calinski_harabasz']=float(calinski_harabasz_score(X, clusters)) if len(np.unique(clusters))>1 else np.nan
    except Exception: metrics['calinski_harabasz']=np.nan
    metrics['pcr_batch_score'], metrics['pcr_batch_r2']=pcr_batch_score(X,batch)
    bio_keys=['nmi','ami','ari','cell_type_asw','isolated_label_f1','clisi','graph_connectivity']
    batch_keys=['batch_asw','modality_asw','ilisi','kbet','pcr_batch_score']
    def mean_keys(keys):
        vals=[metrics[k] for k in keys if k in metrics and np.isfinite(metrics[k])]
        return float(np.mean(vals)) if vals else np.nan
    metrics['bio_conservation_score']=mean_keys(bio_keys)
    metrics['batch_mixing_score']=mean_keys(batch_keys)
    if np.isfinite(metrics['bio_conservation_score']) and np.isfinite(metrics['batch_mixing_score']):
        metrics['overall_integration_score']=0.6*metrics['bio_conservation_score']+0.4*metrics['batch_mixing_score']
    else: metrics['overall_integration_score']=np.nan
    metrics.update({'dataset_id':args.dataset_id,'method':args.method})
    # outputs
    df2=df.copy(); df2['cluster']=clusters; df2['dataset_id']=args.dataset_id; df2['method']=args.method
    df2.to_csv(outdir/'cells_with_labels.csv',index=False)
    df2[['dataset_id','method','cell_id','modality','cell_type','batch','batch_modality','cluster']].to_csv(outdir/'clusters.csv',index=False)
    pd.DataFrame([metrics]).to_csv(outdir/'integration_metrics.csv',index=False)
    pd.DataFrame([metrics]).to_csv(outdir/'metrics_headline.csv',index=False)
    (outdir/'integration_metrics.json').write_text(json.dumps({k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()},indent=2))
    # per-class cluster F1
    rows=[]
    for lab in np.unique(labels[valid]):
        y=(labels==lab).astype(int); best=0; bestcl=''
        for cl in np.unique(clusters):
            score=f1_score(y,(clusters==cl).astype(int),zero_division=0)
            if score>best: best=score; bestcl=cl
        rows.append({'dataset_id':args.dataset_id,'method':args.method,'cell_type':lab,'best_cluster':bestcl,'f1':best,'n_cells':int((labels==lab).sum())})
    pd.DataFrame(rows).to_csv(outdir/'per_class_metrics.csv',index=False)

if __name__=='__main__': main()
