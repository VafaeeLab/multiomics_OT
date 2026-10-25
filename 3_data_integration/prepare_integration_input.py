#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from integration_io import load_matrix, read_names, load_labels, align_features, write_mtx_bundle, normalize_total_log1p, log


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_id', required=True)
    ap.add_argument('--x1', required=True); ap.add_argument('--x2', required=True)
    ap.add_argument('--barcodes1', required=True); ap.add_argument('--barcodes2', required=True)
    ap.add_argument('--features1', default=''); ap.add_argument('--features2', default='')
    ap.add_argument('--labels1', default=''); ap.add_argument('--labels2', default='')
    ap.add_argument('--orientation1', default='cell_by_feature'); ap.add_argument('--orientation2', default='cell_by_feature')
    ap.add_argument('--feature_mode', default='auto', choices=['auto','shared','union'])
    ap.add_argument('--outdir', required=True)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    X1=load_matrix(args.x1,args.orientation1); X2=load_matrix(args.x2,args.orientation2)
    ids1=read_names(args.barcodes1, X1.shape[0], 'mod1_cell')
    ids2=read_names(args.barcodes2, X2.shape[0], 'mod2_cell')
    feats1=read_names(args.features1, X1.shape[1], 'mod1_feature')
    feats2=read_names(args.features2, X2.shape[1], 'mod2_feature')
    X1a,X2a,features,feature_mode_used=align_features(X1,feats1,X2,feats2,args.feature_mode)
    lab1=load_labels(args.labels1,ids1,'mod1'); lab2=load_labels(args.labels2,ids2,'mod2')
    lab1['modality']='mod1'; lab2['modality']='mod2'
    obs=pd.concat([lab1,lab2],ignore_index=True)
    obs['dataset_id']=args.dataset_id
    obs['batch_modality']=obs['batch'].astype(str)+'__'+obs['modality'].astype(str)
    import scipy.sparse as sp
    X=sp.vstack([X1a,X2a],format='csr')
    write_mtx_bundle(X,obs,features,outdir,'combined_raw')
    write_mtx_bundle(normalize_total_log1p(X),obs,features,outdir,'combined_log')
    write_mtx_bundle(X1a,lab1,features,outdir,'mod1_raw')
    write_mtx_bundle(X2a,lab2,features,outdir,'mod2_raw')
    write_mtx_bundle(normalize_total_log1p(X1a),lab1,features,outdir,'mod1_log')
    write_mtx_bundle(normalize_total_log1p(X2a),lab2,features,outdir,'mod2_log')
    meta={
        'dataset_id':args.dataset_id,'n_mod1':int(X1.shape[0]),'n_mod2':int(X2.shape[0]),
        'n_features_mod1_original':int(X1.shape[1]),'n_features_mod2_original':int(X2.shape[1]),
        'n_features_aligned':int(X.shape[1]),'feature_mode_requested':args.feature_mode,'feature_mode_used':feature_mode_used,
        'x1':args.x1,'x2':args.x2,'labels1':args.labels1,'labels2':args.labels2
    }
    (outdir/'input_metadata.json').write_text(json.dumps(meta,indent=2))
    log(json.dumps(meta, indent=2))

if __name__=='__main__': main()
