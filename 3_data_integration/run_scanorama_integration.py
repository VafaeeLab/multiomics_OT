#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared_dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--dataset_id', required=True)
    ap.add_argument('--dimred', type=int, default=30)
    ap.add_argument('--knn', type=int, default=20)
    ap.add_argument('--sigma', type=float, default=15.0)
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--batch_size', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    try:
        import scanorama
    except Exception as e:
        raise SystemExit(f'scanorama is required: {e}')
    p=Path(args.prepared_dir)
    X1=sp.load_npz(p/'mod1_log.npz').tocsr(); X2=sp.load_npz(p/'mod2_log.npz').tocsr()
    # Scanorama matches cells ACROSS the two blocks; standardise each block per
    # feature first so the log1p (non-negative) and CLR (zero-centred) outputs
    # share a location and scale. Within-modality structure is unchanged.
    def _permod(M):
        A=np.asarray(M.todense(), dtype=np.float64)
        sd=A.std(0); sd[sd<1e-9]=1.0
        A=(A-A.mean(0))/sd
        return np.nan_to_num(A)
    X1=_permod(X1); X2=_permod(X2)
    f=pd.read_csv(p/'mod1_log_features.tsv',sep='\t',header=None).iloc[:,0].astype(str).tolist()
    obs1=pd.read_csv(p/'mod1_log_obs.csv'); obs2=pd.read_csv(p/'mod2_log_obs.csv')
    datasets=[X1,X2]; genes=[f,f]
    try:
        integrated=scanorama.integrate(datasets, genes, dimred=args.dimred, knn=args.knn, sigma=args.sigma, alpha=args.alpha, batch_size=args.batch_size, seed=args.seed)
        if isinstance(integrated, tuple): integrated=integrated[0]
    except TypeError:
        integrated=scanorama.integrate(datasets, genes, dimred=args.dimred, knn=args.knn, sigma=args.sigma, alpha=args.alpha, seed=args.seed)
        if isinstance(integrated, tuple): integrated=integrated[0]
    Z1=np.asarray(integrated[0]); Z2=np.asarray(integrated[1])
    k=min(args.dimred, Z1.shape[1], Z2.shape[1])
    Z=np.vstack([Z1[:,:k],Z2[:,:k]])
    obs=pd.concat([obs1,obs2],ignore_index=True)
    out=obs[['cell_id','modality']].copy()
    for i in range(k): out[f'emb_{i}']=Z[:,i]
    out.to_csv(outdir/'embeddings.csv',index=False)
    (outdir/'run_settings.json').write_text(json.dumps(vars(args),indent=2))

if __name__=='__main__': main()
