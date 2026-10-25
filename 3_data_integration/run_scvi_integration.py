#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared_dir', required=True); ap.add_argument('--outdir', required=True); ap.add_argument('--dataset_id', required=True)
    ap.add_argument('--n_latent', type=int, default=30); ap.add_argument('--n_hidden', type=int, default=128); ap.add_argument('--n_layers', type=int, default=2); ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--gene_likelihood', default='zinb', choices=['zinb','nb','poisson'])
    ap.add_argument('--max_epochs', type=int, default=100); ap.add_argument('--batch_size', type=int, default=512); ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    try:
        import anndata as ad
        import scvi
    except Exception as e:
        raise SystemExit(f'scvi-tools/anndata are required: {e}')
    scvi.settings.seed=args.seed
    p=Path(args.prepared_dir)
    X=sp.load_npz(p/'combined_raw.npz').tocsr()
    obs=pd.read_csv(p/'combined_raw_obs.csv')
    var=pd.read_csv(p/'combined_raw_features.tsv',sep='\t',header=None)
    adata=ad.AnnData(X=X, obs=obs.copy())
    adata.obs_names=obs['modality'].astype(str)+'__'+obs['cell_id'].astype(str)
    adata.var_names=var.iloc[:,0].astype(str).values
    # use batch+modality for conservative integration; this prevents donor/modality from leaking as unwanted batch structure
    batch_key='batch_modality' if 'batch_modality' in adata.obs.columns else 'modality'
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    # scVI's NB/ZINB likelihood assumes raw integer counts. The methylome bundles
    # carry a continuous gene-activity score in [0,1]; fitting counts to that is
    # misspecified and collapsed cross-modality mixing to zero. Detect it and use
    # the Gaussian likelihood scVI provides for continuous data.
    _Xd = adata.X
    _m2 = (adata.obs["modality"].astype(str) == "mod2").to_numpy()
    _blk = _Xd[_m2] if _m2.any() else _Xd
    _vals = _blk.data if sp.issparse(_blk) else np.asarray(_blk).ravel()
    if _vals.size > 200000:
        _vals = _vals[np.random.RandomState(0).choice(_vals.size, 200000, replace=False)]
    _sample = _vals
    _is_count = bool(np.allclose(_sample, np.round(_sample))) and float(_sample.max()) > 1.5
    if not _is_count:
        print(f"[scvi] input is not counts (max={float(_sample.max()):.3f}, "
              f"integer={np.allclose(_sample, np.round(_sample))}) -> gene_likelihood='normal'",
              flush=True)
        args.gene_likelihood = "normal"
        # A Gaussian likelihood belongs with log-normalised input, not raw
        # counts -- otherwise mod1's integer counts are the misspecified block.
        _logp = p / "combined_log.npz"
        if _logp.exists():
            _Xl = sp.load_npz(_logp).tocsr()
            if _Xl.shape == adata.shape:
                adata = ad.AnnData(X=_Xl, obs=obs.copy())
                adata.obs_names = obs["modality"].astype(str) + "__" + obs["cell_id"].astype(str)
                adata.var_names = var.iloc[:, 0].astype(str).values
                scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
                print("[scvi] switched input to combined_log to match the Gaussian likelihood",
                      flush=True)
            else:
                print(f"[scvi] WARNING combined_log shape {_Xl.shape} != {adata.shape}; "
                      "keeping combined_raw", flush=True)
        else:
            print("[scvi] WARNING no combined_log.npz; Gaussian on raw counts", flush=True)

    model=scvi.model.SCVI(adata, n_latent=args.n_latent, n_hidden=args.n_hidden, n_layers=args.n_layers, dropout_rate=args.dropout, gene_likelihood=args.gene_likelihood)
    _es = os.environ.get("SCVI_EARLY_STOP", "1") == "1"
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size, early_stopping=_es, check_val_every_n_epoch=max(1,min(10,args.max_epochs//5)))
    Z=model.get_latent_representation()
    out=obs[['cell_id','modality']].copy()
    k=min(args.n_latent,Z.shape[1])
    for i in range(k): out[f'emb_{i}']=Z[:,i]
    out.to_csv(outdir/'embeddings.csv',index=False)
    model.save(str(outdir/'scvi_model'), overwrite=True, save_anndata=False)
    (outdir/'run_settings.json').write_text(json.dumps(vars(args),indent=2))

if __name__=='__main__': main()
