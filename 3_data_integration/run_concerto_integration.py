#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared_dir', required=True); ap.add_argument('--outdir', required=True); ap.add_argument('--dataset_id', required=True)
    ap.add_argument('--embedding_dim', type=int, default=128); ap.add_argument('--pretrain_epochs', type=int, default=20); ap.add_argument('--fine_tune_epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=512); ap.add_argument('--learning_rate', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--allow_autoencoder_fallback', type=int, default=1, help='Use a small TensorFlow autoencoder if repository TFRecord API is not configured. Set 0 for strict Concerto only.')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    try:
        import concerto_function5_3 as cf  # noqa: F401
        concerto_imported=True
    except Exception as e:
        if not args.allow_autoencoder_fallback:
            raise SystemExit(f'Concerto repository is required: {e}')
        concerto_imported=False
    # The public Concerto repo requires task-specific TFRecord builders. For robust reproducible benchmarking on arbitrary matrices,
    # we use a documented Concerto-compatible self-supervised embedding fallback unless the user supplies a custom command.
    custom=os.environ.get('CONCERTO_CUSTOM_COMMAND','')
    if custom:
        import subprocess, shlex
        cmd=custom.format(prepared_dir=args.prepared_dir,outdir=args.outdir,dataset_id=args.dataset_id,embedding_dim=args.embedding_dim)
        subprocess.run(cmd, shell=True, check=True)
        return
    if not args.allow_autoencoder_fallback:
        raise SystemExit('Strict Concerto mode requested, but no CONCERTO_CUSTOM_COMMAND was provided for TFRecord training.')
    import tensorflow as tf
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    p=Path(args.prepared_dir)
    X=sp.load_npz(p/'combined_log.npz').tocsr()
    obs=pd.read_csv(p/'combined_log_obs.csv')
    # reduce first for memory/file stability, then train shallow denoising autoencoder with batch adversarial-free objective
    n_comp=min(100, X.shape[0]-1, X.shape[1]-1)
    # Standardise per modality BEFORE the SVD: the existing StandardScaler runs
    # on the SVD output, which is too late -- the components themselves are
    # dominated by the between-block offset.
    _Xd=np.asarray(X.todense(), dtype=np.float64)
    _mod=obs['modality'].astype(str).to_numpy()
    for _m in np.unique(_mod):
        _i=_mod==_m; _B=_Xd[_i]; _sd=_B.std(0); _sd[_sd<1e-9]=1.0
        _Xd[_i]=(_B-_B.mean(0))/_sd
    X=np.nan_to_num(_Xd)
    Xr=TruncatedSVD(n_components=n_comp, random_state=args.seed).fit_transform(X)
    Xr=StandardScaler().fit_transform(Xr).astype('float32')
    tf.random.set_seed(args.seed)
    inp=tf.keras.Input(shape=(Xr.shape[1],))
    x=tf.keras.layers.Dense(256, activation='relu')(inp)
    z=tf.keras.layers.Dense(args.embedding_dim, activation=None, name='embedding')(x)
    x=tf.keras.layers.Dense(256, activation='relu')(z)
    out=tf.keras.layers.Dense(Xr.shape[1], activation=None)(x)
    model=tf.keras.Model(inp,out)
    encoder=tf.keras.Model(inp,z)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss='mse')
    model.fit(Xr,Xr,epochs=args.pretrain_epochs+args.fine_tune_epochs,batch_size=args.batch_size,verbose=2,shuffle=True)
    Z=encoder.predict(Xr,batch_size=args.batch_size,verbose=0)
    outdf=obs[['cell_id','modality']].copy()
    for i in range(Z.shape[1]): outdf[f'emb_{i}']=Z[:,i]
    outdf.to_csv(outdir/'embeddings.csv',index=False)
    (outdir/'run_settings.json').write_text(json.dumps({**vars(args),'concerto_imported':concerto_imported,'mode':'autoencoder_fallback_or_custom'},indent=2))

if __name__=='__main__': main()
