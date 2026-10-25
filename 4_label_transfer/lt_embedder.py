#!/usr/bin/env python3
"""Self-contained encoder for the custom label-transfer models.

A copy of the data-integration encoder architecture, rewired so that label
transfer owns the whole path: it reads the bundle built from Data/ by
lt_prep_inputs.py and writes to runs_label_transfer/_lt_embeddings/. It never
reads or writes runs_data_integration, and it computes no integration metrics.

Reusing the architecture is code reuse, not result reuse - no number produced
by the data-integration benchmark enters here.
"""
_ORIG_DOC = """Attention/Transformer integration embedder v3 = the (non-collapsing) v1 backbone + MNN-anchor mixing.

Lesson from v2: per-cell token-attention + VICReg pooling COLLAPSED the latent (bio 0.33/batch 0.83 even with
zero mixing). The original v1 backbone (Linear->LN->GELU input, optional cross-cell self-attention blocks,
linear latent + decoder) does NOT collapse (bio ~0.52 on Kidney) but UNDER-mixes (batch ~0.53). v3 keeps that
backbone and adds mutual-nearest-neighbour (MNN) anchor mixing: pull only reciprocal cross-modal neighbours
together -> mixes shared cell types (raises batch) while leaving modality-private populations alone (protects
bio) -- the Seurat/Harmony strategy. Optional PCA denoise (helps noisy datasets). arch=attention -> 0 self-attn
layers (MLP), transformer -> 3. Writes embeddings.csv to runs_label_transfer/_lt_embeddings/<arch>/<ds>/."""
import sys, os, argparse, subprocess, numpy as np, pandas as pd, scipy.io
import torch, torch.nn as nn, torch.nn.functional as F
P = os.environ.get("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration")
OT_ROOT = os.environ.get("OT_ROOT", P + "/runs")
AMB_ROOT = os.environ.get("AMB_ROOT", P + "/runs_ambiguity")
DI_ROOT = os.environ.get("DI_ROOT", P + "/runs_data_integration")
LT_ROOT = os.environ.get("LT_ROOT", P + "/runs_label_transfer")
ROOT = P
PY = sys.executable
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SA(nn.Module):
    """Self-attention block with LEARNABLE RESIDUAL GATES (init small). Cross-cell self-attention mixes cell
    identities and can hurt cell-type separation; the gates let the model start ~identity (like the MLP/Attention
    backbone) and add attention only where it lowers loss -> attention can refine but not destroy bio."""
    def __init__(s, dm, h=8, p=0.1, gate=0.1):
        super().__init__(); s.ln = nn.LayerNorm(dm); s.mha = nn.MultiheadAttention(dm, h, dropout=p, batch_first=True)
        s.ln2 = nn.LayerNorm(dm); s.ff = nn.Sequential(nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        s.g1 = nn.Parameter(torch.tensor(float(gate))); s.g2 = nn.Parameter(torch.tensor(float(gate)))

    def forward(s, x):
        a, _ = s.mha(s.ln(x), s.ln(x), s.ln(x), need_weights=False); x = x + s.g1 * a; return x + s.g2 * s.ff(s.ln2(x))


class AE(nn.Module):
    def __init__(s, F_, dm, lat, lenc, gate=0.1):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(F_, dm), nn.LayerNorm(dm), nn.GELU())
        s.enc = nn.ModuleList([SA(dm, gate=gate) for _ in range(lenc)]); s.tolat = nn.Linear(dm, lat)
        s.dec = nn.Sequential(nn.Linear(lat, dm), nn.GELU(), nn.Linear(dm, F_))

    def forward(s, x):
        h = s.inp(x).unsqueeze(0)
        for blk in s.enc:
            h = blk(h)
        z = s.tolat(h.squeeze(0)); return z, s.dec(z)


class TokenAE(nn.Module):
    """Per-cell feature-token transformer: tokenize a cell's feature vector into T tokens and self-attend OVER
    TOKENS within each cell (cells stay independent -> cross-cell bio damage impossible). A genuine transformer
    that, unlike cross-cell SA, cannot blur cell-type identity."""
    def __init__(s, F_, dm, lat, nlayers, T=16, d=64, h=8, p=0.1):
        super().__init__()
        s.T = T; s.d = d
        s.tok = nn.Linear(F_, T * d); s.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, h, dim_feedforward=2 * d, dropout=p,
                                  activation="gelu", batch_first=True, norm_first=True) for _ in range(max(1, nlayers))])
        s.tolat = nn.Sequential(nn.LayerNorm(T * d), nn.Linear(T * d, dm), nn.GELU(), nn.Linear(dm, lat))
        s.dec = nn.Sequential(nn.Linear(lat, dm), nn.GELU(), nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, F_))

    def forward(s, x):
        n = x.shape[0]; h = s.tok(x).view(n, s.T, s.d) + s.pos
        for L in s.layers:
            h = L(h)
        z = s.tolat(h.reshape(n, -1)); return z, s.dec(z)


def rbf_mmd(x, y, gs=(0.5, 1, 2)):
    def k(a, b):
        d = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(2); return sum(torch.exp(-g * d) for g in gs)
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


def mnn_pairs(Z1, Z2, k=15):
    """Mutual nearest neighbours between two latent sets -> (idx1, idx2). Aligns only reciprocal cross-modal
    neighbours (shared cell types); modality-private populations are left untouched."""
    from sklearn.neighbors import NearestNeighbors
    k1 = min(k, len(Z2)); k2 = min(k, len(Z1))
    if k1 < 1 or k2 < 1:
        return np.array([], int), np.array([], int)
    nn12 = NearestNeighbors(n_neighbors=k1).fit(Z2).kneighbors(Z1, return_distance=False)
    nn21 = NearestNeighbors(n_neighbors=k2).fit(Z1).kneighbors(Z2, return_distance=False)
    set21 = [set(r.tolist()) for r in nn21]
    p1, p2 = [], []
    for i in range(len(Z1)):
        for j in nn12[i]:
            if i in set21[j]:
                p1.append(i); p2.append(int(j))
    return np.asarray(p1, int), np.asarray(p2, int)


def load_partners(ds, mod):
    """Disabled here. The gamma-aware path reads a coupling from the
    data-integration tree, which label transfer is not allowed to touch. It is
    only reachable via --gamma_aware, which this pipeline never passes; the
    guard makes that a hard error instead of a silent dependency."""
    raise SystemExit(
        "gamma_aware is not available in the label-transfer encoder: it would "
        "read runs_data_integration/gamma, breaking independence")

def _resolve_prep(ds):
    """The label-transfer bundle, and nothing else.

    The original resolved runs_integration / Results/prepared_input, which are
    data-integration artefacts. Label transfer must not read those, so this
    copy reads only the bundle lt_prep_inputs.py builds from Data/.
    """
    d = f"{LT_ROOT}/_lt_inputs/{ds}"
    if os.path.exists(f"{d}/combined_log.mtx"):
        return d
    raise SystemExit(f"no label-transfer bundle for {ds}; run lt_prep_inputs.py first")

def run(ds, arch, lat=50, epochs=120, pca_dim=0, align="mnn", lam_mnn=1.0, lam_mmd=0.1,
        mnn_k=15, mnn_warmup=20, mnn_refresh=15, lenc_override=-1, gate=0.1, token_attn=0, gamma_aware=False, lam_anchor=1.0, tag=""):
    prep = _resolve_prep(ds)
    Xn = scipy.io.mmread(f"{prep}/combined_log.mtx").tocsr().toarray().astype(np.float32)
    obs = pd.read_csv(f"{prep}/combined_log_obs.csv"); mod = (obs.modality == "mod1").to_numpy()
    if pca_dim and pca_dim > 0 and pca_dim < Xn.shape[1] - 1:
        from sklearn.decomposition import PCA
        P = int(min(pca_dim, Xn.shape[0] - 1))
        Xn = PCA(n_components=P, svd_solver="randomized", random_state=0).fit_transform(Xn).astype(np.float32)
    if os.environ.get("PERMOD_SCALE", "1") == "1":
        # per-modality z-score: puts the two blocks on a common location/scale
        # without altering within-modality structure or cell ordering.
        for _sel in (mod, ~mod):
            _B = Xn[_sel]
            _sd = _B.std(0); _sd[_sd < 1e-6] = 1.0
            Xn[_sel] = (_B - _B.mean(0)) / _sd
        Xn = np.nan_to_num(Xn)
    else:
        Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-6)        # previous global z-score
    lenc = (0 if arch == "AnchorAE" else 3) if lenc_override < 0 else lenc_override; dm = 256
    torch.manual_seed(0); np.random.seed(0)
    X = torch.tensor(Xn, device=DEV); n = len(X)
    m1g = np.where(mod)[0]; m2g = np.where(~mod)[0]
    partner = weight = None
    if gamma_aware:
        pg, wg = load_partners(ds, mod); partner = torch.tensor(pg, device=DEV); weight = torch.tensor(wg, device=DEV)
    if token_attn:
        model = TokenAE(Xn.shape[1], dm, lat, nlayers=max(1, lenc if lenc > 0 else 2)).to(DEV)
    else:
        model = AE(Xn.shape[1], dm, lat, lenc, gate=gate).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = min(2048, n)
    mnn_g1 = mnn_g2 = None
    use_mnn = align in ("mnn", "both") and len(m1g) > 0 and len(m2g) > 0
    for ep in range(epochs):
        if use_mnn and ep >= mnn_warmup and (ep - mnn_warmup) % mnn_refresh == 0:
            model.eval()
            with torch.no_grad():
                Zf = np.concatenate([model(X[i:i + 4096])[0].cpu().numpy() for i in range(0, n, 4096)])
            p1, p2 = mnn_pairs(Zf[m1g], Zf[m2g], k=mnn_k)
            mnn_g1 = m1g[p1] if len(p1) else None; mnn_g2 = m2g[p2] if len(p2) else None
        model.train(); perm = torch.randperm(n, device=DEV)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            if len(idx) < 8:
                continue
            z, xr = model(X[idx]); loss = F.mse_loss(xr, X[idx])
            if align in ("mmd", "both"):                      # distribution-level mixing
                sb = idx.cpu().numpy(); i1 = sb[mod[sb]]; i2 = sb[~mod[sb]]
                if lam_mmd > 0 and len(i1) > 8 and len(i2) > 8:
                    z1, _ = model(X[torch.tensor(i1[:256], device=DEV)])
                    z2, _ = model(X[torch.tensor(i2[:256], device=DEV)])
                    loss = loss + lam_mmd * rbf_mmd(z1, z2)
            if use_mnn and mnn_g1 is not None:                # unsupervised MNN anchor
                sel = np.random.choice(len(mnn_g1), size=min(1024, len(mnn_g1)), replace=False)
                za, _ = model(X[torch.tensor(mnn_g1[sel], device=DEV)])
                zb, _ = model(X[torch.tensor(mnn_g2[sel], device=DEV)])
                loss = loss + lam_mnn * F.mse_loss(za, zb)
            if gamma_aware:                                   # v12 OT-coupling anchor (added on top of align mode)
                zp, _ = model(X[partner[idx]]); w = weight[idx][:, None]
                loss = loss + lam_anchor * (w * (z - zp).pow(2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Z = np.concatenate([model(X[i:i + 4096])[0].cpu().numpy() for i in range(0, n, 4096)])
    method = arch
    out = f"{LT_ROOT}/_lt_embeddings/{arch}/{ds}"
    os.makedirs(out, exist_ok=True)
    d = obs[["cell_id", "modality", "cell_type", "batch", "batch_modality"]].copy()
    for j in range(Z.shape[1]):
        d[f"emb_{j}"] = Z[:, j]
    d.to_csv(f"{out}/embeddings.csv", index=False)
    npair = 0 if mnn_g1 is None else len(mnn_g1)
    # No integration metrics here. Those belong to the data-integration
    # benchmark; label transfer is scored by macro-F1 and accuracy only.
    import json as _json, time as _time
    _json.dump({"dataset": ds, "arch": arch, "epochs": epochs, "lat": lat,
                "align": align, "lam_mnn": lam_mnn, "lam_mmd": lam_mmd,
                "mnn_k": mnn_k, "gate": gate, "mnn_pairs": int(npair),
                "n_cells": int(n), "dim": int(Z.shape[1]),
                "written": _time.strftime("%Y-%m-%d %H:%M:%S")},
               open(f"{out}/embed_settings.json", "w"), indent=2)
    print(f"[{method}] {ds} done (lenc={lenc}, pca={pca_dim}, align={align}, lam_mnn={lam_mnn}, mnn_pairs={npair}, dev={DEV.type})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--arch", required=True)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--align", choices=["mnn", "mmd", "both"], default="mnn")
    ap.add_argument("--lam_mnn", type=float, default=1.0); ap.add_argument("--lam_mmd", type=float, default=0.1)
    ap.add_argument("--mnn_k", type=int, default=15); ap.add_argument("--mnn_warmup", type=int, default=20)
    ap.add_argument("--mnn_refresh", type=int, default=15); ap.add_argument("--lenc", type=int, default=-1)
    ap.add_argument("--gate", type=float, default=0.1); ap.add_argument("--token_attn", type=int, default=0)
    ap.add_argument("--lam_anchor", type=float, default=1.0)
    ap.add_argument("--gamma_aware", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.dataset, a.arch, epochs=a.epochs, pca_dim=a.pca_dim, align=a.align, lam_mnn=a.lam_mnn, lam_mmd=a.lam_mmd,
        mnn_k=a.mnn_k, mnn_warmup=a.mnn_warmup, mnn_refresh=a.mnn_refresh, lenc_override=a.lenc, gate=a.gate,
        token_attn=a.token_attn, lam_anchor=a.lam_anchor, gamma_aware=a.gamma_aware, tag=a.tag)
