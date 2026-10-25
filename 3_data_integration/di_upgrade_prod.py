#!/usr/bin/env python3
"""Production DI upgrade: ElasticFlow at the regime's selected lambda + removal.

Lambda is fixed per regime from the 920-job selection test:
    co-assay  lambda = 0.1
    diagonal  lambda = 0.75
The test found co-assay barely responds to lambda at all (win rates at or
below coin-flip) while diagonal rna-atac/methylome improve monotonically up to
0.75, so a single global lambda would have been wrong for one regime or the
other.

Four arms per (coupling):
    baseline      the tuned embedding, untouched (computed once, not per
                  coupling - it does not depend on the coupling)
    upgraded      ElasticFlow warp at the regime lambda
    removed       the same warp with ambiguous cells dropped
    rand_removed  the same warp with the SAME NUMBER dropped at random,
                  3 draws averaged

The random arm is what makes the removal claim testable. Removal always helps
some metrics simply by shrinking the problem; only beating a count-matched
random draw shows the ambiguity ranking picked the right cells.

Ambiguity is LF-Amb: data integration is unsupervised, so
neither modality's labels may decide what to drop, and that configuration
scores both modalities label-free.

WNN removal re-embeds the wnn_wsnn.mtx sub-graph instead of slicing embedding
rows, because WNN's product is the graph - dropping coordinates would leave
edges to removed cells implicit in the remaining ones.

Metrics: nmi, ari, asw_label (silhouette), batch_asw, ilisi.

usage: di_upgrade_prod.py <tool> <dataset>
"""
import glob
import json
import os
import sys
import warnings
import zlib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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
for p in (f"{P}/3_data_integration", f"{P}/3_data_integration", P):
    sys.path.insert(0, p)
os.chdir(P)

import scipy.io as sio
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (normalized_mutual_info_score as NMI,
                             adjusted_rand_score as ARI, silhouette_samples)
import fusion_v11 as EFLOW

DI = f"{DI_ROOT}"
OUT = f"{DI}/upgrade_prod"
DET = "LF-Amb"
LAM = {"coassay": 0.1, "diagonal": 0.75}
COUPLINGS = ["maxfuse", "ensemble", "cca_ot", "diffusion"]
RAND_DRAWS = 3
SCORE_CAP = 2500
SEED = 0
PTL = {"rna_adt": "rna-adt", "rna_atac": "rna-atac",
       "rna_meth": "rna-methylome", "rna_rna": "rna-rna"}
CANON = {"coassay": "coassay", "diagonal": "closedset"}
MET = ["nmi", "ari", "asw_label", "batch_asw", "ilisi"]


# inlined from 3_data_integration/evaluate_integration_metrics.py: importing
# that module pulls in integration_io and most of sklearn, which measured 267s
# on a cold gdata cache - longer than the actual work. Copied verbatim so the
# numbers stay comparable with the production tables.
def asw_label(X, labels):
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return np.nan
    try:
        return float((np.mean(silhouette_samples(X, labels,
                                                 metric="euclidean")) + 1) / 2)
    except Exception:
        return np.nan


def batch_asw(X, batch, cell_type):
    vals = []
    for ct in np.unique(cell_type):
        idx = np.where(cell_type == ct)[0]
        if len(idx) < 3 or len(np.unique(batch[idx])) < 2:
            continue
        try:
            s = silhouette_samples(X[idx], batch[idx], metric="euclidean")
            vals.append(1 - np.mean(np.abs(s)))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


def lisi_scores(X, labels, k=30, conserve=False):
    labels = np.asarray(labels).astype(str)
    ncat = len(np.unique(labels))
    if ncat < 2 or X.shape[0] < 3:
        return np.nan
    k = min(k, X.shape[0] - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(X)
    inds = nbr.kneighbors(return_distance=False)[:, 1:]
    eff = []
    for row in inds:
        _, c = np.unique(labels[row], return_counts=True)
        p = c / c.sum()
        eff.append(1 / np.sum(p * p))
    eff = np.asarray(eff)
    if conserve:
        return float(1 - np.mean((eff - 1) / max(1, ncat - 1)))
    return float(np.mean((eff - 1) / max(1, ncat - 1)))


def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def score(Z, ct, mod):
    Z = np.asarray(Z, float)
    ct = np.asarray(ct).astype(str)
    mod = np.asarray(mod).astype(str)
    n = len(Z)
    if n > SCORE_CAP:
        r = np.random.RandomState(SEED)
        i1 = np.where(mod == "mod1")[0]
        i2 = np.where(mod == "mod2")[0]
        k1 = int(round(SCORE_CAP * len(i1) / n))
        s = np.concatenate([r.choice(i1, min(k1, len(i1)), False),
                            r.choice(i2, min(SCORE_CAP - k1, len(i2)), False)])
        Z, ct, mod = Z[s], ct[s], mod[s]
    out = {m: np.nan for m in MET}
    k = len(set(ct.tolist()))
    if len(Z) < 5 or k < 2:
        return out
    cl = MiniBatchKMeans(k, n_init=3, batch_size=1024,
                         random_state=SEED).fit_predict(Z).astype(str)
    out["nmi"] = float(NMI(ct, cl))
    out["ari"] = float(ARI(ct, cl))
    for key, fn in (("asw_label", lambda: asw_label(Z, ct)),
                    ("batch_asw", lambda: batch_asw(Z, mod, ct)),
                    ("ilisi", lambda: lisi_scores(Z, mod, conserve=False))):
        try:
            out[key] = float(fn())
        except Exception:
            pass
    return out


def wnn_reembed(ds, reg, keep, ids):
    """Re-derive a WNN embedding from the sub-graph on the kept cells."""
    base = f"{DI}/wnn/{CANON[reg]}/baseline/{ds}"
    g, cf = f"{base}/wnn_wsnn.mtx", f"{base}/wnn_cells.tsv"
    if not (os.path.exists(g) and os.path.exists(cf)):
        return None
    W = sio.mmread(g).tocsr()
    cells = [l.strip() for l in open(cf) if l.strip()]
    if W.shape[0] != len(cells):
        return None
    pos = {c: i for i, c in enumerate(cells)}
    if any(c not in pos for c in ids):
        return None
    sel = np.array([pos[c] for c in ids])
    sub = W[np.ix_(sel[keep], sel[keep])]
    sub = (sub + sub.T) * 0.5
    dd = np.asarray(sub.sum(1)).ravel()
    dd[dd <= 0] = 1e-12
    Dm = sp.diags(1.0 / np.sqrt(dd))
    L = Dm @ sub @ Dm
    k = min(16, min(L.shape) - 2)
    try:
        _, V = sp.linalg.eigsh(L, k=k, which="LA")
    except Exception:
        return None
    return np.asarray(V, float)


def main():
    tool, ds = sys.argv[1], sys.argv[2]
    m = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))[ds]
    reg, ptr = m["regime"], m["pair_type"]
    ptl = PTL[ptr]
    lam = LAM[reg]

    f = f"{DI}/{tool}/{CANON[reg]}/baseline/{ds}/embeddings.csv"
    if not os.path.exists(f):
        raise SystemExit(f"no baseline embedding: {f}")
    e = pd.read_csv(f, dtype={"cell_id": str})
    cols = [c for c in e.columns if c.startswith("emb_")]
    m1 = e[e.modality == "mod1"].reset_index(drop=True)
    m2 = e[e.modality == "mod2"].reset_index(drop=True)
    n1, n2 = len(m1), len(m2)
    Z = np.concatenate([m1[cols].to_numpy(float), m2[cols].to_numpy(float)])
    ct = np.concatenate([m1.cell_type.astype(str), m2.cell_type.astype(str)])
    mod = np.array(["mod1"] * n1 + ["mod2"] * n2)
    ids = {"mod1": m1.cell_id.astype(str).tolist(),
           "mod2": m2.cell_id.astype(str).tolist()}
    allids = ids["mod1"] + ids["mod2"]
    i1, i2 = np.arange(n1), np.arange(n1, n1 + n2)
    rows = []

    def row(coupling, arm, s, pct, n):
        rows.append(dict(tool=tool, dataset=ds, regime=reg, pair_type=ptl,
                         coupling=coupling, detector=DET, arm=arm, lam=lam,
                         pct_removed=round(pct, 2), n=int(n), **s))

    # baseline does not depend on the coupling; computed once, tagged "none"
    row("none", "baseline", score(Z, ct, mod), 0.0, n1 + n2)

    for coupling in COUPLINGS:
        g = glob.glob(f"{OT_ROOT}/{reg}/*/{ds}/{coupling}/gamma.npy")
        if not g:
            continue
        G = np.asarray(np.load(g[0], mmap_mode="r"), float)
        if G.shape != (n1, n2):
            continue
        try:
            Zf, _ = EFLOW.fuse(Z, i1, i2, G, alphas=(lam,))
        except Exception as ex:
            print(f"  [warn] fuse {coupling}: {ex}", flush=True)
            continue
        Zf = np.asarray(Zf, float)
        row(coupling, "upgraded", score(Zf, ct, mod), 0.0, n1 + n2)

        drop = np.zeros(n1 + n2, bool)
        for off, mn in ((0, "mod1"), (n1, "mod2")):
            af = (f"{AMB_ROOT}/{reg}/{ptl}/{ds}/{coupling}/{DET}"
                  f"/ambiguous_cell_ids_{mn}.csv")
            if os.path.exists(af):
                s = set(pd.read_csv(af, dtype=str)["cell_id"])
                idx = [i for i, c in enumerate(ids[mn]) if c in s]
                if idx:
                    drop[off + np.array(idx, int)] = True
        if not drop.any():
            continue
        keep = ~drop
        nd1, nd2 = int(drop[:n1].sum()), int(drop[n1:].sum())
        pct = 100.0 * drop.mean()
        if keep.sum() < 10 or len(set(ct[keep].tolist())) < 2:
            continue

        if tool == "wnn":
            Zw = wnn_reembed(ds, reg, keep, allids)
            if Zw is not None and len(Zw) == int(keep.sum()):
                row(coupling, "removed_graph", score(Zw, ct[keep], mod[keep]),
                    pct, keep.sum())
            else:
                row(coupling, "removed", score(Zf[keep], ct[keep], mod[keep]),
                    pct, keep.sum())
        else:
            row(coupling, "removed", score(Zf[keep], ct[keep], mod[keep]),
                pct, keep.sum())

        # count-matched random control on the same warped embedding
        acc = []
        for t in range(RAND_DRAWS):
            rs = np.random.RandomState(seed_of(ds, coupling, t))
            rk = np.ones(n1 + n2, bool)
            if nd1:
                rk[rs.choice(n1, nd1, replace=False)] = False
            if nd2:
                rk[n1 + rs.choice(n2, nd2, replace=False)] = False
            if rk.sum() < 10 or len(set(ct[rk].tolist())) < 2:
                continue
            acc.append(score(Zf[rk], ct[rk], mod[rk]))
        if acc:
            row(coupling, "rand_removed",
                {k: float(np.nanmean([a[k] for a in acc])) for k in MET},
                pct, int(keep.sum()))

    if not rows:
        raise SystemExit(f"nothing scored for {tool}/{ds}")
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{tool}__{ds}.csv", index=False)
    print(f"DIPROD_DONE {tool} {ds} rows={len(rows)} lam={lam}", flush=True)


if __name__ == "__main__":
    main()
