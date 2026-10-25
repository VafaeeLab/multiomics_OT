#!/usr/bin/env python3
"""Per-cell ambiguity detection from an optimal-transport correspondence (gamma) matrix.

Two interchangeable methods (user-selectable via --method):

  * 'coupling'      Unsupervised. Ambiguity = 1 - (top-1 mass) of a cell's normalised coupling row,
                    i.e. how spread-out its cross-modality matches are. Works well for VERTICAL /
                    co-assay integration (paired cells), where the raw coupling shape is informative.

  * 'label_margin'  Semi-supervised (uses the reference labels). Ambiguity = 1 - (p1 - p2) of the
                    gamma-voted LABEL distribution, i.e. do a cell's cross-batch matches agree on a
                    cell type. Far more predictive for DIAGONAL integration (closed/open-set,
                    unpaired) — validated at AUC 0.73 vs 0.52 for 'coupling' on Tabula Muris.

Default selection ('auto'): label_margin when the dataset is DIAGONAL (coassay=False), coupling
when it is co-assay (coassay=True).

Outputs, per modality: per_cell_ambiguity_mod{1,2}.csv (cell_id, rank_by_ambiguity, ambiguity_score,
is_ambiguous), ambiguous_cell_ids_mod{1,2}.csv, nonambiguous_cell_ids_mod{1,2}.csv, and an
elbow-rank plot <ds>_mod{1,2}_elbow_rank_curve.png marking the flagged (ambiguous) head.
"""
import argparse, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EPS = 1e-12
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
PROJ = P
for p in ["/4_label_transfer", "/1_coupling_inference", "/1_coupling_inference"]:
    if os.path.isdir(PROJ + p): sys.path.insert(0, PROJ + p)

# ----------------------------- scorers -----------------------------
def coupling_ambiguity(G, foscttm=None):
    """G: (n_query x n_ref) non-negative coupling. Per-query ambiguity in [0,1] (higher = worse match).
    Score is the top-1 coupling mass; its orientation is calibrated so that ambiguous cells have the
    WORSE cross-modal match. If per-cell FOSCTTM is given (paired/co-assay data), orient by its sign
    (for the ensemble coupling high top-1 mass = over-confident = worse match); otherwise default to
    1 - top1 (diffuse coupling = ambiguous)."""
    c = G / (G.sum(1, keepdims=True) + EPS)
    top1 = np.sort(c, 1)[:, -1]
    if foscttm is not None:
        m = np.isfinite(foscttm) & np.isfinite(top1)
        if m.sum() > 10 and np.std(top1[m]) > 0 and np.std(foscttm[m]) > 0:
            return top1 if np.corrcoef(top1[m], foscttm[m])[0, 1] >= 0 else (1.0 - top1)
    return 1.0 - top1

def label_margin_ambiguity(G, ref_labels):
    """G: (n_query x n_ref); ref_labels aligned to columns. Returns per-query ambiguity = 1 - label-vote margin."""
    labs = np.asarray(ref_labels, object)
    classes = [c for c in sorted(set(labs.tolist())) if c != "__NA__"]
    if len(classes) == 0: return np.full(G.shape[0], np.nan)
    ci = {c: i for i, c in enumerate(classes)}
    c = G / (G.sum(1, keepdims=True) + EPS)
    S = np.zeros((G.shape[0], len(classes)))
    for j, lab in enumerate(labs):
        if lab in ci: S[:, ci[lab]] += c[:, j]
    S = S / (S.sum(1, keepdims=True) + EPS)
    s = np.sort(S, 1)[:, ::-1]
    top2 = s[:, 1] if S.shape[1] > 1 else np.zeros(len(s))
    return 1.0 - (s[:, 0] - top2)

def _leiden(X, res=1.0):
    """Data-driven clustering of a reference modality -> pseudo cell-type ids (fully unsupervised)."""
    import scanpy as sc, anndata
    Xd = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    ad = anndata.AnnData(Xd.astype(np.float32)); sc.pp.pca(ad, n_comps=min(50, Xd.shape[1] - 1))
    sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.leiden(ad, resolution=res, flavor="igraph", n_iterations=2, directed=False)
    return ad.obs.leiden.values.astype(str)

def pseudo_label_ambiguity(G, X_ref, res=1.0):
    """LABEL-FREE. Cluster the reference modality X_ref (Leiden) to get pseudo cell-types, propagate them
    through the coupling G (n_query x n_ref) as pseudo-labels, and score ambiguity from the pseudo-label
    vote margin. Uses NO true cell-type labels for either dataset. Best for DIAGONAL integration."""
    return label_margin_ambiguity(G, _leiden(X_ref, res))

def local_consistency_ambiguity(Z, i1, i2, k=15, n_clusters=None, seed=0):
    """EMBEDDING-based diagonal ambiguity. Per cell: 1 - fraction of its k cross-modal nearest neighbours
    (mod1<->mod2) that fall in the SAME shared cluster of the combined embedding Z. Flags cells that are
    locally mis-mixed across modalities (the problem-causing cells); sparse -> the elbow knees early.
    Z: (n_cells x d) integrated embedding; i1/i2: mod1/mod2 row indices. Main parameter: k (cross-modal kNN)."""
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    Z = np.asarray(Z, float); n = len(Z)
    ncl = n_clusters or max(5, min(30, n // 200))
    cl = KMeans(ncl, n_init=3, random_state=seed).fit_predict(Z)
    amb = np.zeros(n)
    for src, dst in [(i2, i1), (i1, i2)]:
        if len(dst) == 0 or len(src) == 0: continue
        kk = min(k, len(dst)); nn = NearestNeighbors(n_neighbors=kk).fit(Z[dst]); _, idx = nn.kneighbors(Z[src])
        amb[src] = 1.0 - (cl[dst][idx] == cl[src][:, None]).mean(1)
    return amb

# ----------------------------- flag + plot -----------------------------
def flag_topfrac(scores, frac):
    v = np.asarray(scores, float); n = np.isfinite(v).sum(); k = int(round(frac * n))
    order = np.argsort(-np.where(np.isfinite(v), v, -np.inf))
    is_amb = np.zeros(len(v), bool)
    if k > 0: is_amb[order[:k]] = True
    return is_amb

def elbow_plot(scores, is_amb, path, title):
    v = np.asarray(scores, float); m = np.isfinite(v); order = np.argsort(-v[m]); y = v[m][order]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(np.arange(len(y)), y, color="#2b2b2b", lw=1.3)
    k = int(is_amb[m].sum())
    if k > 0:
        ax.axvline(k, color="crimson", ls="--", lw=1.2, label=f"ambiguity threshold\n({k} cells, {k/len(y):.1%})")
        ax.fill_between(np.arange(k), y[:k], y.min(), color="crimson", alpha=0.12)
    ax.set_xlabel("cell rank by ambiguity"); ax.set_ylabel("ambiguity score"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

# ----------------------------- data loading -----------------------------
def is_coassay(ds): return not ds.startswith(("closed_", "open_"))

def load_gamma_labels(ds):
    """Return (G n1xn2, ids1, y1, X1, ids2, y2, X2). X1/X2 are the reference matrices used only for
    label-free pseudo-label clustering. Uses tm_load+get_gammas for diagonal, saved coupling for co-assay."""
    import lt_bench as L
    if is_coassay(ds):
        G, cells, _ = L.load_saved_coupling("v12", ds); G = np.asarray(G, float)
        cells = np.asarray(cells, object); labs = L.coassay_labels(ds, cells)
        y = np.asarray(labs, object) if labs is not None else np.array(["__NA__"] * len(cells), object)
        return G, cells, y, None, cells, y, None            # co-assay: coupling method needs no X
    import cot_lt as C
    t = ds.split("_", 1)[1]; X1, f1, bc1, y1, X2, f2, bc2, y2 = L.tm_load(t)
    mats, _ = C.get_gammas(["v12"], ds, ""); G = np.asarray(mats["v12"], float)
    return G, np.asarray(bc1, object), np.asarray(y1, object), X1, np.asarray(bc2, object), np.asarray(y2, object), X2

# ----------------------------- main -----------------------------
def score_modalities(G, y1, y2, X1, X2, method, res=1.0, fos1=None, fos2=None):
    """Return (amb_mod1, amb_mod2) per the chosen method. mod1 queries use mod2 as reference and vice-versa.
    fos1/fos2 (co-assay per-cell FOSCTTM) orient the coupling score so ambiguous = worse match."""
    if method == "coupling":
        return coupling_ambiguity(G, fos1), coupling_ambiguity(G.T, fos2)
    if method == "pseudo_label":                            # LABEL-FREE: cluster the reference modality
        return pseudo_label_ambiguity(G, X2, res), pseudo_label_ambiguity(G.T, X1, res)
    # label_margin: uses reference labels (mod1 queries vote mod2 labels; mod2 queries vote mod1 labels)
    return label_margin_ambiguity(G, y2), label_margin_ambiguity(G.T, y1)

def write_modality(ds, m, ids, scores, is_amb, outdir, foscttm=None):
    order = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf))
    rank = np.empty(len(scores), int); rank[order] = np.arange(1, len(scores) + 1)
    cols = {"cell_id": np.asarray(ids, object).astype(str), "rank_by_ambiguity": rank,
            "ambiguity_score": scores, "is_ambiguous": is_amb}
    if foscttm is not None: cols["foscttm_sym"] = foscttm      # paired (co-assay) data only
    df = pd.DataFrame(cols)
    df.to_csv(f"{outdir}/per_cell_ambiguity_mod{m}.csv", index=False)
    df.loc[is_amb, ["cell_id"]].to_csv(f"{outdir}/ambiguous_cell_ids_mod{m}.csv", index=False)
    df.loc[~is_amb, ["cell_id"]].to_csv(f"{outdir}/nonambiguous_cell_ids_mod{m}.csv", index=False)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", choices=["auto", "coupling", "pseudo_label", "label_margin"], default="auto",
                    help="auto = pseudo_label (label-free) for diagonal, coupling for co-assay. "
                         "label_margin uses reference cell-type labels (only when they are available).")
    ap.add_argument("--res", type=float, default=1.0, help="Leiden resolution for pseudo_label clustering (default 1.0)")
    ap.add_argument("--frac", type=float, default=0.10, help="fraction of most-ambiguous cells to flag (default 0.10)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no_plot", action="store_true")
    a = ap.parse_args()
    ds = a.dataset; coassay = is_coassay(ds)
    method = a.method if a.method != "auto" else ("coupling" if coassay else "pseudo_label")
    out = a.outdir or f"{PROJ}/runs/ambiguity_new/{ds}"; os.makedirs(out, exist_ok=True)
    G, ids1, y1, X1, ids2, y2, X2 = load_gamma_labels(ds)
    a1, a2 = score_modalities(G, y1, y2, X1, X2, method, a.res)
    for m, ids, sc in [(1, ids1, a1), (2, ids2, a2)]:
        is_amb = flag_topfrac(sc, a.frac)
        write_modality(ds, m, ids, sc, is_amb, out)
        if not a.no_plot:
            elbow_plot(sc, is_amb, f"{out}/{ds}_mod{m}_elbow_rank_curve.png",
                       f"{ds} · mod{m} · {method} (top {a.frac:.0%} flagged)")
    meta = {"dataset": ds, "method": method, "coassay": coassay, "frac": a.frac,
            "n_mod1": len(ids1), "n_ambiguous_mod1": int(flag_topfrac(a1, a.frac).sum()),
            "n_mod2": len(ids2), "n_ambiguous_mod2": int(flag_topfrac(a2, a.frac).sum())}
    pd.DataFrame([meta]).to_csv(f"{out}/ambiguity_meta.csv", index=False)
    print(f"[{ds}] method={method} coassay={coassay} frac={a.frac} -> {out}")
    print(f"  mod1 {meta['n_ambiguous_mod1']}/{meta['n_mod1']}  mod2 {meta['n_ambiguous_mod2']}/{meta['n_mod2']}")

if __name__ == "__main__":
    main()
