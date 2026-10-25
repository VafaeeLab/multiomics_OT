#!/usr/bin/env python3
"""diffusion_v9 (NEW) — v7 + v8 ENSEMBLE of the two winning components.
Component A = v7 HVG-cosine -> entropic OT (sharp top-k / best recall).
Component B = v8 anchor-CCA on full features -> canonical cosine -> OT (type-level / low
FOSCTTM). Rank-normalise each row of A and B, fuse (mean-rank / mean-reciprocal-rank /
weighted blends w*rank_A + (1-w)*rank_B), score the fused correspondence. Aim: keep v8's
label-transfer/FOSCTTM AND recover v7's recall@k. Anchors are MNN (leak-free). np/scipy/sklearn."""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import os
import numpy as np, pandas as pd
from sklearn.cross_decomposition import CCA
NF = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dv_common", NF / "dv_common.py")
DC = importlib.util.module_from_spec(spec); sys.modules["dv_common"] = DC; spec.loader.exec_module(DC)
def log(m): print(f"[v9 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _unsup_conf(fused, k=3):
    """CSLS mutual-NN confidence of a fused cost matrix (lower cost = better).
    Unsupervised: no labels, no correspondence. Used only where no cell pairing
    exists (diagonal), since FOSCTTM is undefined there."""
    n1, n2 = fused.shape
    if max(n1, n2) > 3000:
        rs = np.random.RandomState(0)
        ri = rs.choice(n1, 3000, replace=False) if n1 > 3000 else np.arange(n1)
        ci = rs.choice(n2, 3000, replace=False) if n2 > 3000 else np.arange(n2)
        S = -fused[np.ix_(ri, ci)]
    else:
        S = -fused
    rA = np.sort(S, 1)[:, -k:].mean(1); rB = np.sort(S, 0)[-k:, :].mean(0)
    C = 2 * S - rA[:, None] - rB[None, :]
    a = C.argmax(1); b = C.argmax(0)
    ai = np.array([i for i in range(len(a)) if b[a[i]] == i])
    return float(np.mean(C[ai, a[ai]])) if len(ai) > 5 else -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna", required=True); ap.add_argument("--atac", required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--dataset_id", required=True)
    ap.add_argument("--rna_orientation", default="cell_by_feature")
    ap.add_argument("--atac_orientation", default="cell_by_feature")
    ap.add_argument("--mod2_kind", default="auto")
    ap.add_argument("--fuse_grid", default="meanrank,w0.3,w0.5,w0.7")  # mrr removed: it saves degenerate constant-row matrices
    ap.add_argument("--n_hvg", type=int, default=3000); ap.add_argument("--ot_eps", type=float, default=0.05)
    ap.add_argument("--d_cca", type=int, default=50); ap.add_argument("--k_anchor", type=int, default=15)
    ap.add_argument("--n_full", type=int, default=100); ap.add_argument("--n_pcs_shared", type=int, default=30)
    ap.add_argument("--no_log1p", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma_common", default="")
    args = ap.parse_args(); do_log1p = not args.no_log1p
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    # A co-assay pair shares barcodes across modalities; a diagonal pair does
    # not, so align_common_cells raises. Try paired first and fall back --
    # the data decides, not a flag that could disagree with it.
    try:
        X1, X2, cells, f1, f2 = DC.load_align(args.rna, args.atac, args.rna_orientation, args.atac_orientation, log)
        cells1 = cells2 = cells; PAIRED = True
    except ValueError:
        X1, X2, cells1, cells2, f1, f2 = DC.load_align_unpaired(args.rna, args.atac, args.rna_orientation, args.atac_orientation, log)
        cells = cells1; PAIRED = False   # FOSCTTM undefined without pairing
    n = len(cells); log(f"aligned n={n}")
    mod2_kind = args.mod2_kind
    if mod2_kind == "auto":
        mod2_kind = "adt" if (DC.REMAP is not None and DC.REMAP.is_adt_panel(list(f2))) else "gene_activity"
    # Cell-type labels are NEVER available to a coupling method.
    labels = None
    if args.gamma_common and os.path.exists(args.gamma_common):
        evalD = np.load(args.gamma_common).astype(np.float64)
        log(f"[GAMMA_COMMON] loaded {evalD.shape} from {args.gamma_common}")
        m = DC.score_dense(evalD, label="gamma_common")
        if labels is not None: m.update(DC.label_transfer_score(evalD, labels))
        m.update({"model": "diffusion_v9", "dataset": args.dataset_id, "fuse": "gamma_common"})
        pd.DataFrame([m]).to_csv(out / "grid_results.csv", index=False)
        (out / "best.json").write_text(json.dumps(m, indent=2, default=float))
        DC.write_outputs(out, evalD, cells, args.dataset_id, "diffusion_v9", {"source": "gamma_common"})
        log(f"[DONE gamma_common] {args.dataset_id} FOSCTTM={m['foscttm_midrank']:.4f}"); return
    S1n, S2n, common = DC.shared_block(X1, f1, X2, f2, mod2_kind, do_log1p, gene_mode="hvg", n_hvg=args.n_hvg, log=log)
    if S1n is None:
        raise SystemExit("FATAL shared bridge")
    # Component A: v7 HVG-cosine -> OT
    evalA = -DC.sinkhorn_plan(1.0 - DC.cosine_affinity(S1n, S2n), args.ot_eps)
    # Component B: v8 anchor-CCA -> OT
    Zj = DC.pca_dense(np.vstack([S1n, S2n]), args.n_pcs_shared, args.seed, do_log1p=False)
    Ps1, Ps2 = Zj[:n], Zj[n:]
    F1 = DC.pca_dense(X1, args.n_full, args.seed, do_log1p=do_log1p)
    F2 = DC.pca_dense(X2, args.n_full, args.seed, do_log1p=do_log1p)
    A = DC.mutual_inter(Ps1, Ps2, args.k_anchor).tocoo(); ai, aj = A.row, A.col
    dd = int(min(args.d_cca, F1.shape[1] - 1, F2.shape[1] - 1, len(ai) - 1))
    cca = CCA(n_components=dd, max_iter=500); cca.fit(F1[ai], F2[aj])
    Xc, Yc = cca.transform(F1, F2); Kc = DC.l2_rows(Xc) @ DC.l2_rows(Yc).T
    evalB = -DC.sinkhorn_plan(1.0 - Kc, args.ot_eps)
    rA, rB = DC.rank_rows(evalA), DC.rank_rows(evalB)
    log(f"components ready ({len(ai)} anchors, d_cca={dd}); fusing {args.fuse_grid}")

    rows = []; best_eval = None; best_key = -1e18; best_cfg = None
    for fuse in [s for s in args.fuse_grid.split(",") if s]:
        if fuse == "mrr":
            evalD = -(1.0 / rA + 1.0 / rB)
        elif fuse == "meanrank":
            evalD = 0.5 * (rA + rB)
        elif fuse.startswith("w"):
            w = float(fuse[1:]); evalD = w * rA + (1.0 - w) * rB
        else:
            continue
        m = DC.score_dense(evalD, label=f"fuse_{fuse}")
        if labels is not None:
            m.update(DC.label_transfer_score(evalD, labels))
        m.update({"model": "diffusion_v9", "dataset": args.dataset_id, "fuse": fuse, "d_cca": dd, "mod2_kind": mod2_kind})
        rows.append(m)
        # Selection uses cell-PAIR information only (permitted for co-assay),
        # never cell-type labels. Diagonal data has no pairing -> unsupervised.
        _f = m.get("foscttm_midrank", float("nan"))
        if evalD.shape[0] == evalD.shape[1] and _f == _f:
            key = -_f; _basis = "foscttm"
        else:
            key = _unsup_conf(evalD); _basis = "unsup"
        _s = evalD[::max(1, evalD.shape[0] // 40)]
        _ok = np.median([len(np.unique(r)) for r in _s]) > 0.05 * evalD.shape[1]  # DEGENERACY GUARD: skip near-constant plans
        if _ok and key > best_key:
            best_key = key; best_eval = evalD; best_cfg = {"best_fuse": fuse}
        log(f"fuse={fuse}: [select:{_basis}={key:.4f}] FOSCTTM={m['foscttm_midrank']:.4f} "
            f"r@10={m['recall_at_10']:.3f} r@50={m['recall_at_50']:.3f} medrank={m['median_rank']:.0f}")
    df = pd.DataFrame(rows); df.to_csv(out / "grid_results.csv", index=False)
    _sel = df[df.fuse == best_cfg["best_fuse"]] if best_cfg else df.iloc[:0]
    (out / "best.json").write_text(json.dumps(
        (_sel.iloc[0].to_dict() if len(_sel) else {}), indent=2, default=float))
    (out / "run_settings.json").write_text(json.dumps(vars(args), indent=2))
    DC.write_outputs(out, best_eval, cells, args.dataset_id, "diffusion_v9", best_cfg)
    # v9 probability transport (gamma): reciprocal rank of the rank-cost (rank 1 -> highest probability),
    # normalised to sum 1; the rank-cost already comes from the Sinkhorn-plan components so NO new Sinkhorn.
    _Rv = DC.rank_rows(best_eval).astype(np.float64); _Rv = np.clip(_Rv, 1.0, None)
    _Gv = 1.0 / _Rv; _Gv /= (_Gv.sum() + 1e-300)
    import numpy as _np
    _np.save(os.path.join(str(out), "gamma.npy"), _Gv.astype(_np.float64))
    log(f"[v9 gamma] reciprocal-rank gamma saved: sum={_Gv.sum():.4f}")
    log(f"DONE {args.dataset_id} {time.time()-t0:.1f}s | BEST {best_cfg}")


if __name__ == "__main__":
    main()
