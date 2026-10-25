#!/usr/bin/env python3
"""Tuned ensemble runner: fuse diffusion(v1) + cca(v9) + TUNED maxfuse via unsupervised CSLS-MNN
weight selection over a VAST weight grid (0.025 step) with a chosen CSLS k, save gamma + per-cell
FOSCTTM to OUT. If maxfuse is missing for a dataset, fuses v1+v9 only. Env: DS, CSLS_K, OUT."""
import os, json, numpy as np, pandas as pd
from scipy.stats import rankdata
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
DS = os.environ["DS"]; K = int(os.environ.get("CSLS_K", "10")); OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
# Where the three component couplings live. The tree is
# runs/<regime>/<pair-type>/<dataset>/<method>/, so COUPLING_BASE is the
# dataset directory; without it, fall back to the flat runs/<method>/<DS>.
BASE = os.environ.get("COUPLING_BASE", "")
def comp(method):
    return f"{BASE}/{method}" if BASE else f"{OT_ROOT}/{method}/{DS}"

def cells_of(p, col):
    return pd.read_csv(p)[col].astype(str).tolist()

def cells_any(method, which="mod1"):
    """Cell ids for a component, whichever file the run happened to write.
    Co-assay runs emit a per-cell FOSCTTM table; diagonal runs emit cells_mod*.csv
    or, for diffusion, row/col names inside gamma_metadata.npz."""
    d = comp(method)
    for f in (f"{d}/per_cell_foscttm.csv", f"{d}/per_cell_nfoscttm_metrics.csv"):
        if os.path.exists(f) and which == "mod1":
            return cells_of(f, "cell_id")
    f = f"{d}/cells_{which}.csv"
    if os.path.exists(f):
        return cells_of(f, "cell_id")
    z = f"{d}/gamma_metadata.npz"
    if os.path.exists(z):
        dd = np.load(z, allow_pickle=True)
        k = "row_names" if which == "mod1" else "col_names"
        if k in dd.files:
            return [str(x) for x in dd[k]]
    return None
g1 = np.load(f"{comp('diffusion')}/gamma.npy").astype(np.float64)
c1 = cells_any("diffusion", "mod1")
d9 = np.load(f"{comp('cca_ot')}/pairwise_matrix_common.npy").astype(np.float64)
c9 = cells_any("cca_ot", "mod1")
has_mf = os.path.exists(f"{comp('maxfuse')}/pairwise_matrix_common.npy")
SQUARE = (g1.ndim == 2 and g1.shape[0] == g1.shape[1])
def align(D, cf, ct):
    """Reorder D's rows (and, when square, its columns) into ct's order.
    A diagonal plan is n1 x n2 with independent axes, so the column axis is
    left alone; the components all come from the same inputs in the same order,
    and a shape check below catches it if that ever stops being true."""
    if cf is None or ct is None or list(cf) == list(ct):
        return D
    pos = {c: i for i, c in enumerate(cf)}
    if not all(c in pos for c in ct):
        return D
    idx = np.array([pos[c] for c in ct])
    return D[np.ix_(idx, idx)] if SQUARE else D[idx, :]
d9 = align(d9, c9, c1); n = len(c1) if c1 is not None else g1.shape[0]
if d9.shape != g1.shape:
    raise SystemExit(f"component shapes disagree: diffusion {g1.shape} vs cca {d9.shape}")
r1 = np.apply_along_axis(rankdata, 1, -g1); r9 = np.apply_along_axis(rankdata, 1, d9)
if has_mf:
    mf = np.load(f"{comp('maxfuse')}/pairwise_matrix_common.npy").astype(np.float64)
    cmf = cells_any("maxfuse", "mod1")
    mf = align(mf, cmf, c1)
    if mf.shape != g1.shape:
        raise SystemExit(f"component shapes disagree: diffusion {g1.shape} vs maxfuse {mf.shape}")
    rmf = np.apply_along_axis(rankdata, 1, mf)

def fos(D):
    if D.shape[0] != D.shape[1]:
        return None            # diagonal pair: no true match to rank against
    dg = np.diag(D)
    fr = ((D < dg[:, None]).sum(1) + 0.5 * ((D == dg[:, None]).sum(1) - 1)) / max(1, n - 1)
    fc = ((D < dg[None, :]).sum(0) + 0.5 * ((D == dg[None, :]).sum(0) - 1)) / max(1, n - 1)
    return 0.5 * (fr + fc)
def csls(S, k):
    rA = np.sort(S, 1)[:, -k:].mean(1); rB = np.sort(S, 0)[-k:, :].mean(0); return 2 * S - rA[:, None] - rB[None, :]
def unsup(fused, k):
    n1, n2 = fused.shape
    if max(n1, n2) > 3000:
        rs = np.random.RandomState(0)
        ri = rs.choice(n1, 3000, replace=False) if n1 > 3000 else np.arange(n1)
        ci = rs.choice(n2, 3000, replace=False) if n2 > 3000 else np.arange(n2)
        S = -fused[np.ix_(ri, ci)]
    else: S = -fused
    C = csls(S, k); a = C.argmax(1); b = C.argmax(0); ai = np.array([i for i in range(len(a)) if b[a[i]] == i])
    return float(np.mean(C[ai, a[ai]])) if len(ai) > 5 else -1e9
def mk(w): return w[0]*r1 + w[1]*r9 + (w[2]*rmf if has_mf else 0.0)

step = 0.025; va = np.round(np.arange(0, 1 + 1e-9, step), 3); W = []
for a in va:
    for b in va[va <= 1 - a + 1e-9]:
        W.append((float(a), float(b), float(round(1 - a - b, 3))))
if not has_mf: W = [(a, b, 0.0) for (a, b, c) in W if c == 0.0]
best = max(W, key=lambda w: unsup(mk(w), K))
fused = mk(best); fosv = fos(fused)
def sinkhorn(C, eps=0.05, it=150):
    C = np.asarray(C, np.float64); s = np.median(C) + 1e-12; Kk = np.exp(-C / (s * eps))
    aa = np.full(C.shape[0], 1/C.shape[0]); bb = np.full(C.shape[1], 1/C.shape[1]); u = np.ones(C.shape[0]); v = np.ones(C.shape[1])
    for _ in range(it): u = aa/(Kk@v+1e-300); v = bb/(Kk.T@u+1e-300)
    G = u[:, None]*Kk*v[None, :]; return G/(G.sum()+1e-300)
np.save(f"{OUT}/gamma.npy", sinkhorn(fused).astype(np.float64))
if fosv is not None:
    pd.DataFrame({"cell_id": c1, "foscttm_sym": fosv}).to_csv(f"{OUT}/per_cell_foscttm.csv", index=False)
else:
    pd.DataFrame({"cell_id": c1}).to_csv(f"{OUT}/cells_mod1.csv", index=False)
json.dump({"model": "ensemble_tuned", "csls_k": K, "grid": "vast_0.025", "selected_w": list(best),
           "has_maxfuse": has_mf, "Mean_FOSCTTM": (float(np.mean(fosv)) if fosv is not None else None)}, open(f"{OUT}/run_settings.json", "w"), indent=2)
print(f"ENSEMBLE_TUNED {DS} w={best} k={K} "
      f"FOSCTTM={(f'{np.mean(fosv):.4f}' if fosv is not None else 'n/a (diagonal)')}", flush=True)
