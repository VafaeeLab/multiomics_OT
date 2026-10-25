import os
#!/usr/bin/env python3
"""Make ensemble_final.py work on diagonal pairs.

Four separate paired-only assumptions, all invisible while it only ever ran on
co-assay data:

  1. cell ids read from per_cell_foscttm.csv / per_cell_nfoscttm_metrics.csv,
     which diagonal runs do not write (FOSCTTM is undefined unpaired). They
     write cells_mod1.csv, and diffusion writes gamma_metadata.npz.
  2. align() reorders with D[np.ix_(idx, idx)] -- one index array for both axes.
  3. fos() takes np.diag of the fused plan.
  4. unsup() subsamples with np.ix_(idx, idx), the same bug just fixed in
     cca_ot._unsup_conf.

The weight search itself is already unsupervised (CSLS mutual-NN), so it needs
no ground truth and carries over to diagonal unchanged -- which is the reason
the ensemble is usable here at all.
"""
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
p = f"{P}/1_coupling_inference/ensemble_final.py"
s = open(p).read()
orig = s

# ---- 1. tolerant cell-id reader -------------------------------------------
old = '''def cells_of(p, col): return pd.read_csv(p)[col].astype(str).tolist()'''
new = '''def cells_of(p, col):
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
    return None'''
assert old in s
s = s.replace(old, new, 1)

# ---- 2. use it, and only align when the plan is square --------------------
s = s.replace(
 '''c1 = cells_of(f"{comp('diffusion')}/per_cell_foscttm.csv", "cell_id")''',
 '''c1 = cells_any("diffusion", "mod1")''')
s = s.replace(
 '''c9 = cells_of(f"{comp('cca')}/per_cell_nfoscttm_metrics.csv", "cell_id")''',
 '''c9 = cells_any("cca", "mod1")''')
s = s.replace(
 '''    cmf = cells_of(f"{comp('maxfuse')}/per_cell_foscttm.csv", "cell_id")''',
 '''    cmf = cells_any("maxfuse", "mod1")''')

s = s.replace(
'''def align(D, cf, ct):
    if cf == ct: return D
    p = {c: i for i, c in enumerate(cf)}; idx = np.array([p[c] for c in ct]); return D[np.ix_(idx, idx)]
d9 = align(d9, c9, c1); n = len(c1)''',
'''SQUARE = (g1.ndim == 2 and g1.shape[0] == g1.shape[1])
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
    raise SystemExit(f"component shapes disagree: diffusion {g1.shape} vs cca {d9.shape}")''')

s = s.replace('''    rmf = np.apply_along_axis(rankdata, 1, align(mf, cmf, c1))''',
'''    mf = align(mf, cmf, c1)
    if mf.shape != g1.shape:
        raise SystemExit(f"component shapes disagree: diffusion {g1.shape} vs maxfuse {mf.shape}")
    rmf = np.apply_along_axis(rankdata, 1, mf)''')

# ---- 3. FOSCTTM only where it is defined ----------------------------------
s = s.replace(
'''def fos(D):
    dg = np.diag(D)''',
'''def fos(D):
    if D.shape[0] != D.shape[1]:
        return None            # diagonal pair: no true match to rank against
    dg = np.diag(D)''')

# ---- 4. rectangular-safe unsupervised confidence --------------------------
s = s.replace(
'''    m = fused.shape[0]
    if m > 3000:
        idx = np.random.RandomState(0).choice(m, 3000, replace=False); S = -fused[np.ix_(idx, idx)]
    else: S = -fused''',
'''    n1, n2 = fused.shape
    if max(n1, n2) > 3000:
        rs = np.random.RandomState(0)
        ri = rs.choice(n1, 3000, replace=False) if n1 > 3000 else np.arange(n1)
        ci = rs.choice(n2, 3000, replace=False) if n2 > 3000 else np.arange(n2)
        S = -fused[np.ix_(ri, ci)]
    else: S = -fused''')

# ---- 5. outputs -----------------------------------------------------------
s = s.replace(
'''pd.DataFrame({"cell_id": c1, "foscttm_sym": fosv}).to_csv(f"{OUT}/per_cell_foscttm.csv", index=False)''',
'''if fosv is not None:
    pd.DataFrame({"cell_id": c1, "foscttm_sym": fosv}).to_csv(f"{OUT}/per_cell_foscttm.csv", index=False)
else:
    pd.DataFrame({"cell_id": c1}).to_csv(f"{OUT}/cells_mod1.csv", index=False)''')
s = s.replace('''"Mean_FOSCTTM": float(np.mean(fosv))''',
              '''"Mean_FOSCTTM": (float(np.mean(fosv)) if fosv is not None else None)''')
s = s.replace('''print(f"ENSEMBLE_TUNED {DS} w={best} k={K} FOSCTTM={np.mean(fosv):.4f}", flush=True)''',
              '''print(f"ENSEMBLE_TUNED {DS} w={best} k={K} "
      f"FOSCTTM={(f'{np.mean(fosv):.4f}' if fosv is not None else 'n/a (diagonal)')}", flush=True)''')

assert s != orig
open(p, "w").write(s)
import ast; ast.parse(s)
print("ensemble_final.py: diagonal-capable, parses OK")
for k in ("cells_any", "SQUARE", "no true match", "ri = rs.choice"):
    print(f"   contains {k!r}: {k in s}")
