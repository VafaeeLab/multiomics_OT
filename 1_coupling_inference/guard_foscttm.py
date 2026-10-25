import os
#!/usr/bin/env python3
"""Make the FOSCTTM machinery skip, rather than crash, on a diagonal pair.

FOSCTTM asks how many cells sit closer to a cell than its TRUE PARTNER does.
A diagonal pair has no partner, and its coupling is rectangular (n1 x n2), so
every diagonal-based expression here breaks:

    maxfuse   (3615,2575) vs (2575,1)
    cca       (1567,2575) vs  (527,1)

np.diag on a rectangular matrix returns min(n1,n2) values, which then fail to
broadcast. The fix is to return NaN metrics for a rectangular plan and skip the
per-cell file, while still writing the coupling itself -- the coupling is the
deliverable; FOSCTTM is only its co-assay evaluation.
"""
import re

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
CS = f"{P}/1_coupling_inference"

# ------------------------------------------------- dv_common.score_dense ----
p = f"{CS}/dv_common.py"
s = open(p).read()
old = '''def score_dense(D, label=""):
    D = np.asarray(D, np.float64); n = D.shape[0]'''
new = '''def score_dense(D, label=""):
    D = np.asarray(D, np.float64); n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        # Rectangular plan => diagonal pair => no true match to rank against.
        _nan = float("nan")
        out = {"label": label, "n": n, "foscttm_midrank": _nan,
               "foscttm_strict": _nan, "mrr": _nan, "median_rank": _nan,
               "hungarian": _nan, "paired": False}
        out.update({f"recall_at_{k}": _nan for k in (1, 5, 10, 50)})
        return out'''
if "Rectangular plan" not in s:
    assert old in s, "score_dense signature not found"
    s = s.replace(old, new, 1)
    print("dv_common.score_dense: returns NaN metrics for a rectangular plan")

# ------------------------------------------------ dv_common.write_outputs ---
old_w = '''    md = PSF.metrics_from_matrix(np.asarray(eval_D, np.float64))
    pd.DataFrame({"cell_id": list(map(str, cells)),
                  "foscttm_sym": md["foscttm_sym"], "nfoscttm_sym": md["nfoscttm_sym"]}).to_csv(
        outdir / "per_cell_nfoscttm_metrics.csv", index=False)'''
new_w = '''    _sq = (np.asarray(eval_D).ndim == 2 and
           np.asarray(eval_D).shape[0] == np.asarray(eval_D).shape[1])
    md = PSF.metrics_from_matrix(np.asarray(eval_D, np.float64)) if _sq else None
    if _sq:
        pd.DataFrame({"cell_id": list(map(str, cells)),
                      "foscttm_sym": md["foscttm_sym"], "nfoscttm_sym": md["nfoscttm_sym"]}).to_csv(
            outdir / "per_cell_nfoscttm_metrics.csv", index=False)
    else:
        # diagonal pair: record the row labels so the coupling stays interpretable
        pd.DataFrame({"cell_id": list(map(str, cells))}).to_csv(
            outdir / "cells_mod1.csv", index=False)'''
if "_sq = (np.asarray(eval_D)" not in s:
    assert old_w in s, "write_outputs body not found"
    s = s.replace(old_w, new_w, 1)
    s = s.replace('"Mean_nFOSCTTM_Sym": float(np.mean(md["nfoscttm_sym"])),',
                  '"Mean_nFOSCTTM_Sym": (float(np.mean(md["nfoscttm_sym"])) if md else float("nan")),')
    print("dv_common.write_outputs: skips the per-cell file for a rectangular plan")
open(p, "w").write(s)

# --------------------------------------------------------- maxfuse_run -----
p = f"{CS}/maxfuse_run.py"
t = open(p).read()
old_m = '''fos = fos_pc(D)'''
new_m = '''# FOSCTTM needs a square plan with the true match on the diagonal
_square = (D.shape[0] == D.shape[1])
fos = fos_pc(D) if (PAIRED and _square) else None'''
if "_square = (D.shape[0]" not in t:
    assert old_m in t, "fos_pc call not found"
    t = t.replace(old_m, new_m, 1)
    t = re.sub(
        r'pd\.DataFrame\(\{"cell_id": list\(map\(str, cells\)\), "foscttm_sym": fos\}\)\.to_csv\(f"\{OUT\}/per_cell_foscttm\.csv", index=False\)',
        'if fos is not None:\n'
        '    pd.DataFrame({"cell_id": list(map(str, cells)), "foscttm_sym": fos}).to_csv(f"{OUT}/per_cell_foscttm.csv", index=False)\n'
        'else:\n'
        '    pd.DataFrame({"cell_id": list(map(str, cells1))}).to_csv(f"{OUT}/cells_mod1.csv", index=False)\n'
        '    pd.DataFrame({"cell_id": list(map(str, cells2))}).to_csv(f"{OUT}/cells_mod2.csv", index=False)',
        t)
    t = t.replace('"Mean_FOSCTTM": float(np.mean(fos))',
                  '"Mean_FOSCTTM": (float(np.mean(fos)) if fos is not None else None)')
    open(p, "w").write(t)
    print("maxfuse_run: FOSCTTM guarded; cell lists written instead when unpaired")

import ast
print()
for fn in ("dv_common.py", "maxfuse_run.py", "cca_ot.py"):
    try:
        ast.parse(open(f"{CS}/{fn}").read()); print(f"  {fn:<16} parses OK")
    except SyntaxError as e:
        print(f"  {fn:<16} SYNTAX ERROR line {e.lineno}: {e.msg}")
