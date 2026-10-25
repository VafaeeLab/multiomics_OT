#!/usr/bin/env python3
"""Submit 11 DI tools x 78 pairs = 858 baseline jobs.

Each tool job is chained afterok to its own pair's prep job, using the ids
captured at submit time in logs/prep78/jobids.json. Name-based dependency does
not work here: qstat truncates to 10 characters, and ds[:9] is "MULTIOME_" for
eleven pairs and "fallopia" for three.

WNN is submitted for all 78 as asked; on the 49 diagonal pairs run_di_tool.py
records NOT_APPLICABLE instead of an embedding, because WNN weights two assays
measured in the same cells and a diagonal pair has none. That gives 29 real WNN
results, not 78.

Existing baselines are cleared first: they were produced from the prepared
inputs built 07-27..08-14, and run_di_tool skips when embeddings.csv exists, so
leaving them would silently keep results from the superseded matrices.
"""
import json, os, shutil, subprocess, sys

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
MAP = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))
DS = sorted(MAP)
TOOLS = ["AnchorAE", "AnchorFormer", "AnchorMix", "seurat", "harmony",
         "symphony", "scvi", "scanorama", "fastmnn", "concerto", "wnn"]
# heavier tools get more walltime; the deep models and Concerto are the slow ones
LONG = {"AnchorAE", "AnchorFormer", "AnchorMix", "concerto", "scvi"}
DRY = "--dry-run" in sys.argv

# --- clear stale baselines ----------------------------------------------------
n_del = 0
for t in TOOLS:
    for reg in ("coassay", "closedset", "openset"):
        for ds in DS:
            d = f"{DI_ROOT}/{t}/{reg}/baseline/{ds}"
            if os.path.isdir(d) and not DRY:
                shutil.rmtree(d); n_del += 1
            elif os.path.isdir(d):
                n_del += 1
print(f"{'would clear' if DRY else 'cleared'} {n_del} stale baseline directories")

os.makedirs(f"{P}/logs/di78", exist_ok=True)
for tag, wt, mem in (("std", "6:00:00", "190GB"), ("long", "20:00:00", "190GB")):
    open(f"{P}/logs/di78/di_{tag}.pbs", "w").write(f"""#!/bin/bash
#PBS -P yr31
#PBS -q normal
#PBS -l ncpus=12,mem={mem},walltime={wt},storage=gdata/yr31,wd
#PBS -j oe
#PBS -m n
cd {P}
module load R/4.3.1 >/dev/null 2>&1 || true
module load python3/3.10.4 >/dev/null 2>&1 || true
export R_LIBS_USER={P}/R_analysis/R_libs_4.3
/g/data/yr31/mp8713/pytorch_env/bin/python3 -u \\
  3_data_integration/run_di_tool.py "$DS" "$TOOL" --force
""")

prep_ids = {}
f = f"{P}/logs/prep78/jobids.json"
if os.path.exists(f):
    prep_ids = json.load(open(f))
    live = subprocess.run(["qstat", "-w", "-u", "mp8713"],
                          capture_output=True, text=True).stdout
    prep_ids = {k: v for k, v in prep_ids.items() if v in live}
print(f"{len(prep_ids)} prep jobs still live -- their tools will be chained")

n = 0
for i, ds in enumerate(DS):
    for t in TOOLS:
        tag = "long" if t in LONG else "std"
        cmd = ["qsub", "-N", f"{t[:4]}{i:02d}",
               "-o", f"{P}/logs/di78/{ds}_{t}.log",
               "-v", f"DS={ds},TOOL={t}"]
        if ds in prep_ids:
            cmd += ["-W", f"depend=afterok:{prep_ids[ds]}"]
        if DRY:
            n += 1; continue
        r = subprocess.run(cmd + [f"{P}/logs/di78/di_{tag}.pbs"],
                           cwd=P, capture_output=True, text=True)
        if r.returncode == 0:
            n += 1
        else:
            print(f"  FAILED {ds}/{t}: {r.stderr.strip()[:110]}")
print(f"{'would submit' if DRY else 'submitted'} {n} jobs "
      f"({len(TOOLS)} tools x {len(DS)} pairs)")
