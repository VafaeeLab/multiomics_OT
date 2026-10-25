#!/usr/bin/env python3
"""Submit the LT upgrade grid: 11 tools x 78 datasets = 858 jobs.

One job per (tool, dataset); the four couplings run inside because they share
the baseline predictions and the label bookkeeping, so splitting on coupling
would re-read the same files four times for no gain.

Resources are 4 CPU / 32 GB. The 8 CPU / 190 GB figure used by earlier waves
was sized for holding a ~3 GB dense gamma three times over; here one gamma is
held at a time and the smoke test peaked far below that. Oversized asks are not
free - they left 32 ambiguity jobs unschedulable for hours.
"""
import argparse
import json
import os
import subprocess

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
# AnchorAE-SVM excluded: it does not perform well enough to be informative
TOOLS = ["AnchorAE-HGB", "AnchorAE-LT", "AnchorFormer", "Concerto", "Harmony",
         "ItClust", "Seurat", "SingleR", "Symphony", "scmap"]
M = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))
DS = sorted(M)

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--tools", default="")
ap.add_argument("--datasets", default="")
ap.add_argument("--walltime", default="")
a = ap.parse_args()

os.makedirs(f"{P}/logs/lt_upgrade_lm", exist_ok=True)
pbs = f"{P}/4_label_transfer/run_lt_upgrade_lm.pbs"
open(pbs, "w").write("""#!/bin/bash
#PBS -P yr31
#PBS -q normal
#PBS -l ncpus=4,mem=32GB,walltime=3:00:00,storage=gdata/yr31,wd
#PBS -j oe
#PBS -m n
cd /g/data/yr31/mp8713/project_data_integration
module load python3/3.10.4 >/dev/null 2>&1 || true
/g/data/yr31/mp8713/pytorch_env/bin/python3 -u \\
  4_label_transfer/lt_upgrade_lm.py "$TOOL" "$DS"
""")

if a.tools:
    TOOLS = [t for t in a.tools.split(",") if t in TOOLS]
if a.datasets:
    _want = set(a.datasets.split(","))
    DS = [d for d in DS if d in _want]
if a.walltime:
    import re as _re
    _pbs = open(pbs).read()
    open(pbs, "w").write(_re.sub(r"walltime=[0-9:]+",
                                 "walltime=" + a.walltime, _pbs))
pairs = [(t, ds) for t in TOOLS for ds in DS]
print(f"  {len(TOOLS)} tools x {len(DS)} datasets = {len(pairs)} jobs")
if a.dry_run:
    raise SystemExit("  dry run - nothing submitted")

n = 0
for i, (t, ds) in enumerate(pairs):
    r = subprocess.run(
        ["qsub", "-N", f"lu{i % 1000:03d}",
         "-o", f"{P}/logs/lt_upgrade_lm/{t}__{ds}.log",
         "-v", f"TOOL={t},DS={ds}", pbs],
        cwd=P, capture_output=True, text=True)
    if r.returncode == 0:
        n += 1
    else:
        print(f"  FAILED {t}/{ds}: {r.stderr.strip()[:110]}")
print(f"  submitted {n}/{len(pairs)}")
