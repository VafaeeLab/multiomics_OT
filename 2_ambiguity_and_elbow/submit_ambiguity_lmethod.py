#!/usr/bin/env python3
"""Submit 4 couplings x 78 pairs = 312 L-method / exp(10a) ambiguity jobs.

Mirrors submit_ambiguity_all.py. One job per (dataset, coupling), each running
pseudo_label and label_margin so the two share the single load of a gamma that
reaches ~3 GB dense.

Only two detectors run here rather than three, so there is no SVD embedding to
build and no local_consistency pass; pseudo_label's two Leiden runs are still
the slow step, so the walltime is kept at the production value rather than
trimmed on an assumption.

Job names carry an index, not the dataset name: qstat truncates to 10
characters and ds[:9] collides across the 11 MULTIOME_* and 3 fallopian* pairs,
which is how an earlier wave double-submitted.

  --dry-run   count the jobs without submitting
  --k         exp() constant, default 10
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
MAP = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))
COUPLINGS = ["diffusion", "cca_ot", "maxfuse", "ensemble"]
DS = sorted(MAP)

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--k", type=float, default=10.0)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

os.makedirs(f"{P}/logs/ambiguity_lmethod", exist_ok=True)
pbs = f"{P}/2_ambiguity_and_elbow/run_amb_lmethod.pbs"
open(pbs, "w").write("""#!/bin/bash
#PBS -P yr31
#PBS -q normal
#PBS -l ncpus=8,mem=190GB,walltime=8:00:00,storage=gdata/yr31,wd
#PBS -j oe
#PBS -m n
cd /g/data/yr31/mp8713/project_data_integration
module load python3/3.10.4 >/dev/null 2>&1 || true
/g/data/yr31/mp8713/pytorch_env/bin/python3 -u \\
  2_ambiguity_and_elbow/run_ambiguity_lmethod.py \\
  --dataset "$DS" --coupling "$COUP" --k "$KEXP" $EXTRA
""")

q = subprocess.run(["qstat", "-w", "-u", "mp8713"], capture_output=True,
                   text=True).stdout
print(f"{sum(1 for l in q.splitlines() if l[:1].isdigit())} jobs already queued")

n = 0
for i, ds in enumerate(DS):
    for c in COUPLINGS:
        if a.dry_run:
            n += 1
            continue
        r = subprocess.run(
            ["qsub", "-N", f"lm{c[:3]}{i:02d}",
             "-o", f"{P}/logs/ambiguity_lmethod/{ds}_{c}.log",
             "-v", f"DS={ds},COUP={c},KEXP={a.k:g},"
                   f"EXTRA={'--force' if a.force else ''}",
             pbs],
            cwd=P, capture_output=True, text=True)
        if r.returncode == 0:
            n += 1
        else:
            print(f"  FAILED {ds}/{c}: {r.stderr.strip()[:110]}")
print(f"{'would submit' if a.dry_run else 'submitted'} {n} jobs "
      f"({len(COUPLINGS)} couplings x {len(DS)} pairs)")
