#!/usr/bin/env python3
"""Submit 11 label-transfer tools x 78 pairs = 858 baseline jobs.

The four custom tools read a DI embedding, so each is chained afterok to that
pair's own DI job when one is still queued. The DI jobs were named
<tool[:4]><index> with index = position in sorted(datasets), so the id can be
recovered from qstat -w by name -- name matching on the DATASET would not work,
since qstat truncates to 10 characters and ds[:9] collides across the eleven
MULTIOME_* pairs and the three fallopian* pairs.

    AnchorAE-LT / AnchorAE-HGB / AnchorAE-SVM   wait on that pair's attention job
    AnchorFormer                                wait on its transformer job
    the 7 external tools                        no DI dependency, run immediately

Existing LT results for these tools are cleared first, because run_lt_tool.py
skips when metrics.csv exists and anything already there predates the rebuilt
normalised inputs.
"""
import json, os, re, shutil, subprocess, sys

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
CUSTOM = ["AnchorAE-LT", "AnchorFormer", "AnchorAE-HGB", "AnchorAE-SVM"]
EXTERNAL = ["Seurat", "Symphony", "SingleR", "scmap", "ItClust", "Concerto",
            "Harmony"]
TOOLS = CUSTOM + EXTERNAL
DI_DEP = {"AnchorAE-LT": "atte", "AnchorAE-HGB": "atte", "AnchorAE-SVM": "atte",
          "AnchorFormer": "tran"}
# SVM and the deep external tools are the slow ones
LONG = {"AnchorAE-SVM", "Concerto", "ItClust", "Seurat"}
DRY = "--dry-run" in sys.argv

n_del = 0
for t in TOOLS:
    for reg in ("coassay", "closedset", "openset"):
        for ds in DS:
            d = f"{LT_ROOT}/{t}/{reg}/baseline/{ds}"
            if os.path.isdir(d):
                n_del += 1
                if not DRY:
                    shutil.rmtree(d)
print(f"{'would clear' if DRY else 'cleared'} {n_del} stale LT directories")

os.makedirs(f"{P}/logs/lt78", exist_ok=True)
for tag, wt in (("std", "6:00:00"), ("long", "20:00:00")):
    open(f"{P}/logs/lt78/lt_{tag}.pbs", "w").write(f"""#!/bin/bash
#PBS -P yr31
#PBS -q normal
#PBS -l ncpus=12,mem=190GB,walltime={wt},storage=gdata/yr31,wd
#PBS -j oe
#PBS -m n
cd {P}
module load R/4.3.1 >/dev/null 2>&1 || true
module load python3/3.10.4 >/dev/null 2>&1 || true
export R_LIBS_USER={P}/R_analysis/R_libs_4.3
export PROJECT_ROOT={P}
/g/data/yr31/mp8713/pytorch_env/bin/python3 -u \\
  4_label_transfer/run_lt_tool.py "$DS" "$TOOL" --force
""")

# live DI job ids, keyed by (prefix, dataset index)
q = subprocess.run(["qstat", "-w", "-u", "mp8713"],
                   capture_output=True, text=True).stdout
live = {}
for line in q.splitlines():
    mm = re.match(r"^(\d+)\.\S+\s+\S+\s+\S+\s+(atte|tran)(\d{2})\b", line)
    if mm:
        live[(mm.group(2), int(mm.group(3)))] = mm.group(1)
print(f"{len(live)} DI jobs still live -- dependent LT jobs will chain to them")

n = 0
for i, ds in enumerate(DS):
    for t in TOOLS:
        tag = "long" if t in LONG else "std"
        cmd = ["qsub", "-N", f"L{t[:6].replace('-','')}{i:02d}"[:14],
               "-o", f"{P}/logs/lt78/{ds}_{t}.log",
               "-v", f"DS={ds},TOOL={t}"]
        dep = live.get((DI_DEP[t], i)) if t in DI_DEP else None
        if dep:
            cmd += ["-W", f"depend=afterok:{dep}"]
        if DRY:
            n += 1
            continue
        r = subprocess.run(cmd + [f"{P}/logs/lt78/lt_{tag}.pbs"],
                           cwd=P, capture_output=True, text=True)
        if r.returncode == 0:
            n += 1
        else:
            print(f"  FAILED {ds}/{t}: {r.stderr.strip()[:110]}")
print(f"{'would submit' if DRY else 'submitted'} {n} jobs "
      f"({len(TOOLS)} tools x {len(DS)} pairs)")
