#!/usr/bin/env python3
"""Build the label-transfer input bundle for one pair, from the normalised data.

Label transfer must not read anything produced by the data-integration
benchmark. This script therefore builds its OWN bundle, under

    runs_label_transfer/_lt_inputs/<dataset>/

from data, which holds the normalised matrices and
(obs_normalized.csv, for the labels). It does not touch runs_data_integration,
Results/prepared_input or runs_integration at any point.

prepare_integration_input.py is reused because it is a neutral utility that
intersects the shared features and writes the combined matrix; reusing a
function is not the same as reading another benchmark's results. The bundle it
writes here is owned by label transfer and is written nowhere else.

The regime is taken from foscttm_input_map.json, which is the roster, not a
result.

    usage: lt_prep_inputs.py <dataset> [<dataset> ...] [--force]
"""
import json
import os
import subprocess
import sys

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
PY = os.environ.get("PY", sys.executable)
OUT_ROOT = f"{LT_ROOT}/_lt_inputs"
MAP = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))


def build(ds, force=False):
    v = MAP[ds]
    src = f"{P}/data/{ds}"
    out = f"{OUT_ROOT}/{ds}"
    if os.path.exists(f"{out}/combined_log.mtx") and not force:
        return "skip", out
    os.makedirs(out, exist_ok=True)

    # Labels come from the obs that matches the NORMALISED matrices. The
    # count-level obs.csv lists pre-QC cells and would mislabel the rows.
    obs = pd.read_csv(f"{P}/data/{ds}/obs_normalized.csv")
    lab = {}
    for m in ("mod1", "mod2"):
        sub = obs[obs.modality == m][["cell_id", "cell_type"]].copy()
        sub["batch"] = m
        f = f"{out}/_labels_{m}.csv"
        sub.to_csv(f, index=False)
        lab[m] = f

    cmd = [PY, f"{P}/3_data_integration/prepare_integration_input.py",
           "--dataset_id", ds,
           "--x1", f"{src}/mod1_normalized.mtx",
           "--x2", f"{src}/mod2_normalized.mtx",
           "--barcodes1", f"{src}/mod1_barcodes_normalized.tsv",
           "--barcodes2", f"{src}/mod2_barcodes_normalized.tsv",
           "--features1", f"{src}/mod1_features_normalized.tsv",
           "--features2", f"{src}/mod2_features_normalized.tsv",
           "--labels1", lab["mod1"], "--labels2", lab["mod2"],
           # the normalised matrices are stored genes x cells
           "--orientation1", "feature_by_cell",
           "--orientation2", "feature_by_cell",
           "--feature_mode", "shared",
           "--outdir", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"FAILED: {r.stderr.strip()[-300:]}", out
    if not os.path.exists(f"{out}/combined_log.mtx"):
        return "FAILED: no combined_log.mtx written", out
    return "built", out


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    dss = args or sorted(MAP)
    n_ok = n_skip = n_bad = 0
    for ds in dss:
        if ds not in MAP:
            print(f"  {ds}: not in the roster", flush=True)
            n_bad += 1
            continue
        st, out = build(ds, force)
        print(f"  {ds:26s} {st}", flush=True)
        if st == "built":
            n_ok += 1
        elif st == "skip":
            n_skip += 1
        else:
            n_bad += 1
    print(f"built {n_ok}, skipped {n_skip}, failed {n_bad}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
