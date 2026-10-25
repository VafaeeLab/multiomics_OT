#!/usr/bin/env python3
"""Build one pair's label-transfer bundle and both of its encoder embeddings.

One job per dataset pair does the whole chain, so nothing has to be sequenced
across jobs:

    Sample_dataset/<ds>
        -> runs_label_transfer/_lt_inputs/<ds>/            (lt_prep_inputs.py)
        -> runs_label_transfer/_lt_embeddings/attention/<ds>/embeddings.csv
        -> runs_label_transfer/_lt_embeddings/transformer/<ds>/embeddings.csv

Encoder settings are FIXED at the published defaults of the architecture rather
than inherited from the data-integration sweep, which tuned epochs per dataset.
Inheriting those would reintroduce the dependency this rebuild exists to
remove, and a single fixed setting is also the honest thing to describe in the
methods.

    usage: lt_build_embed.py <dataset> [--force]
"""
import argparse
import os
import subprocess
import sys
import time

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
LTS = f"{P}/4_label_transfer"
EPOCHS = 120                      # the architecture's own default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--force", action="store_true")
    # Which encoders to build. Both are built by default, because the full
    # benchmark needs both. Pass --arch attention to build only the AnchorAE
    # encoder, which is all the AnchorAE-LT and AnchorAE-HGB models read.
    ap.add_argument("--arch", default="AnchorAE,AnchorFormer")
    a = ap.parse_args()
    ds = a.dataset
    os.chdir(P)

    t0 = time.time()
    cmd = [PY, f"{LTS}/lt_prep_inputs.py", ds] + (["--force"] if a.force else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-1500:])
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-1500:])
        raise SystemExit(f"bundle failed for {ds}")
    print(f"[bundle] {ds} {time.time() - t0:.0f}s", flush=True)

    for arch in [x.strip() for x in a.arch.split(",") if x.strip()]:
        out = f"{LT_ROOT}/_lt_embeddings/{arch}/{ds}/embeddings.csv"
        if os.path.exists(out) and not a.force:
            print(f"[{arch}] {ds} already present, skipping", flush=True)
            continue
        t1 = time.time()
        r = subprocess.run(
            [PY, f"{LTS}/lt_embedder.py", "--dataset", ds, "--arch", arch,
             "--epochs", str(EPOCHS)],
            capture_output=True, text=True)
        sys.stdout.write(r.stdout[-1500:])
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-2000:])
            raise SystemExit(f"{arch} failed for {ds}")
        if not os.path.exists(out):
            raise SystemExit(f"{arch} wrote nothing for {ds}")
        print(f"[{arch}] {ds} {time.time() - t1:.0f}s", flush=True)

    print(f"LT_EMBED_DONE {ds} total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
