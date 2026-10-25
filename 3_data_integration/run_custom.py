#!/usr/bin/env python3
"""Run one CUSTOM integration model on one dataset pair and normalise its output.

The four custom models were each written against a different tree and each writes
somewhere different; this puts them all through one input resolver and lands every
result at the benchmark's canonical location:

    runs_data_integration/<tool>/<regime>/baseline/<dataset>/embeddings.csv

with the same columns the external tools emit (cell_id, modality, cell_type,
batch, batch_modality, emb_*), so the sweep scripts can score them unchanged.

  attention    integration_attn_embedder_v3.py --arch attention
  transformer  integration_attn_embedder_v3.py --arch transformer
  anchormix    3_data_integration/model20_v2_1.py  (= model22)
  attnv4       attn_v4.py

usage: run_custom.py <dataset> <attention|transformer|anchormix|attnv4> [--epochs N]
"""
import os, sys, glob, shutil, subprocess, argparse
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
ROOT = P
PY = os.environ.get("PY", sys.executable)


def resolve_prep(ds):
    """Diagonal tree first, then runs_integration.

    humanRNAmC and mouseRNAmC exist in BOTH trees, so checking runs_integration
    first labelled these diagonal RNA-methylome pairs "coassay" and sent their
    results to the wrong regime directory (where a stale copy then caused an
    immediate skip). Membership of Results/prepared_input is authoritative for
    the regime, so it is checked first.
    """
    for r in ("closedset", "openset"):
        d = f"{ROOT}/Results/prepared_input/{r}/{ds}"
        if os.path.exists(f"{d}/combined_log.mtx"):
            return d, r
    c = f"{ROOT}/runs_integration/benchmarking_{ds}/prepared_input"
    if os.path.exists(f"{c}/combined_log.mtx"):
        return c, "coassay"
    base, reg = ds, None
    if ds.startswith("closed_"):
        base, reg = ds[len("closed_"):], "closedset"
    elif ds.startswith("open_"):
        base, reg = ds[len("open_"):], "openset"
    if reg:
        d = f"{ROOT}/Results/prepared_input/{reg}/{base}"
        if os.path.exists(f"{d}/combined_log.mtx"):
            return d, reg
    for r in ("closedset", "openset"):
        d = f"{ROOT}/Results/prepared_input/{r}/{ds}"
        if os.path.exists(f"{d}/combined_log.mtx"):
            return d, r
    raise SystemExit(f"no prepared input for {ds}")


def normalise(src_csv, prep, out_dir):
    """Attach the benchmark's metadata columns and write the canonical file."""
    d = pd.read_csv(src_csv)
    emb = [c for c in d.columns if c.startswith("emb_")]
    if not emb:                                   # rename bare numeric columns
        keep = [c for c in d.columns if c not in ("cell_id", "modality", "cell_type",
                                                  "batch", "batch_modality", "dataset_id")]
        d = d.rename(columns={c: f"emb_{i}" for i, c in enumerate(keep)})
        emb = [f"emb_{i}" for i in range(len(keep))]
    o = pd.read_csv(f"{prep}/combined_log_obs.csv")
    meta = o[["cell_id", "modality", "cell_type", "batch"]].copy()
    if "cell_id" in d.columns:
        # the unique key is (cell_id, modality), never cell_id alone: in co-assay
        # the same cell appears once per modality, so a dict keyed on the id holds
        # only one of its two rows and both lookups return it, silently collapsing
        # two genuinely different per-modality positions into one duplicated row
        if "modality" in d.columns:
            src_key = list(zip(d.cell_id.astype(str), d.modality.astype(str)))
            want = list(zip(meta.cell_id.astype(str), meta.modality.astype(str)))
        else:
            src_key = list(d.cell_id.astype(str))
            want = list(meta.cell_id.astype(str))
            if len(set(src_key)) != len(src_key):
                raise SystemExit(
                    f"{src_csv} repeats cell_id but carries no modality column, "
                    f"so rows cannot be matched unambiguously")
        pos = {k: j for j, k in enumerate(src_key)}
        missing = [k for k in want if k not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} cells missing from {src_csv}")
        d = d.iloc[[pos[k] for k in want]].reset_index(drop=True)
    elif len(d) != len(o):
        raise SystemExit(f"row mismatch {len(d)} vs {len(o)}")
    out = pd.concat([meta.reset_index(drop=True), d[emb].reset_index(drop=True)], axis=1)
    out["batch_modality"] = out.batch.astype(str) + "__" + out.modality.astype(str)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(f"{out_dir}/embeddings.csv", index=False)
    print(f"WROTE {out_dir}/embeddings.csv  rows={len(out)} dims={len(emb)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("model", choices=["AnchorAE", "AnchorFormer", "AnchorMix", "attnv4"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if embeddings.csv is already present")
    a = ap.parse_args()
    ds, model = a.dataset, a.model
    prep, reg = resolve_prep(ds)
    out_dir = f"{DI_ROOT}/{model}/{reg}/baseline/{ds}"
    if os.path.exists(f"{out_dir}/embeddings.csv") and not a.force:
        print(f"SKIP {ds}/{model} already present", flush=True); return
    os.chdir(ROOT)

    if model in ("AnchorAE", "AnchorFormer"):
        subprocess.run([PY, f"{ROOT}/3_data_integration/integration_attn_embedder_v3.py",
                        "--dataset", ds, "--arch", model, "--epochs", str(a.epochs)], check=True)
        tag = {"AnchorAE": "AnchorAE", "AnchorFormer": "AnchorFormer"}[model]
        src = f"{DI_ROOT}/{tag}/{ds}/embeddings.csv"
        if not os.path.exists(src):
            raise SystemExit(f"{model} wrote nothing at {src}")
        normalise(src, prep, out_dir)

    elif model == "AnchorMix":
        work = f"{DI_ROOT}/anchormix/_work/{ds}"
        os.makedirs(work, exist_ok=True)
        pre = f"{work}/{ds}"
        subprocess.run([PY, f"{ROOT}/3_data_integration/model20_v2_1.py",
                        "--dataset_id", ds, "--out_prefix", pre,
                        "--x1", f"{prep}/mod1_log.mtx", "--x2", f"{prep}/mod2_log.mtx",
                        "--barcodes1", f"{prep}/mod1_log_barcodes.tsv",
                        "--barcodes2", f"{prep}/mod2_log_barcodes.tsv",
                        "--labels1", f"{prep}/mod1_log_obs.csv",
                        "--labels2", f"{prep}/mod2_log_obs.csv",
                        "--label_id_col", "cell_id", "--batch_col", "batch",
                        "--max_epochs", "200", "--device", "auto",
                        "--w_dynamic_gamma", "0"], check=True)
        normalise(f"{pre}_embeddings.csv", prep, out_dir)

    elif model == "attnv4":
        subprocess.run([PY, f"{ROOT}/attn_v4.py", "--dataset", ds,
                        "--epochs", str(a.epochs)], check=True)
        cand = glob.glob(f"{DI_ROOT}/attnv4/*/baseline/{ds}/embeddings.csv")
        if not cand:
            raise SystemExit("attnv4 wrote nothing")
        if os.path.dirname(cand[0]) != out_dir:
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(cand[0], f"{out_dir}/embeddings.csv")
        print(f"WROTE {out_dir}/embeddings.csv", flush=True)

    print(f"CUSTOM_DONE {ds} {model}", flush=True)


if __name__ == "__main__":
    main()
