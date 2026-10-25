#!/usr/bin/env python3
"""Run ONE data-integration tool on ONE pair, baseline only, tuned parameters.

    runs_data_integration/<tool>/<regime>/baseline/<dataset>/embeddings.csv

with the benchmark's columns (cell_id, modality, cell_type, batch,
batch_modality, emb_*), identical for all 11 tools so downstream scoring reads
one format.

    AnchorAE      attention     integration_attn_embedder_v3.py --arch attention
    Anchorformer  transformer   integration_attn_embedder_v3.py --arch transformer
    AnchorMix     anchormix     3_data_integration/model20_v2_1.py
    Seurat        seurat        run_seurat_v5_integration.R
    Harmony       harmony       run_harmony_integration.R
    Symphony      symphony      run_symphony_integration.R
    scVI          scvi          run_scvi_integration.py
    Scanorama     scanorama     run_scanorama_integration.py
    FastMNN       fastmnn       run_fastmnn_integration.R
    Concerto      concerto      run_concerto_integration.py
    WNN           wnn           run_wnn_integration.R      co-assay only

Parameters come from di_params.resolve, which reproduces the benchmark's
emit_dataset_context policy. 3_data_integration/run_one_tool.sh is not used:
it hardcodes the co-assay prep path so every diagonal pair fails, and its
dispatch is `if seurat ... else run_scvi_integration.py`, which silently runs
scVI for any other tool name.

WNN weights two assays measured in the SAME cells, so it is only defined for
co-assay pairs. On a diagonal pair it exits with a recorded NOT_APPLICABLE
rather than producing an embedding that would look like a result.

usage: run_di_tool.py <dataset> <tool> [--force]
"""
import argparse, json, os, shutil, subprocess, sys, time

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
sys.path.insert(0, f"{P}/3_data_integration")
from run_custom import resolve_prep, normalise
from di_params import resolve

S = f"{P}/3_data_integration"
PY = os.environ.get("PY", sys.executable)
RSCRIPT = "/apps/R/4.3.1/bin/Rscript"      # R/4.3.1 matches R_libs_4.3
CUSTOM = {"AnchorAE", "AnchorFormer", "AnchorMix"}
EXTERNAL = {"seurat", "harmony", "symphony", "scvi", "scanorama", "fastmnn",
            "concerto", "wnn"}
TOOLS = sorted(CUSTOM | EXTERNAL)


def build_cmd(tool, q, prep, work, ds):
    """The tool's own CLI, filled from the resolved tuned parameters."""
    if tool == "seurat":
        # k_weight cannot exceed the cells available to weight an anchor and
        # variable_features cannot exceed the shared space; both are hard
        # errors in Seurat rather than a silent fallback
        kw = min(q["SEURAT_K_WEIGHT"], max(5, min(q["n1"], q["n2"]) // 2))
        vf = min(q["SEURAT_VARIABLE_FEATURES"], q["p_shared"])
        return [RSCRIPT, f"{S}/run_seurat_v5_integration.R",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--dims", str(min(q["SEURAT_DIMS"], q["emb_dim"])),
                "--variable_features", str(vf),
                "--k_anchor", str(q["SEURAT_K_ANCHOR"]),
                "--k_filter", str(q["SEURAT_K_FILTER"]),
                "--k_score", str(q["SEURAT_K_SCORE"]),
                "--k_weight", str(kw), "--seed", "42"]
    if tool == "harmony":
        return [RSCRIPT, f"{S}/run_harmony_integration.R",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--pca_dim", str(q["HARMONY_PCA_DIM"]),
                "--emb_dim", str(q["HARMONY_EMB_DIM"]),
                "--theta", str(q["HARMONY_THETA"]),
                "--lambda", str(q["HARMONY_LAMBDA"]),
                "--sigma", str(q["HARMONY_SIGMA"]),
                "--max_iter", str(q["HARMONY_MAX_ITER"]), "--seed", "42"]
    if tool == "symphony":
        # this script exposes dims / n_variable_genes / k_clusters only -- the
        # harmony theta/lambda/sigma in the hyperparameter table are internal to
        # its Harmony step and are not settable here, so they are not passed
        return [RSCRIPT, f"{S}/run_symphony_integration.R",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--dims", str(q["SYMPHONY_NPCS"]),
                "--n_variable_genes", str(q["SYMPHONY_NVG"]), "--seed", "42"]
    if tool == "fastmnn":
        return [RSCRIPT, f"{S}/run_fastmnn_integration.R",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--d", str(q["FASTMNN_D"]), "--k", str(q["FASTMNN_K"]),
                "--seed", "42"]
    if tool == "wnn":
        return [RSCRIPT, f"{S}/run_wnn_integration.R",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--dims", str(q["WNN_DIMS"]),
                "--mod2_norm", q["WNN_MOD2_NORM"]]
    if tool == "scvi":
        return [PY, f"{S}/run_scvi_integration.py",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--n_latent", str(q["SCVI_N_LATENT"]),
                "--n_hidden", str(q["SCVI_N_HIDDEN"]),
                "--n_layers", str(q["SCVI_N_LAYERS"]),
                "--dropout", str(q["SCVI_DROPOUT"]),
                "--max_epochs", str(q["SCVI_MAX_EPOCHS"]),
                "--batch_size", str(q["SCVI_BATCH_SIZE"])]
    if tool == "scanorama":
        return [PY, f"{S}/run_scanorama_integration.py",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--dimred", str(q["SCANORAMA_DIMRED"]),
                "--knn", str(q["SCANORAMA_KNN"]),
                "--sigma", str(q["SCANORAMA_SIGMA"]),
                "--alpha", str(q["SCANORAMA_ALPHA"]),
                "--batch_size", str(q["SCANORAMA_BATCH_SIZE"])]
    if tool == "concerto":
        return [PY, f"{S}/run_concerto_integration.py",
                "--prepared_dir", prep, "--outdir", work, "--dataset_id", ds,
                "--embedding_dim", str(q["CONCERTO_EMBEDDING_DIM"]),
                "--pretrain_epochs", str(q["CONCERTO_PRETRAIN_EPOCHS"]),
                "--fine_tune_epochs", str(q["CONCERTO_FINETUNE_EPOCHS"]),
                "--batch_size", str(q["CONCERTO_BATCH_SIZE"]),
                "--learning_rate", str(q["CONCERTO_LR"])]
    raise SystemExit(f"no command for {tool}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("tool", choices=TOOLS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args()
    ds, tool = a.dataset, a.tool

    prep, reg = resolve_prep(ds)
    out_dir = f"{DI_ROOT}/{tool}/{reg}/baseline/{ds}"
    if os.path.exists(f"{out_dir}/embeddings.csv") and not a.force:
        print(f"SKIP {ds}/{tool} already present", flush=True)
        return
    os.chdir(P)

    q = resolve(ds)
    if tool == "wnn" and q["regime"] != "coassay":
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"dataset": ds, "tool": "wnn", "status": "NOT_APPLICABLE",
                   "reason": "WNN weights two assays profiled in the same "
                             "cells; a diagonal pair has no shared cells"},
                  open(f"{out_dir}/status.json", "w"), indent=2)
        print(f"NOT_APPLICABLE {ds} wnn (diagonal pair)", flush=True)
        return

    print(f"[{ds}] tool={tool} regime={reg} n1={q['n1']} n2={q['n2']} "
          f"P={q['p_shared']} emb_dim={q['emb_dim']} "
          f"override={q['has_override']}", flush=True)

    if tool in CUSTOM:
        cmd = [PY, f"{P}/3_data_integration/run_custom.py",
               ds, tool, "--epochs", str(a.epochs)]
        if a.force:
            cmd.append("--force")      # otherwise run_custom skips and --force is a lie
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"{tool} failed on {ds}")
        print(f"DI_DONE {ds} {tool}", flush=True)
        return

    work = f"{DI_ROOT}/{tool}/_work/{ds}"
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, R_LIBS_USER=f"{P}/R_analysis/R_libs_4.3")
    cmd = build_cmd(tool, q, prep, work, ds)
    print("RUN " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:])
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-4000:])
        raise SystemExit(f"{tool} failed on {ds} (rc={r.returncode})")

    src = f"{work}/embeddings.csv"
    if not os.path.exists(src):
        cand = [f for f in os.listdir(work)
                if f.endswith(".csv") and "embed" in f]
        if not cand:
            raise SystemExit(f"{tool} wrote no embedding csv in {work}: "
                             f"{os.listdir(work)}")
        src = f"{work}/{cand[0]}"
    normalise(src, prep, out_dir)

    if tool == "wnn":
        # normalise() writes only embeddings.csv; the graph, the modality
        # weights and the two reductions live in _work and are what a later
        # coupling-based upgrade needs, so they are carried across too.
        kept = []
        for f in sorted(os.listdir(work)):
            if f.startswith("wnn_"):
                shutil.copy(f"{work}/{f}", f"{out_dir}/{f}")
                kept.append(f)
        print(f"  kept WNN artefacts: {kept}", flush=True)

    json.dump({"dataset": ds, "tool": tool, "regime": reg, "status": "ok",
               "seconds": round(time.time() - t0, 1),
               "params": {k: v for k, v in q.items() if k.isupper()}},
              open(f"{out_dir}/run_params.json", "w"), indent=2)
    print(f"DI_DONE {ds} {tool}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
