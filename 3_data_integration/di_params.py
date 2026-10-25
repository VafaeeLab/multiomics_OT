#!/usr/bin/env python3
"""Resolve the tuned baseline parameters for one dataset, for every DI tool.

This reproduces 3_data_integration/emit_dataset_context.py, which is the
benchmark's parameter policy: per-dataset overrides from
recommended_baseline_hyperparameters.csv where they exist, size-derived formulas
otherwise. Of the 78 in-scope pairs only BF21, BM27, WF26 and WM34 carry
overrides; the other 74 have always used the formulas, so the formulas ARE the
tuned setting for them.

It is reproduced rather than called because emit_dataset_context reads sizes
from dataset_inventory.csv, which lists the ORIGINAL matrices and covers 1 of
our 78 pairs. Sizes here come from the prepared bundle the tools actually
consume, so a pair whose cell or feature count changed in the rebuild gets
parameters matching its real input.

One deliberate departure: emb_dim is additionally clamped to P_shared - 1.
emit_dataset_context derives it from per-modality feature counts, which can
exceed the shared feature space the tools are given, and asking for more
components than the matrix has is a hard error in Seurat and FastMNN.

usage: di_params.py <dataset>            # prints the resolved values
       from di_params import resolve     # returns a dict
"""
import json, os, sys
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
HP = f"{P}/3_data_integration/recommended_baseline_hyperparameters.csv"


def _prep_dir(ds, regime):
    return (f"{P}/Results/prepared_input/closedset/{ds}" if regime == "diagonal"
            else f"{P}/runs_integration/benchmarking_{ds}/prepared_input")


def _override(ds, method):
    """Per-dataset tuned block from the hyperparameter table, if present."""
    try:
        hp = pd.read_csv(HP)
    except Exception:
        return {}
    c0, c1 = hp.columns[0], hp.columns[1]
    r = hp[(hp[c0].astype(str) == ds) & (hp[c1].astype(str) == method)]
    if r.empty:
        return {}
    for c in hp.columns:
        v = r.iloc[0][c]
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                return json.loads(v)
            except Exception:
                pass
    return {}


def resolve(ds):
    MAP = json.load(open(f"{P}/1_coupling_inference/foscttm_input_map.json"))
    v = MAP[ds]
    reg, ptr = v["regime"], v["pair_type"]
    prep = _prep_dir(ds, reg)
    d = f"{P}/data/{ds}"

    o = pd.read_csv(f"{prep}/combined_log_obs.csv")
    n1 = int((o.modality == "mod1").sum())
    n2 = int((o.modality == "mod2").sum())
    p_shared = len(open(f"{prep}/combined_log_features.tsv").read().split())
    # per-modality feature counts, matching emit_dataset_context's p1/p2
    pm = []
    for m in ("mod1", "mod2"):
        f = f"{d}/{m}_features_normalized.tsv"
        pm.append(sum(1 for line in open(f) if line.strip())
                  if os.path.exists(f) else p_shared)
    p1, p2 = pm
    n_total = n1 + n2
    p_min = max(2, min(p1, p2) if min(p1, p2) > 0 else max(p1, p2, 2))
    emb_dim = min(30, max(2, min(n1, n2, p_min) - 1))
    emb_dim = max(2, min(emb_dim, p_shared - 1))      # see docstring

    seurat = _override(ds, "Seurat")
    harmony = _override(ds, "Harmony")
    con = _override(ds, "Concerto")
    sym = _override(ds, "Symphony")

    return {
        "dataset": ds, "regime": reg, "pair_type": ptr, "prep": prep,
        "n1": n1, "n2": n2, "p1": p1, "p2": p2, "p_shared": p_shared,
        "n_total": n_total, "emb_dim": emb_dim, "seed": 42,
        "has_override": bool(seurat or harmony or con or sym),

        "SEURAT_DIMS": int(str(seurat.get("dims", "1:30")).replace("1:", "")),
        "SEURAT_VARIABLE_FEATURES": int(seurat.get("variable_features",
                                                   min(3000, p_min))),
        "SEURAT_K_ANCHOR": int(seurat.get("k.anchor", 5)),
        "SEURAT_K_FILTER": int(seurat.get("k.filter", 200)),
        "SEURAT_K_SCORE": int(seurat.get("k.score", 30)),
        "SEURAT_K_WEIGHT": int(seurat.get("TransferData.k.weight", 50)),

        "HARMONY_PCA_DIM": int(harmony.get("pca_dim", emb_dim)),
        "HARMONY_EMB_DIM": int(harmony.get("emb_dim", emb_dim)),
        "HARMONY_THETA": float(harmony.get("theta", 2.0)),
        "HARMONY_LAMBDA": float(harmony.get("lambda", 1.0)),
        "HARMONY_SIGMA": float(harmony.get("sigma", 0.1)),
        "HARMONY_MAX_ITER": int(harmony.get("max_iter", 20)),

        "SYMPHONY_NPCS": int(sym.get("npcs", emb_dim)),
        "SYMPHONY_NVG": int(sym.get("n_variable_genes", min(2000, p_shared))),
        "SYMPHONY_THETA": float(sym.get("harmony.theta", 2.0)),
        "SYMPHONY_LAMBDA": float(sym.get("harmony.lambda", 1.0)),
        "SYMPHONY_SIGMA": float(sym.get("harmony.sigma", 0.1)),

        "SCANORAMA_DIMRED": emb_dim,
        "SCANORAMA_KNN": 20 if n_total < 50000 else 30,
        "SCANORAMA_SIGMA": 15.0, "SCANORAMA_ALPHA": 0.10,
        "SCANORAMA_BATCH_SIZE": 5000 if n_total < 100000 else 10000,

        "FASTMNN_D": emb_dim, "FASTMNN_K": 20 if n_total < 50000 else 30,

        "SCVI_N_LATENT": emb_dim, "SCVI_N_HIDDEN": 128, "SCVI_N_LAYERS": 2,
        "SCVI_DROPOUT": 0.1,
        "SCVI_MAX_EPOCHS": 100 if n_total < 50000 else 50,
        "SCVI_BATCH_SIZE": 512 if n_total >= 10000 else 256,

        "CONCERTO_EMBEDDING_DIM": int(con.get("embedding_dim", 128)),
        "CONCERTO_PRETRAIN_EPOCHS": int(con.get("pretrain_epochs", 20)),
        "CONCERTO_FINETUNE_EPOCHS": int(con.get("fine_tune_epochs", 10)),
        "CONCERTO_BATCH_SIZE": int(con.get("batch_size",
                                           512 if n_total >= 10000 else 256)),
        "CONCERTO_LR": float(con.get("learning_rate", 1e-4)),

        # WNN needs the two modalities measured in the SAME cells
        "WNN_DIMS": int(str(seurat.get("dims", "1:30")).replace("1:", "")),
        "WNN_MOD2_NORM": ("LogNormalize" if ptr in ("rna_atac", "rna_rna",
                                                    "rna_meth") else "CLR"),
    }


if __name__ == "__main__":
    r = resolve(sys.argv[1])
    for k, v in r.items():
        print(f"{k:<26} {v}")
