#!/usr/bin/env python3
"""Regenerate every coupling in runs/ using the tuned parameters.

Two waves, because ensemble is not independent: ensemble_final.py reads the
diffusion, cca_ot and maxfuse couplings off disk and fuses them, so it cannot
run beside them. It is submitted at the same time but held on a PBS afterok
dependency against its own three component jobs, so it starts the moment they
land without anyone having to come back and launch it.

    wave A   maxfuse, diffusion, cca_ot   78 pairs   234 jobs
             ensemble                     78 pairs    78 jobs (held)
    wave B   scot, unioncom, mmdma        29 co-assay 87 jobs
                                                     ---
                                                     399

Parameters come from model_selection/results/coupling__selected.csv. FOSCTTM
requires ground-truth cell pairing so it exists only on co-assay, which means
the diagonal groups inherit:

    diagonal/rna-atac       <- coassay/rna-atac      (same modality pair)
    diagonal/rna-methylome  <- coassay/rna-methylome (same modality pair)
    diagonal/rna-rna        <- coassay/rna-atac      (user decision: after gene
                               activity conversion rna-atac is also gene-space
                               against gene-space, the closest analogue)

One tool/group has no winner at all: unioncom on coassay/rna-adt, because the
three CITE pairs exceeded the CPU time limit during tuning. It falls back to
the rna-atac configuration, which is safe here because unioncom's rna-atac and
rna-methylome winners are identical (eps=0.001, kmax=5, beta=0.5) and round 1
of the coordinate-descent sweep measured kmax as having no effect at all.

    usage: submit_couplings.py A [--dry-run]
           submit_couplings.py B [--dry-run]
"""
import json, os, subprocess, sys
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
CS = f"{P}/1_coupling_inference"
PY = os.environ.get("PY", sys.executable)
PT = {"rna_adt": "rna-adt", "rna_atac": "rna-atac", "rna_meth": "rna-methylome",
      "rna_rna": "rna-rna"}
WAVE_A = ["maxfuse", "diffusion", "cca_ot"]
WAVE_B = ["scot", "unioncom", "mmdma", "diffusion_GW"]

wave = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
DRY = "--dry-run" in sys.argv

MAP = json.load(open(f"{CS}/foscttm_input_map.json"))
SEL = pd.read_csv(f"{P}/model_selection/results/coupling__selected.csv")

# ---- tuned parameters, keyed (tool, pair_type_label) ------------------------
TUNED = {}
for _, r in SEL.iterrows():
    cfg = {c: r[c] for c in SEL.columns
           if c not in ("tool", "regime", "pair_type", "mean_foscttm",
                        "n_datasets", "group_datasets", "complete")
           and pd.notna(r[c])}
    TUNED[(r.tool, r.pair_type)] = cfg


_UC = {}


def _unioncom_cfg(pt_label):
    """UnionCom's configuration from the coordinate-descent sweep.

    From the GPU probe, not the old coordinate-descent sweep. That sweep chose
    eps=0.02, which collapses the embedding and reports a fake FOSCTTM near
    zero; four of the five couplings it produced are degenerate. The probe's
    winner (eps=0.0005, rho=5) is interior to a tested range on both live axes.
    """
    if not _UC:
        import json as _json
        f = f"{P}/model_selection/results/unioncom__production_params.json"
        d = _json.load(open(f))
        for g, c in d.items():
            if g.startswith("_"):
                continue
            _UC[g] = {k: (int(v) if float(v) == int(float(v)) else float(v))
                      for k, v in c.items()}
    return _UC.get(pt_label)


def params(tool, pt_label):
    """Winner for this group, or the documented inherited fallback."""
    if tool == "unioncom":
        c = _unioncom_cfg(pt_label)
        if c:
            return c, "coordinate-descent sweep"
    if (tool, pt_label) in TUNED:
        return TUNED[(tool, pt_label)], "tuned"
    if pt_label == "rna-rna" and (tool, "rna-atac") in TUNED:
        return TUNED[(tool, "rna-atac")], "inherited from coassay/rna-atac"
    if (tool, "rna-atac") in TUNED:            # unioncom on rna-adt
        return TUNED[(tool, "rna-atac")], "inherited from coassay/rna-atac"
    return None, "NO PARAMETERS"


def num(v):
    f = float(v)
    return str(int(f)) if f == int(f) else str(f)


def build(tool, ds, m, cfg, out):
    """Same CLI the tuner used, writing to the final location."""
    env = {}
    if tool == "scot":
        cmd = [PY, f"{CS}/scot_final.py", ds, num(cfg["k"]), num(cfg["eps"]), out]
    elif tool == "unioncom":
        # rho was hardcoded to 10 here, which is the value the coordinate
        # descent showed to be wrong; it and output_dim now come from cfg.
        cmd = [PY, f"{CS}/unioncom_final.py", ds, num(cfg["kmax"]),
               num(cfg["epsilon"]), num(cfg["beta"]), num(cfg["rho"]), out]
        env = dict(UC_OUTPUT_DIM=num(cfg.get("output_dim", 8)))
    elif tool == "mmdma":
        cmd = [PY, f"{CS}/mmdma_final.py", ds, num(cfg["d"]),
               num(cfg["lam_orth"]), num(cfg["lam_dist"]), "0.005", out,
               "--epochs", "10000"]
    elif tool == "maxfuse":
        env = dict(DS=ds, NN=num(cfg["nn"]), SVD=num(cfg["svd"]),
                   WT=num(cfg["wt"]), OUT=out)
        cmd = [PY, f"{CS}/maxfuse_run.py"]
    elif tool in ("diffusion", "diffusion_GW"):
        script = "diffusion_v1.py" if tool == "diffusion" else "diffusion_v2.py"
        cmd = [PY, f"{CS}/{script}",
               "--matrix1", m["rna"], "--matrix2", m["atac"],
               "--modality1", "rna", "--modality2", "other",
               "--orientation1", m["ori1"], "--orientation2", m["ori2"],
               "--cell_id_alignment", "True",
               "--outdir", out, "--dataset_id", ds,
               "--hvg", num(cfg["hvg"]), "--eps", num(cfg["eps"]),
               "--alpha", num(cfg["alpha"])]
        if tool == "diffusion_GW":
            cmd += ["--refine_rounds", num(cfg["refine_rounds"])]
    elif tool == "cca_ot":
        cmd = [PY, f"{CS}/cca_ot.py",
               "--rna", m["rna"], "--atac", m["atac"],
               "--rna_orientation", m["ori1"], "--atac_orientation", m["ori2"],
               "--mod2_kind", m["kind"], "--outdir", out, "--dataset_id", ds,
               "--n_hvg", num(cfg["n_hvg"]), "--ot_eps", num(cfg["ot_eps"]),
               "--d_cca", num(cfg["d_cca"]), "--k_anchor", num(cfg["k_anchor"])]
    elif tool == "ensemble":
        base = f"{OT_ROOT}/{m['regime']}/{PT[m['pair_type']]}/{ds}"
        env = dict(DS=ds, CSLS_K=num(cfg["csls_k"]), OUT=out,
                   COUPLING_BASE=base)
        cmd = [PY, f"{CS}/ensemble_final.py"]
    else:
        raise SystemExit(f"no command for {tool}")
    return cmd, env


os.makedirs(f"{P}/logs/coupling", exist_ok=True)
PBS = f"{P}/logs/coupling/cpl.pbs"
open(PBS, "w").write(f"""#!/bin/bash
#PBS -P yr31
#PBS -q normal
#PBS -l ncpus=6,mem=120GB,jobfs=50GB,walltime=24:00:00,storage=gdata/yr31,wd
#PBS -j oe
#PBS -m n
cd {P}
module load python3/3.10.4 >/dev/null 2>&1 || true
export PROJECT_ROOT={P}
export R_LIBS_USER={P}/R_analysis/R_libs_4.3
mkdir -p "$OUTDIR"
eval "$CMDLINE"
""")


def submit(tool, ds, m, after=None):
    pt = PT[m["pair_type"]]
    cfg, prov = params(tool, pt)
    if cfg is None:
        print(f"  SKIP {tool}/{ds}: no parameters")
        return None, prov
    out = f"{OT_ROOT}/{m['regime']}/{pt}/{ds}/{tool}"
    cmd, env = build(tool, ds, m, cfg, out)
    line = " ".join(f"'{c}'" for c in cmd)
    v = [f"OUTDIR={out}", f"CMDLINE={line}"] + [f"{k}={x}" for k, x in env.items()]
    q = ["qsub", "-N", f"c{tool[:3]}{ds[:8]}"[:14],
         "-o", f"{P}/logs/coupling/{tool}__{ds}.log",
         "-v", ",".join(v)]
    if after:
        q += ["-W", "depend=afterok:" + ":".join(after)]
    q += [PBS]
    if DRY:
        return "DRY", prov
    r = subprocess.run(q, cwd=P, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL {tool}/{ds}: {r.stderr.strip()[:130]}")
        return None, prov
    return r.stdout.strip(), prov


tools = WAVE_A if wave == "A" else WAVE_B
if "--tools" in sys.argv:                      # e.g. --tools scot,mmdma
    want = sys.argv[sys.argv.index("--tools") + 1].split(",")
    tools = [t for t in tools if t in want]
    print(f"tool filter: {tools}")
n = 0
prov_count = {}
ONLY = ("coassay" if "--coassay" in sys.argv else
        "diagonal" if "--diagonal" in sys.argv else None)
for ds, m in sorted(MAP.items()):
    if wave == "B" and m["regime"] != "coassay":
        continue
    if ONLY and m["regime"] != ONLY:
        continue
    comp = []
    for tool in tools:
        jid, prov = submit(tool, ds, m)
        if jid:
            n += 1
            prov_count[prov] = prov_count.get(prov, 0) + 1
            if wave == "A" and jid != "DRY":
                comp.append(jid)
    if wave == "A" and comp:
        jid, prov = submit("ensemble", ds, m, after=comp)
        if jid:
            n += 1
            prov_count[prov] = prov_count.get(prov, 0) + 1

print(f"\nwave {wave}: submitted {n} jobs")
for k, v in sorted(prov_count.items()):
    print(f"   {v:>4}  {k}")
