#!/bin/bash
# Run the whole study on the example dataset pair.
#
# By default it uses mutoD2RNAATAC, a diagonal RNA-ATAC pair of human adult
# kidney with 3,660 cells in modality 1 and 2,385 in modality 2. A second pair,
# trevinoR3RNAATAC, is also included. Both are already in data/.
#
#     ./run_example.sh
#
# Results are written into Results/:
#     Results/OT_Results/                  coupling, ambiguity, elbow plot
#     Results/Data_integration_results/    embedding and the ElasticFlow upgrade
#     Results/Label_transfer_results/      predictions and the BlendGate upgrade
#     Results/prepared_input/              the combined matrix the tools read
#
# Set PY first if the interpreter you want is not the one on your path, for
# example:  export PY=~/miniconda3/envs/pytorch_env/bin/python
set -uo pipefail

CODE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-python3}

export PROJECT_ROOT="$CODE"
export OT_ROOT="$CODE/Results/OT_Results"
export AMB_ROOT="$CODE/Results/OT_Results"
export DI_ROOT="$CODE/Results/Data_integration_results"
export LT_ROOT="$CODE/Results/Label_transfer_results"
export PYTHONUNBUFFERED=1

DS=mutoD2RNAATAC           # or trevinoR3RNAATAC, the other included pair
PAIR="$CODE/data/$DS"
OBS="$PAIR/obs_normalized.csv"
BASE="$OT_ROOT/diagonal/rna-atac/$DS"
PREP="$CODE/Results/prepared_input/closedset/$DS"

DI_TOOL=AnchorAE           # Others: AnchorFormer, AnchorMix, seurat, harmony, ...
LT_TOOL=AnchorAE-LT        # Others: AnchorFormer, AnchorAE-HGB, SingleR, ...

step () { echo; echo "===== $* ====="; }

# the matrices ship compressed, because together they are 228 MB unpacked
if [ ! -f "$PAIR/mod1_normalized.mtx" ]; then
  step "unpacking the matrices"
  gunzip -k "$PAIR/mod1_normalized.mtx.gz" "$PAIR/mod2_normalized.mtx.gz"
fi

step "step 0 of 4: build the combined matrix the integration tools read"
mkdir -p "$PREP"
cd "$CODE/3_data_integration"
$PY prepare_integration_input.py --dataset_id "$DS" \
  --x1 "$PAIR/mod1_normalized.mtx" --x2 "$PAIR/mod2_normalized.mtx" \
  --barcodes1 "$PAIR/mod1_barcodes_normalized.tsv" \
  --barcodes2 "$PAIR/mod2_barcodes_normalized.tsv" \
  --features1 "$PAIR/mod1_features_normalized.tsv" \
  --features2 "$PAIR/mod2_features_normalized.tsv" \
  --labels1 "$OBS" --labels2 "$OBS" \
  --orientation1 feature_by_cell --orientation2 feature_by_cell \
  --outdir "$PREP"

step "step 1 of 4: CCA-OT coupling -> Results/OT_Results/"
cd "$CODE/1_coupling_inference"
$PY cca_ot.py \
  --rna "$PAIR/mod1_normalized.mtx" --atac "$PAIR/mod2_normalized.mtx" \
  --rna_orientation feature_by_cell --atac_orientation feature_by_cell \
  --mod2_kind gene_activity \
  --outdir "$BASE/cca_ot" --dataset_id "$DS" \
  --n_hvg 2000 --ot_eps 0.10 --d_cca 30 --k_anchor 15

step "step 2 of 4: ambiguity and elbow plot -> Results/OT_Results/"
cd "$CODE/2_ambiguity_and_elbow"
$PY run_ambiguity_lmethod.py --dataset "$DS" --coupling cca_ot --out "$AMB_ROOT"
$PY make_elbow_plots.py --dataset "$DS" --coupling cca_ot \
  --root "$AMB_ROOT" --out "$OT_ROOT/elbow_plots"

step "step 3 of 4: $DI_TOOL and DI-ElasticFlow -> Results/Data_integration_results/"
cd "$CODE/3_data_integration"
$PY run_di_tool.py "$DS" "$DI_TOOL"
$PY di_upgrade_prod.py "$DI_TOOL" "$DS"

step "step 4 of 4: $LT_TOOL and LT-BlendGate -> Results/Label_transfer_results/"
cd "$CODE/4_label_transfer"
$PY lt_build_embed.py "$DS" --arch AnchorAE   # AnchorAE encoder only
$PY run_lt_tool.py "$DS" "$LT_TOOL"
$PY lt_upgrade_lm.py "$LT_TOOL" "$DS"

step "finished"
echo "coupling and ambiguity : $OT_ROOT"
echo "data integration       : $DI_ROOT"
echo "label transfer         : $LT_ROOT"
