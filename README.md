# multiomics_OT

**Finding the cells that single-cell multi-omics methods cannot match, and using
that to make integration and label transfer better.**

Code for *Optimal transport reveals unresolved cells in single-cell
multi-omics*, by Mehran Piran and Fatemeh Vafaee, Vafaee Lab, UNSW Sydney.

---

## What this does

When you have two measurements of the same tissue — say RNA and ATAC — most
tools quietly decide which cell in one matches which cell in the other, and then
never tell you how sure they were. Every cell in the answer looks equally
trustworthy. It is not.

This tool makes that hidden step visible and then uses it. It does four things,
and you can stop after any of them.

**1. It works out the correspondence between the two modalities.** The result is
a coupling matrix: one row per cell in the first modality, one column per cell
in the second, and each entry says how much the two cells belong together. Four
ways of building it are provided.

**2. It scores how uncertain each cell's match is.** A cell whose correspondence
points firmly at one group of cells is easy. A cell whose correspondence is
split between two groups is not, and that cell gets a high ambiguity score. The
tool then finds the natural cut-off in those scores and tells you which cells
fall above it.

**3. It improves data integration.** DI-ElasticFlow nudges the query cells
toward where the coupling says they belong, then drops the uncertain ones.

**4. It improves label transfer.** LT-BlendGate blends the coupling's opinion
into another tool's prediction, but only where that tool was unsure, then drops
the uncertain query cells.

Steps 3 and 4 work on top of eleven integration tools and ten label-transfer
tools, including Seurat, Harmony, scVI and SingleR. You do not replace the tool
you already use; you improve what it gave you.

---

## Install

```bash
git clone https://github.com/VafaeeLab/multiomics_OT.git
cd multiomics_OT
conda env create -f environment.yml
conda activate multiomics_ot
```

If you would rather use an environment you already have, Python 3.10 or newer
and `pip install -r requirements.txt` is enough.

There is nothing to compile and nothing to install system-wide. The tool is a
set of scripts you run in place.

---

## Try it in one command

A small dataset is included, so you can see the whole thing work before you
point it at your own data.

```bash
./run_example.sh
```

This takes a few minutes. It runs all four steps on a pair of 1,028 and 1,027
cells and writes everything into `Results/`.

If the Python you want is not the first one on your path, say so first:

```bash
export PY=~/miniconda3/envs/multiomics_ot/bin/python
./run_example.sh
```

The results of that run are already in this repository, so you can compare what
you get against what you should get.

---

## The four steps, one at a time

Set these once per session. `PROJECT_ROOT` is the folder you cloned into.

```bash
export PROJECT_ROOT=$PWD
export OT_ROOT=$PROJECT_ROOT/Results/OT_Results
export AMB_ROOT=$PROJECT_ROOT/Results/OT_Results
export DI_ROOT=$PROJECT_ROOT/Results/Data_integration_results
export LT_ROOT=$PROJECT_ROOT/Results/Label_transfer_results

DS=trevinoR3RNAATAC
PAIR=$PROJECT_ROOT/Sample_dataset/$DS
BASE=$OT_ROOT/diagonal/rna-atac/$DS
PREP=$PROJECT_ROOT/Results/prepared_input/closedset/$DS
```

### Step 0. Build the matrix the integration tools read

The integration tools need one combined matrix on a shared gene space, rather
than the two separate ones. This builds it, once per dataset pair.

```bash
cd 3_data_integration
python3 prepare_integration_input.py --dataset_id $DS \
  --x1 $PAIR/mod1_normalized.mtx --x2 $PAIR/mod2_normalized.mtx \
  --barcodes1 $PAIR/mod1_barcodes_normalized.tsv \
  --barcodes2 $PAIR/mod2_barcodes_normalized.tsv \
  --features1 $PAIR/mod1_features_normalized.tsv \
  --features2 $PAIR/mod2_features_normalized.tsv \
  --labels1 $PAIR/obs_normalized.csv --labels2 $PAIR/obs_normalized.csv \
  --orientation1 feature_by_cell --orientation2 feature_by_cell \
  --outdir $PREP
```

### Step 1. Get the coupling matrix

```bash
cd ../1_coupling_inference
python3 cca_ot.py \
  --rna $PAIR/mod1_normalized.mtx --atac $PAIR/mod2_normalized.mtx \
  --rna_orientation feature_by_cell --atac_orientation feature_by_cell \
  --mod2_kind gene_activity \
  --outdir $BASE/cca_ot --dataset_id $DS \
  --n_hvg 2000 --ot_eps 0.05 --d_cca 30 --k_anchor 20
```

That gives you CCA-OT. Three more are available, and the fourth combines the
first three, so it has to run last:

```bash
python3 diffusion_v1.py --matrix1 $PAIR/mod1_normalized.mtx \
  --matrix2 $PAIR/mod2_normalized.mtx \
  --modality1 rna --modality2 other \
  --orientation1 feature_by_cell --orientation2 feature_by_cell \
  --cell_id_alignment True --outdir $BASE/diffusion --dataset_id $DS \
  --hvg 2000 --eps 0.05 --alpha 0.5

DS=$DS NN=15 SVD=30 WT=0.3 OUT=$BASE/maxfuse python3 maxfuse_run.py

DS=$DS CSLS_K=10 OUT=$BASE/ensemble COUPLING_BASE=$BASE python3 ensemble_final.py
```

| Coupling | Script | What it leans on |
|---|---|---|
| CCA-OT | `cca_ot.py` | variation the two feature sets share |
| Diffusion-OT | `diffusion_v1.py` | the shape of each modality's own manifold |
| MaxFuse | `maxfuse_run.py` | the published MaxFuse method, good when the link is weak |
| Ensemble-OT | `ensemble_final.py` | a rank consensus of the other three |

**Check the shape of the result.** For this pair it has to be 1,028 rows by
1,027 columns. If you get a square matrix, the cell-name files are missing and
the two modalities have been matched by position rather than by cell. See the
note on name files further down.

**Output:** `gamma.npy`, the coupling. Also `pairwise_matrix_common.npy`, the
distance matrix, which FOSCTTM and Ensemble-OT use. For a co-assay pair, where
each cell's true partner is known, you also get `per_cell_foscttm.csv`.

### Step 2. Score the ambiguity and draw the elbow

```bash
cd ../2_ambiguity_and_elbow
python3 run_ambiguity_lmethod.py --dataset $DS --coupling cca_ot --out $AMB_ROOT
python3 make_elbow_plots.py --dataset $DS --coupling cca_ot \
  --root $AMB_ROOT --out $OT_ROOT/elbow_plots
```

For each cell, the coupling mass along its row is pooled by cell group on the
other side, which gives every group a vote. The ambiguity score is one minus
the gap between the two strongest votes. A cell that votes clearly for one group
scores low. A cell torn between two groups scores high.

The cut-off is the elbow of the sorted score curve. We find it by fitting two
straight lines to the curve of `exp(10 × score)` and taking the split point
where they fit best, and we never flag more than half the cells. The transform
matters: on the raw curve the rule almost always runs past the half-way limit
and so never finds a real elbow.

Three versions are written, because a modality can be scored with no labels at
all or against the other side's known cell types:

| Folder | Modality 1 scored by | Modality 2 scored by | Use it for |
|---|---|---|---|
| `pseudo_label_pseudo_label` | Leiden clusters, no labels | Leiden clusters, no labels | data integration |
| `pseudo_label_margin_label` | Leiden clusters, no labels | true labels of modality 1 | label transfer, 1 to 2 |
| `margin_label_pseudo_label` | true labels of modality 2 | Leiden clusters, no labels | label transfer, 2 to 1 |

**Output:** a score and a rank for every cell, the list of flagged cells, the
list of kept cells, the elbow figures, and `summary.json`. That last file
records `rule_gap_pp`, which is how far apart this rule and an unrelated rule
put the elbow. When two different rules agree, the elbow is real.

On the sample pair the cut lands at 387 of 1,028 cells in the first modality,
which is 37.6 per cent, and at 513 of 1,027 in the second, which is 50.0 per
cent. The second one hit the half-way limit rather than a genuine elbow, and the
figure says so.

### Step 3. Integrate, then improve it

```bash
cd ../3_data_integration
python3 run_di_tool.py $DS AnchorAE
python3 di_upgrade_prod.py AnchorAE $DS
```

Mind the argument order. The first script takes the dataset first; the second
takes the tool first.

Eleven tools can go in that slot: **AnchorAE**, **AnchorFormer**, **AnchorMix**
(ours), and Seurat, Harmony, Symphony, fastMNN, scVI, Scanorama, Concerto and
WNN. WNN needs both modalities measured in the same cells, so it cannot run on a
diagonal pair. The published tools need their own packages installed.

DI-ElasticFlow holds the reference cells still and pulls each query cell toward
where its coupling partners sit, while a smoothness penalty keeps neighbours
moving together. Writing `Z` for the embedding, `B` for the targets and `L` for
the graph Laplacian, the displacement `U` solves

```
(I + lambda · L) U = B − Z
```

A bigger lambda makes the movement stiffer. It is 0.1 for co-assay pairs and
0.75 for diagonal ones, because a test across the benchmark found that co-assay
pairs barely respond to lambda while diagonal pairs keep improving up to 0.75.

### Step 4. Transfer labels, then improve that

```bash
cd ../4_label_transfer
python3 lt_build_embed.py $DS --arch AnchorAE
python3 run_lt_tool.py $DS AnchorAE-LT
python3 lt_upgrade_lm.py AnchorAE-LT $DS
```

Ten tools are available: **AnchorAE-LT**, **AnchorFormer**, **AnchorAE-HGB**
(ours), and Seurat, Symphony, SingleR, scmap, ItClust, Concerto and Harmony.

LT-BlendGate mixes the coupling's class vote into the tool's own prediction. The
weight for a query cell is

```
lambda_i = lambda_max × (1 − base_confidence_i) × vote_confidence_i
```

so the coupling is only listened to when the tool is unsure **and** the vote is
sharp. A confident tool keeps its answer, which is what stops the coupling from
pulling a correct prediction off target. `lambda_max` is 0.5.

Both directions are scored. Set A uses modality 1 as the reference, set B uses
modality 2.

---

## What the sample results look like

Four arms are scored in both steps 3 and 4. The fourth one is the point of the
whole exercise.

**Data integration**, AnchorAE with CCA-OT:

| Arm | NMI | ARI | iLISI |
|---|---|---|---|
| baseline | 0.4385 | 0.3180 | 0.665 |
| upgraded | 0.4385 | 0.3180 | 0.665 |
| upgraded, ambiguous cells removed | 0.5456 | 0.4605 | 0.640 |
| upgraded, same number removed at random | 0.4768 | 0.3527 | 0.698 |

**Label transfer**, AnchorAE-LT with RNA as the reference:

| Arm | Accuracy | Macro-F1 |
|---|---|---|
| baseline | 0.6095 | 0.5720 |
| upgraded | 0.6436 | 0.5975 |
| upgraded, ambiguous cells removed | 0.7025 | 0.6341 |
| upgraded, same number removed at random | 0.6452 | 0.5975 |

Three things in there are worth dwelling on.

**Why the random arm exists.** Throwing cells away makes almost any score go up,
because what is left is easier. So beating the plain upgraded row proves
nothing. The only honest test is whether removing the flagged cells beats
removing the same number at random, and it does in both tables.

**Why the upgraded row is identical to the baseline above.** That is the
do-no-harm rule doing its job. ElasticFlow could not find a move that improved
mixing without damaging the structure, so it handed back the original embedding
untouched.

**Why iLISI goes the wrong way.** The cells we flag are disproportionately the
ones bridging the two modalities, so removing them costs you mixing. We report
it rather than hide it.

These are one pair. The paper reports averages over 78.

---

## Using it on your own data

Each dataset pair needs one folder with seven files:

```
Sample_dataset/<your_dataset>/
    mod1_normalized.mtx              modality 1, genes by cells
    mod2_normalized.mtx              modality 2, genes by cells
    mod1_barcodes_normalized.tsv     cell names of modality 1
    mod1_features_normalized.tsv     gene names of modality 1
    mod2_barcodes_normalized.tsv     cell names of modality 2
    mod2_features_normalized.tsv     gene names of modality 2
    obs_normalized.csv               cell metadata, with a cell_type column
```

Then add an entry for it in
`1_coupling_inference/foscttm_input_map.json`, which is the register every step
reads. Copy an existing entry and change the paths, the regime (`coassay` if
both modalities were measured in the same cells, `diagonal` if not) and the
modality pair.

**One trap worth knowing about.** The matrix loader also looks for sidecar files
called `<matrix name>_barcodes.tsv` and `<matrix name>_features.tsv`. Those
names refer to the matrix **rows and columns**, not to biology. Because these
matrices are stored genes by cells, the file called "barcodes" holds **gene**
names. That looks wrong and is correct. If those files are missing, the loader
numbers the cells 0, 1, 2 and so on, and the coupling step then matches cell 0
to cell 0, which is meaningless. The sample dataset includes both naming
patterns so you can see what is expected.

---

## What is in this repository

| Folder | What it holds |
|---|---|
| `1_coupling_inference/` | the four couplings, FOSCTTM scoring, and the dataset register |
| `2_ambiguity_and_elbow/` | ambiguity scoring, the elbow rule, the elbow plot |
| `3_data_integration/` | integration runners, DI-ElasticFlow, and wrappers for the published tools |
| `4_label_transfer/` | label-transfer runners, LT-BlendGate, and wrappers for the published tools |
| `model_selection/` | the tuned settings for all eight couplings and all twenty-two tools |
| `Sample_dataset/` | one small dataset pair you can run immediately |
| `Results/` | what that pair produces, so you can check your own run |

Four scripts run the whole benchmark rather than one pair, and submit cluster
jobs: `submit_couplings.py`, `submit_ambiguity_lmethod.py`, `submit_di_78.py`
and `submit_lt_78.py`.

**Not included.** Third-party packages you install yourself: MaxFuse, SCOT,
UnionCom, Concerto, scVI, Scanorama, and the R packages Seurat, Harmony,
Symphony and batchelor. The other 77 dataset pairs come from public deposits,
and their accession numbers are in the paper's supplementary material.

**One note on naming.** Inside the code our three models used to be called
`attention`, `transformer` and `anchormix`. Those names are gone. The code and
the paper now agree: **AnchorAE**, **AnchorFormer** and **AnchorMix**. The words
"attention" and "transformer" still turn up in comments where they describe a
mechanism rather than a model, which is correct.

---

## Citing this

> Piran, M. and Vafaee, F. *Optimal transport reveals unresolved cells in
> single-cell multi-omics.* 2026.

## Contact

Mehran Piran, m.piran@unsw.edu.au
Fatemeh Vafaee, f.vafaee@unsw.edu.au
