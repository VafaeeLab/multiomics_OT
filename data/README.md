# Sample datasets

Two dataset pairs you can run the tool on straight away. Both are diagonal
RNA-ATAC pairs, which means the two modalities were measured in different cells
of the same tissue. Both come from published studies, and both are
redistributed here in the pre-processed form the paper used, not as raw data.

`datasets.csv` holds the same table in machine-readable form.

| Folder | Cells | Tissue | Accession | Source study |
|---|---|---|---|---|
| `mutoD2RNAATAC/` | 3,660 RNA and 2,385 ATAC, 10 cell types | human adult kidney | GEO **GSE151302** | Muto et al., *Nature Communications* 12, 2190 (2021) |
| `trevinoR3RNAATAC/` | 1,028 RNA and 1,027 ATAC | developing human cerebral cortex | GEO **GSE162170** | Trevino et al., *Cell* 184, 5053-5069 (2021) |

Please cite the original study if you use either dataset.

## What is in each folder

One folder per dataset, and the folder name is how every step of the tool finds
it. Nothing is nested below that.

```
mod1_normalized.mtx.gz          modality 1, genes by cells
mod2_normalized.mtx.gz          modality 2, genes by cells
mod1_barcodes_normalized.tsv    cell names of modality 1
mod1_features_normalized.tsv    gene names of modality 1
mod2_barcodes_normalized.tsv    cell names of modality 2
mod2_features_normalized.tsv    gene names of modality 2
mod1_normalized_barcodes.tsv    the same names again, under the second
mod1_normalized_features.tsv      naming pattern the matrix loader searches for
mod2_normalized_barcodes.tsv
mod2_normalized_features.tsv
obs_normalized.csv              cell metadata, including cell type
obs.csv
metadata.json                   where the data came from
preprocessing.json              what was done to them
```

The matrices are stored compressed. `run_example.sh` unpacks them on the first
run, or you can do it by hand:

```bash
cd mutoD2RNAATAC
gunzip -k mod1_normalized.mtx.gz mod2_normalized.mtx.gz
```

**Keep the second set of name files.** The matrix loader looks for
`<matrix name>_barcodes.tsv` to name the matrix **rows** and
`<matrix name>_features.tsv` to name its **columns**. These matrices are stored
genes by cells, so the file called "barcodes" holds gene names. That looks wrong
and is correct. Without those files the loader numbers the cells 0, 1, 2 and the
coupling step then matches cell 0 to cell 0, which is meaningless.

## Adding your own

Make a folder here named after your dataset, put the same seven files in it,
then add an entry for it in `../1_coupling_inference/foscttm_input_map.json`.
Copy an existing entry and change the regime (`coassay` if both modalities were
measured in the same cells, `diagonal` if not) and the modality pair.
