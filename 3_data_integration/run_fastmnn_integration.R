#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(optparse)})
source(file.path(Sys.getenv("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration"), "3_data_integration", "standard_integration_utils.R"))
suppressPackageStartupMessages({library(SingleCellExperiment); library(batchelor)})
opts <- list(make_option("--prepared_dir", type="character"), make_option("--outdir", type="character"), make_option("--dataset_id", type="character"), make_option("--d", type="integer", default=30), make_option("--k", type="integer", default=20), make_option("--seed", type="integer", default=42))
args <- parse_args(OptionParser(option_list=opts)); set.seed(args$seed)
b1 <- read_bundle(args$prepared_dir, "mod1_log"); b2 <- read_bundle(args$prepared_dir, "mod2_log")
# fastMNN searches for mutual nearest neighbours BETWEEN the blocks; with one
# block non-negative and the other zero-centred no MNN pair is ever found.
b1$X <- permod_scale(b1$X, rep("mod1", nrow(b1$X)))
b2$X <- permod_scale(b2$X, rep("mod2", nrow(b2$X)))
sce1 <- SingleCellExperiment(list(logcounts=Matrix::t(b1$X))); rownames(sce1) <- make.unique(b1$features); colnames(sce1) <- paste0("mod1__", b1$obs$cell_id)
sce2 <- SingleCellExperiment(list(logcounts=Matrix::t(b2$X))); rownames(sce2) <- make.unique(b2$features); colnames(sce2) <- paste0("mod2__", b2$obs$cell_id)
common <- intersect(rownames(sce1), rownames(sce2))
if (length(common) < 2) stop(sprintf("FastMNN: only %d shared features between modalities", length(common)))
sce1 <- sce1[common, , drop=FALSE]; sce2 <- sce2[common, , drop=FALSE]
res <- batchelor::fastMNN(sce1, sce2, d=min(args$d, length(common) - 1L), k=args$k)
Z <- as.matrix(reducedDim(res, "corrected"))
obs <- rbind(b1$obs, b2$obs)
write_embedding_csv(Z[, seq_len(min(args$d, ncol(Z))), drop=FALSE], obs, args$outdir, args$dataset_id)
writeLines(jsonlite::toJSON(list(status="ok", args=as.list(args)), auto_unbox=TRUE, pretty=TRUE), file.path(args$outdir,"method_status.json"))
