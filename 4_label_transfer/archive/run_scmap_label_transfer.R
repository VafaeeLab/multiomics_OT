#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(Matrix)
  library(data.table)
})
cmd_args_all <- commandArgs(FALSE)
file_arg <- grep("^--file=", cmd_args_all, value=TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "standard_label_transfer_utils.R"))

option_list <- list(
  make_option("--dataset_id", type="character"),
  make_option("--x1", type="character"), make_option("--x2", type="character"),
  make_option("--barcodes1", type="character", default=""), make_option("--barcodes2", type="character", default=""),
  make_option("--features1", type="character", default=""), make_option("--features2", type="character", default=""),
  make_option("--labels1", type="character"), make_option("--labels2", type="character"),
  make_option("--out_dir", type="character"),
  make_option("--orientation1", type="character", default="cell_by_feature"),
  make_option("--orientation2", type="character", default="cell_by_feature"),
  make_option("--label_id_col", type="character", default="cell_id"),
  make_option("--label_col", type="character", default="cell_type"),
  make_option("--min_shared_features", type="integer", default=20),
  make_option("--threshold", type="double", default=0.7),
  make_option("--n_features", type="integer", default=500),
  make_option("--w", type="integer", default=10),
  make_option("--max_dense_entries", type="double", default=2.5e8)
)
args <- parse_args(OptionParser(option_list=option_list))
method <- "label_transfer_scmap"
dir.create(args$out_dir, recursive=TRUE, showWarnings=FALSE)

if (!requireNamespace("scmap", quietly=TRUE) || !requireNamespace("SingleCellExperiment", quietly=TRUE) || !requireNamespace("scater", quietly=TRUE)) {
  write_status(args$out_dir, method, args$dataset_id, "skipped_missing_package", "Install Bioconductor packages scmap, SingleCellExperiment, and scater.")
  quit(save="no", status=strict_quit_status())
}

tryCatch({
  rec1 <- read_mtx_genes_by_cells(args$x1, args$barcodes1, args$features1, args$orientation1, "mod1")
  rec2 <- read_mtx_genes_by_cells(args$x2, args$barcodes2, args$features2, args$orientation2, "mod2")
  common <- prepare_common_features(rec1, rec2, min_shared_features=args$min_shared_features)
  M1 <- normalize_counts(common$M1); M2 <- normalize_counts(common$M2)
  if ((nrow(M1) * max(ncol(M1), ncol(M2))) > args$max_dense_entries) {
    stop(sprintf("scmap dense matrix guard triggered: features*cells %.3g > max_dense_entries %.3g", nrow(M1) * max(ncol(M1), ncol(M2)), args$max_dense_entries))
  }
  lab1 <- load_labels(args$labels1, args$label_id_col, args$label_col)
  lab2 <- load_labels(args$labels2, args$label_id_col, args$label_col)
  y1 <- align_labels(colnames(M1), lab1)
  y2 <- align_labels(colnames(M2), lab2)

  make_sce <- function(M, labels) {
    sce <- SingleCellExperiment::SingleCellExperiment(assays=list(normcounts=as.matrix(M)))
    SingleCellExperiment::logcounts(sce) <- log2(SingleCellExperiment::normcounts(sce) + 1)
    SummarizedExperiment::rowData(sce)$feature_symbol <- rownames(sce)
    SummarizedExperiment::colData(sce)$cell_type <- labels
    sce
  }
  run_dir <- function(refM, queryM, refLabels, queryLabels, direction) {
    keep <- !is.na(refLabels) & refLabels != ""
    ref <- make_sce(refM[, keep, drop=FALSE], refLabels[keep])
    query <- make_sce(queryM, queryLabels)
    ref <- scmap::selectFeatures(ref, n_features=args$n_features, suppress_plot=TRUE)
    ref <- scmap::indexCluster(ref, cluster_col="cell_type")
    idx <- S4Vectors::metadata(ref)$scmap_cluster_index
    res <- scmap::scmapCluster(projection=query, index_list=list(ref=idx), threshold=args$threshold)
    labels <- as.character(res$scmap_cluster_labs[, "ref"])
    labels[is.na(labels) | labels == "NA"] <- "unassigned"
    # scmap stores similarities in metadata depending on version; use NA if absent.
    score <- tryCatch(as.numeric(res$scmap_cluster_siml[, "ref"]), error=function(e) rep(NA_real_, length(labels)))
    standard_pred_df(args$dataset_id, method, direction, colnames(queryM), queryLabels, labels, score)
  }
  out <- rbind(
    run_dir(M1, M2, y1, y2, "mod1_to_mod2"),
    run_dir(M2, M1, y2, y1, "mod2_to_mod1")
  )
  data.table::fwrite(out, file.path(args$out_dir, "standardized_predictions_raw.csv"))
  data.table::fwrite(data.frame(feature_alignment=common$mode, n_shared_features=common$n_shared), file.path(args$out_dir, "feature_alignment.csv"))
}, error=function(e) {
  msg <- conditionMessage(e)
  na <- grepl("No comparable feature space", msg, fixed=TRUE)
  write_status(args$out_dir, method, args$dataset_id, if (na) "skipped_not_applicable" else "skipped_error", msg)
  quit(save="no", status=if (na) 0L else strict_quit_status())
})
