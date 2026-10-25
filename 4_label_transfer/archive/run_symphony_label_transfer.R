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
  make_option("--dims", type="integer", default=30),
  make_option("--knn_k", type="integer", default=15),
  make_option("--n_variable_genes", type="integer", default=3000)
)
args <- parse_args(OptionParser(option_list=option_list))
method <- "label_transfer_Symphony"
dir.create(args$out_dir, recursive=TRUE, showWarnings=FALSE)

if (!requireNamespace("symphony", quietly=TRUE)) {
  write_status(args$out_dir, method, args$dataset_id, "skipped_missing_package", "Install the symphony R package.")
  quit(save="no", status=strict_quit_status())
}
if (!requireNamespace("RANN", quietly=TRUE)) {
  write_status(args$out_dir, method, args$dataset_id, "skipped_missing_package", "Install the RANN R package for kNN classification.")
  quit(save="no", status=strict_quit_status())
}

extract_cells_by_dim <- function(z, n_cells) {
  if (is.null(z)) return(NULL)
  z <- as.matrix(z)
  if (nrow(z) == n_cells) return(z)
  if (ncol(z) == n_cells) return(t(z))
  NULL
}
extract_ref_embedding <- function(ref, n_cells) {
  for (nm in c("Z_corr", "Z", "Z_orig", "Z_pca", "Z_cos")) {
    if (!is.null(ref[[nm]])) {
      z <- extract_cells_by_dim(ref[[nm]], n_cells)
      if (!is.null(z)) return(z)
    }
  }
  stop("Could not extract reference Symphony embedding from reference object.")
}
extract_query_embedding <- function(query, n_cells) {
  for (nm in c("Z", "Zq", "Z_corr", "Z_pca")) {
    if (!is.null(query[[nm]])) {
      z <- extract_cells_by_dim(query[[nm]], n_cells)
      if (!is.null(z)) return(z)
    }
  }
  stop("Could not extract query Symphony embedding from query object.")
}
knn_vote <- function(refZ, queryZ, refLabels, k) {
  k <- min(k, nrow(refZ))
  nn <- RANN::nn2(refZ, queryZ, k=k)
  labs <- character(nrow(queryZ)); scores <- numeric(nrow(queryZ))
  for (i in seq_len(nrow(queryZ))) {
    li <- refLabels[nn$nn.idx[i, ]]
    tab <- sort(table(li), decreasing=TRUE)
    labs[i] <- names(tab)[1]
    scores[i] <- as.numeric(tab[1]) / k
  }
  list(labels=labs, scores=scores)
}

tryCatch({
  rec1 <- read_mtx_genes_by_cells(args$x1, args$barcodes1, args$features1, args$orientation1, "mod1")
  rec2 <- read_mtx_genes_by_cells(args$x2, args$barcodes2, args$features2, args$orientation2, "mod2")
  common <- prepare_common_features(rec1, rec2, min_shared_features=args$min_shared_features)
  M1 <- normalize_counts(common$M1); M2 <- normalize_counts(common$M2)
  lab1 <- load_labels(args$labels1, args$label_id_col, args$label_col)
  lab2 <- load_labels(args$labels2, args$label_id_col, args$label_col)
  y1 <- align_labels(colnames(M1), lab1)
  y2 <- align_labels(colnames(M2), lab2)

  run_dir <- function(refM, queryM, refLabels, queryLabels, direction) {
    keep <- !is.na(refLabels) & refLabels != ""
    refM <- refM[, keep, drop=FALSE]
    refLabels <- refLabels[keep]
    ref_meta <- data.frame(cell_type=refLabels, batch="reference", row.names=colnames(refM), stringsAsFactors=FALSE)
    query_meta <- data.frame(batch=rep("query", ncol(queryM)), row.names=colnames(queryM), stringsAsFactors=FALSE)
    topn <- min(args$n_variable_genes, nrow(refM))
    reference <- symphony::buildReference(refM, ref_meta, vars=NULL, K=min(100, max(2, floor(ncol(refM) / 10))), verbose=FALSE, do_umap=FALSE, do_normalize=FALSE, topn=topn, d=min(args$dims, topn - 1))
    query <- symphony::mapQuery(queryM, query_meta, reference, vars=NULL, do_normalize=FALSE, do_umap=FALSE)
    refZ <- extract_ref_embedding(reference, ncol(refM))
    queryZ <- extract_query_embedding(query, ncol(queryM))
    pred <- knn_vote(refZ, queryZ, refLabels, args$knn_k)
    standard_pred_df(args$dataset_id, method, direction, colnames(queryM), queryLabels, pred$labels, pred$scores)
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
