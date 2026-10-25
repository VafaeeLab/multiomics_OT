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
  make_option("--k_anchor", type="integer", default=5),
  make_option("--k_filter", type="integer", default=200),
  make_option("--k_score", type="integer", default=30),
  make_option("--k_weight", type="integer", default=50)
)
args <- parse_args(OptionParser(option_list=option_list))
method <- "label_transfer_SeuratTransferData"
dir.create(args$out_dir, recursive=TRUE, showWarnings=FALSE)

if (!requireNamespace("Seurat", quietly=TRUE)) {
  write_status(args$out_dir, method, args$dataset_id, "skipped_missing_package", "Install the Seurat R package.")
  quit(save="no", status=strict_quit_status())
}

tryCatch({
  rec1 <- read_mtx_genes_by_cells(args$x1, args$barcodes1, args$features1, args$orientation1, "mod1")
  rec2 <- read_mtx_genes_by_cells(args$x2, args$barcodes2, args$features2, args$orientation2, "mod2")
  common <- prepare_common_features(rec1, rec2, min_shared_features=args$min_shared_features)
  M1 <- common$M1; M2 <- common$M2
  lab1 <- load_labels(args$labels1, args$label_id_col, args$label_col)
  lab2 <- load_labels(args$labels2, args$label_id_col, args$label_col)
  y1 <- align_labels(colnames(M1), lab1)
  y2 <- align_labels(colnames(M2), lab2)

  make_obj <- function(M, labels) {
    so <- Seurat::CreateSeuratObject(counts=M, min.cells=0, min.features=0)
    so$cell_type <- labels
    so <- Seurat::NormalizeData(so, verbose=FALSE)
    so <- Seurat::FindVariableFeatures(so, nfeatures=min(2000, nrow(M)), verbose=FALSE)
    so <- Seurat::ScaleData(so, features=rownames(M), verbose=FALSE)
    npcs <- min(args$dims, ncol(M) - 1L, nrow(M) - 1L)
    so <- Seurat::RunPCA(so, npcs=npcs, features=rownames(M), verbose=FALSE)
    so
  }
  run_dir <- function(refM, queryM, refLabels, queryLabels, direction) {
    keep <- !is.na(refLabels) & refLabels != ""
    ref <- make_obj(refM[, keep, drop=FALSE], refLabels[keep])
    query <- make_obj(queryM, queryLabels)
    dims_use <- seq_len(min(args$dims, ncol(Seurat::Embeddings(ref, "pca")), ncol(Seurat::Embeddings(query, "pca"))))
    anchors <- Seurat::FindTransferAnchors(reference=ref, query=query, features=rownames(refM), dims=dims_use, reference.reduction="pca", k.anchor=args$k_anchor, k.filter=args$k_filter, k.score=args$k_score, verbose=FALSE)
    pred <- Seurat::TransferData(anchorset=anchors, refdata=ref$cell_type, dims=dims_use, k.weight=args$k_weight, verbose=FALSE)
    pred_lab <- as.character(pred$predicted.id)
    score <- if ("prediction.score.max" %in% colnames(pred)) as.numeric(pred$prediction.score.max) else rep(NA_real_, length(pred_lab))
    standard_pred_df(args$dataset_id, method, direction, colnames(queryM), queryLabels, pred_lab, score)
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
