#!/usr/bin/env Rscript
# Shared R utilities for final label-transfer benchmark wrappers.

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x) || x == "") y else x

read_names <- function(path, n = NULL, prefix = "id") {
  if (!is.null(path) && !is.na(path) && nzchar(path) && file.exists(path)) {
    vals <- data.table::fread(path, header = FALSE, sep = "\t", colClasses = "character", data.table = FALSE)[[1]]
    return(as.character(vals))
  }
  if (is.null(n)) return(NULL)
  sprintf("%s_%06d", prefix, seq_len(n))
}

auto_sidecars <- function(matrix_path) {
  d <- dirname(matrix_path)
  b <- basename(matrix_path)
  stem <- sub("\\.gz$", "", b)
  stem <- sub("\\.mtx$", "", stem)
  stem <- sub("\\.npz$", "", stem)
  list(
    barcodes = file.path(d, paste0(stem, "_barcodes.tsv")),
    features = file.path(d, paste0(stem, "_features.tsv"))
  )
}

read_mtx_genes_by_cells <- function(path, barcodes = NULL, features = NULL, orientation = "cell_by_feature", name = "matrix") {
  if (!file.exists(path)) stop(sprintf("missing matrix: %s", path))
  sc <- auto_sidecars(path)
  if (is.null(barcodes) || !nzchar(barcodes)) barcodes <- if (file.exists(sc$barcodes)) sc$barcodes else NULL
  if (is.null(features) || !nzchar(features)) features <- if (file.exists(sc$features)) sc$features else NULL
  if (grepl("\\.mtx(\\.gz)?$", path)) {
    M <- Matrix::readMM(path)
  } else if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    M <- readRDS(path); if (is.data.frame(M)) M <- as.matrix(M)
  } else {
    stop(sprintf("R tool wrappers expect .mtx/.mtx.gz or .rds for %s: %s", name, path))
  }
  if (!inherits(M, "Matrix")) M <- Matrix::Matrix(M, sparse = TRUE)
  M <- as(M, "dgCMatrix")
  if (orientation == "cell_by_feature") {
    cell_ids <- if (!is.null(barcodes) && nzchar(barcodes)) read_names(barcodes, nrow(M), "cell") else if (!is.null(rownames(M))) rownames(M) else read_names(NULL, nrow(M), "cell")
    feature_ids <- if (!is.null(features) && nzchar(features)) read_names(features, ncol(M), "feature") else if (!is.null(colnames(M))) colnames(M) else read_names(NULL, ncol(M), "feature")
    M <- Matrix::t(M)
  } else if (orientation == "feature_by_cell") {
    feature_ids <- if (!is.null(features) && nzchar(features)) read_names(features, nrow(M), "feature") else if (!is.null(rownames(M))) rownames(M) else read_names(NULL, nrow(M), "feature")
    cell_ids <- if (!is.null(barcodes) && nzchar(barcodes)) read_names(barcodes, ncol(M), "cell") else if (!is.null(colnames(M))) colnames(M) else read_names(NULL, ncol(M), "cell")
  } else { stop("orientation must be cell_by_feature or feature_by_cell") }
  rownames(M) <- make.unique(as.character(feature_ids)); colnames(M) <- make.unique(as.character(cell_ids))
  list(M = M, cell_ids = as.character(cell_ids), feature_ids = as.character(feature_ids))
}

strict_quit_status <- function() { if (Sys.getenv("LT_REQUIRE_ALL_METHODS", "0") == "1") 1L else 0L }

load_labels <- function(path, id_col = "cell_id", label_col = "cell_type") {
  df <- data.table::fread(path, data.table = FALSE)
  if (!id_col %in% colnames(df)) {
    cand <- intersect(c("cell_id", "barcode", "barcodes", "cell", "id"), colnames(df))
    id_col <- if (length(cand)) cand[[1]] else colnames(df)[[1]]
  }
  if (!label_col %in% colnames(df)) {
    cand <- intersect(c("cell_type", "celltype", "label", "labels", "annotation"), colnames(df))
    if (!length(cand)) stop(sprintf("could not find label column in %s", path))
    label_col <- cand[[1]]
  }
  out <- data.frame(cell_id = as.character(df[[id_col]]), true_label = as.character(df[[label_col]]), stringsAsFactors = FALSE)
  out <- out[!duplicated(out$cell_id), , drop = FALSE]
  out
}

align_labels <- function(cell_ids, labels_df) {
  m <- labels_df$true_label
  names(m) <- labels_df$cell_id
  as.character(m[as.character(cell_ids)])
}

prepare_common_features <- function(rec1, rec2, min_shared_features = 20, allow_index_if_equal = TRUE) {
  shared <- intersect(rownames(rec1$M), rownames(rec2$M))
  if (length(shared) >= min_shared_features) {
    return(list(M1 = rec1$M[shared, , drop = FALSE], M2 = rec2$M[shared, , drop = FALSE], mode = "feature_id", n_shared = length(shared)))
  }
  if (allow_index_if_equal && nrow(rec1$M) == nrow(rec2$M) && nrow(rec1$M) >= min_shared_features) {
    n <- nrow(rec1$M)
    rn <- sprintf("feature_%06d", seq_len(n))
    M1 <- rec1$M; M2 <- rec2$M
    rownames(M1) <- rn; rownames(M2) <- rn
    return(list(M1 = M1, M2 = M2, mode = "feature_index", n_shared = n))
  }
  stop(sprintf("No comparable feature space: shared_features=%d, nrow1=%d, nrow2=%d. Provide gene activity / shared features for this method.", length(shared), nrow(rec1$M), nrow(rec2$M)))
}

normalize_counts <- function(M) {
  cs <- Matrix::colSums(M)
  cs[cs <= 0] <- 1
  M <- Matrix::t(Matrix::t(M) / cs) * 1e4
  M@x <- log1p(M@x)
  M
}

write_status <- function(out_dir, method, dataset_id, status, message) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  status_df <- data.frame(dataset_id = dataset_id, method = method, status = status, message = message, stringsAsFactors = FALSE)
  data.table::fwrite(status_df, file.path(out_dir, "method_status.csv"))
  writeLines(jsonlite::toJSON(as.list(status_df[1,]), auto_unbox = TRUE, pretty = TRUE), file.path(out_dir, "method_status.json"))
}

standard_pred_df <- function(dataset_id, method, direction, target_ids, true_labels, pred_labels, scores = NA, status = NULL) {
  if (is.null(status)) {
    status <- ifelse(is.na(pred_labels) | pred_labels == "" | pred_labels %in% c("unassigned", "unclassified", "ambiguous"), "unclassified", "accepted")
  }
  data.frame(
    dataset_id = dataset_id,
    method = method,
    direction = direction,
    target_cell_id = as.character(target_ids),
    true_label = as.character(true_labels),
    predicted_label = as.character(pred_labels),
    prediction_score = as.numeric(scores),
    transfer_status = as.character(status),
    stringsAsFactors = FALSE
  )
}
