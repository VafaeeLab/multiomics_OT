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
  make_option("--prune", action="store_true", default=TRUE),
  make_option("--no_prune", action="store_false", dest="prune"),
  make_option("--de_n", type="integer", default=500),
  make_option("--de_method", type="character", default="wilcox"),
  make_option("--fine_tune", action="store_true", default=TRUE),
  make_option("--no_fine_tune", action="store_false", dest="fine_tune")
)
args <- parse_args(OptionParser(option_list=option_list))
method <- "label_transfer_SingleR"
dir.create(args$out_dir, recursive=TRUE, showWarnings=FALSE)

if (!requireNamespace("SingleR", quietly=TRUE) || !requireNamespace("SummarizedExperiment", quietly=TRUE)) {
  write_status(args$out_dir, method, args$dataset_id, "skipped_missing_package", "Install Bioconductor packages SingleR and SummarizedExperiment.")
  quit(save="no", status=strict_quit_status())
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
  keep1 <- !is.na(y1) & y1 != "" & y1 != "NA"
  keep2 <- !is.na(y2) & y2 != "" & y2 != "NA"
  SE <- SummarizedExperiment::SummarizedExperiment

  run_dir <- function(refM, queryM, refLabels, queryLabels, direction) {
    ref <- SE(assays=list(logcounts=refM[, !is.na(refLabels) & refLabels != "", drop=FALSE]))
    labs <- refLabels[!is.na(refLabels) & refLabels != ""]
    test <- SE(assays=list(logcounts=queryM))
    pred <- SingleR::SingleR(test=test, ref=ref, labels=labs, de.n=args$de_n, de.method=args$de_method, fine.tune=args$fine_tune)
    lab <- as.character(pred$labels)
    if (args$prune && "pruned.labels" %in% colnames(as.data.frame(pred))) {
      lab <- as.character(pred$pruned.labels)
    } else if (args$prune) {
      pr <- SingleR::pruneScores(pred)
      if (!is.null(pr)) lab[is.na(pr)] <- NA
    }
    score <- rep(NA_real_, length(lab))
    if ("delta.next" %in% colnames(as.data.frame(pred))) score <- as.numeric(pred$delta.next)
    else if (!is.null(pred$scores)) score <- apply(as.matrix(pred$scores), 1, max, na.rm=TRUE)
    standard_pred_df(args$dataset_id, method, direction, colnames(queryM), queryLabels, lab, score)
  }

  out1 <- run_dir(M1, M2, y1, y2, "mod1_to_mod2")
  out2 <- run_dir(M2, M1, y2, y1, "mod2_to_mod1")
  out <- rbind(out1, out2)
  data.table::fwrite(out, file.path(args$out_dir, "standardized_predictions_raw.csv"))
  data.table::fwrite(data.frame(feature_alignment=common$mode, n_shared_features=common$n_shared), file.path(args$out_dir, "feature_alignment.csv"))
}, error=function(e) {
  msg <- conditionMessage(e)
  na <- grepl("No comparable feature space", msg, fixed=TRUE)
  write_status(args$out_dir, method, args$dataset_id, if (na) "skipped_not_applicable" else "skipped_error", msg)
  quit(save="no", status=if (na) 0L else strict_quit_status())
})
