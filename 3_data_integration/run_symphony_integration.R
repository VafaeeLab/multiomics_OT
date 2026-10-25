#!/usr/bin/env Rscript
# Symphony integration: build a Symphony reference on mod1 cells, map mod2 cells into it, and write
# the joint harmonised embedding (all cells in the shared space) as embeddings.csv — same schema as
# the other integration runners.
suppressPackageStartupMessages({library(optparse)})
source(file.path(Sys.getenv("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration"),
                 "3_data_integration", "standard_integration_utils.R"))
suppressPackageStartupMessages({library(symphony); library(Matrix)})
opts <- list(make_option("--prepared_dir", type="character"), make_option("--outdir", type="character"),
             make_option("--dataset_id", type="character"), make_option("--dims", type="integer", default=30),
             make_option("--n_variable_genes", type="integer", default=2000), make_option("--k_clusters", type="integer", default=0),
             make_option("--seed", type="integer", default=42))
args <- parse_args(OptionParser(option_list=opts)); set.seed(args$seed)
dir.create(args$outdir, recursive=TRUE, showWarnings=FALSE)
tryCatch({
  b <- read_bundle(args$prepared_dir, "combined_log")
  Xs <- permod_scale(b$X, as.character(b$obs$modality))   # common scale across modalities
  obs <- b$obs; ncell <- nrow(obs)       # Xs: cells x features (sparse)
  m1 <- which(obs$modality == "mod1"); m2 <- which(obs$modality == "mod2")
  # Drop genes that are constant within either modality: after standardisation
  # they are all-zero columns and Symphony's vst selection turns them into NaN.
  .sd1 <- apply(Xs[m1, , drop=FALSE], 2, stats::sd)
  .sd2 <- apply(Xs[m2, , drop=FALSE], 2, stats::sd)
  .keep <- which(is.finite(.sd1) & is.finite(.sd2) & .sd1 > 1e-8 & .sd2 > 1e-8)
  if (length(.keep) >= 10 && length(.keep) < ncol(Xs)) {
    cat(sprintf("[symphony] keeping %d/%d informative features\n", length(.keep), ncol(Xs)))
    Xs <- Xs[, .keep, drop=FALSE]
  }
  Xs[!is.finite(Xs)] <- 0
  # shift non-negative for vst: symphony's variable-gene selection needs
  # positive values (log10 of the mean); a single global constant moves the
  # origin without changing any relative distance or the block alignment.
  .mn <- min(Xs); if (is.finite(.mn) && .mn < 0) Xs <- Xs - .mn
  refM <- as(Matrix::t(Xs[m1, , drop=FALSE]), "CsparseMatrix")       # features x cells (sparse for symphony)
  queryM <- as(Matrix::t(Xs[m2, , drop=FALSE]), "CsparseMatrix")
  feats <- readLines(file.path(args$prepared_dir, "combined_log_features.tsv"))
  if (length(feats) != nrow(refM)) feats <- as.character(seq_len(nrow(refM)))   # fallback
  rownames(refM) <- make.unique(feats); rownames(queryM) <- rownames(refM)      # symphony selects vargenes by name
  colnames(refM) <- paste0("mod1__", obs$cell_id[m1]); colnames(queryM) <- paste0("mod2__", obs$cell_id[m2])
  ref_ct <- as.character(obs$cell_type[m1]); ref_ct[is.na(ref_ct) | ref_ct == ""] <- "unknown"
  ref_meta <- data.frame(cell_type=ref_ct, batch="reference", row.names=colnames(refM), stringsAsFactors=FALSE)
  query_meta <- data.frame(batch=rep("query", ncol(queryM)), row.names=colnames(queryM), stringsAsFactors=FALSE)
  # top-variance gene selection, replacing Symphony's vst (undefined on centred data)
  if (nrow(refM) > args$n_variable_genes) {
    .v <- apply(Xs[m1, , drop=FALSE], 2, stats::var)
    .v[!is.finite(.v)] <- 0
    .sel <- order(.v, decreasing=TRUE)[seq_len(args$n_variable_genes)]
    .sel <- sort(.sel)
    cat(sprintf("[symphony] top-variance selection: %d of %d genes\n", length(.sel), ncol(Xs)))
    refM <- refM[.sel, , drop=FALSE]; queryM <- queryM[.sel, , drop=FALSE]
  }
  topn <- nrow(refM)
  d <- min(args$dims, topn - 1L, ncol(refM) - 1L)
  Kv <- if (args$k_clusters > 0) min(args$k_clusters, ncol(refM) - 1L) else min(100, max(2, floor(ncol(refM) / 10)))
  reference <- symphony::buildReference(refM, ref_meta, vars=NULL, K=Kv,
                                        verbose=FALSE, do_umap=FALSE, do_normalize=FALSE, topn=topn, d=d)
  query <- symphony::mapQuery(queryM, query_meta, reference, vars=NULL, do_normalize=FALSE, do_umap=FALSE)
  ex <- function(z, n) { z <- as.matrix(z); if (nrow(z) == n) z else if (ncol(z) == n) t(z) else NULL }
  refZ <- NULL; for (nm in c("Z_corr", "Z", "Z_orig")) if (!is.null(reference[[nm]])) { z <- ex(reference[[nm]], ncol(refM)); if (!is.null(z)) { refZ <- z; break } }
  qZ  <- NULL; for (nm in c("Z", "Zq", "Z_corr")) if (!is.null(query[[nm]])) { z <- ex(query[[nm]], ncol(queryM)); if (!is.null(z)) { qZ <- z; break } }
  if (is.null(refZ) || is.null(qZ)) stop("Could not extract Symphony embeddings")
  k <- min(ncol(refZ), ncol(qZ))
  Zfull <- matrix(NA_real_, ncell, k); Zfull[m1, ] <- refZ[, seq_len(k)]; Zfull[m2, ] <- qZ[, seq_len(k)]
  write_embedding_csv(Zfull, obs, args$outdir, args$dataset_id)
  writeLines(jsonlite::toJSON(list(status="ok", dims=k), auto_unbox=TRUE, pretty=TRUE), file.path(args$outdir, "method_status.json"))
}, error=function(e) {
  writeLines(jsonlite::toJSON(list(status="error", message=conditionMessage(e)), auto_unbox=TRUE, pretty=TRUE),
             file.path(args$outdir, "method_status.json"))
  quit(save="no", status=1)
})
