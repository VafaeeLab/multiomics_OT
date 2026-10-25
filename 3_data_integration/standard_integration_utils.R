suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
})
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration")
R43_LIB <- Sys.getenv("R43_LIB", file.path(PROJECT_ROOT, "R_analysis/R_libs_4.3"))
dir.create(R43_LIB, recursive=TRUE, showWarnings=FALSE)
.libPaths(unique(c(R43_LIB, "/apps/R/4.3.1/lib64/R/library", .libPaths())))

read_bundle <- function(prepared_dir, prefix="combined_raw") {
  X <- Matrix::readMM(file.path(prepared_dir, paste0(prefix, ".mtx")))
  X <- as(X, "CsparseMatrix")
  obs <- data.table::fread(file.path(prepared_dir, paste0(prefix, "_obs.csv")), data.table=FALSE)
  feats <- data.table::fread(file.path(prepared_dir, paste0(prefix, "_features.tsv")), header=FALSE, data.table=FALSE)[[1]]
  list(X=X, obs=obs, features=as.character(feats))
}

log_norm <- function(X, scale_factor=1e4) {
  X <- as(X, "dgCMatrix")
  rs <- Matrix::rowSums(X); rs[rs <= 0] <- 1
  Y <- Diagonal(x=scale_factor/rs) %*% X
  Y@x <- log1p(Y@x)
  as(Y, "dgCMatrix")
}

write_embedding_csv <- function(Z, obs, outdir, dataset_id) {
  dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
  k <- ncol(Z)
  emb <- as.data.frame(Z)
  colnames(emb) <- paste0("emb_", seq_len(k)-1L)
  out <- cbind(obs[, c("cell_id", "modality"), drop=FALSE], emb)
  data.table::fwrite(out, file.path(outdir, "embeddings.csv"))
}


# Standardise each modality separately (z-score per feature within modality).
# Needed because modalities are normalised by different documented methods
# (RNA: library+log1p, non-negative; ADT: CLR, zero-centred), which leaves the
# two blocks offset from each other inside the shared feature matrix.
permod_scale <- function(X, modality) {
  X <- as.matrix(X)
  for (m in unique(modality)) {
    idx <- which(modality == m)
    if (length(idx) < 2) next
    B <- X[idx, , drop=FALSE]
    mu <- colMeans(B)
    sdv <- apply(B, 2, stats::sd); sdv[!is.finite(sdv) | sdv < 1e-9] <- 1
    X[idx, ] <- sweep(sweep(B, 2, mu, "-"), 2, sdv, "/")
  }
  X[!is.finite(X)] <- 0
  X
}
