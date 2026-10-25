#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(optparse)})
source(file.path(Sys.getenv("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration"), "3_data_integration", "standard_integration_utils.R"))
suppressPackageStartupMessages({library(harmony); library(irlba)})
opts <- list(make_option("--prepared_dir", type="character"), make_option("--outdir", type="character"), make_option("--dataset_id", type="character"), make_option("--pca_dim", type="integer", default=30), make_option("--emb_dim", type="integer", default=30), make_option("--theta", type="double", default=2.0), make_option("--lambda", type="double", default=1.0), make_option("--sigma", type="double", default=0.1), make_option("--max_iter", type="integer", default=20), make_option("--seed", type="integer", default=42))
args <- parse_args(OptionParser(option_list=opts)); set.seed(args$seed)
b <- read_bundle(args$prepared_dir, "combined_log")
X <- as.matrix(b$X)
# per-modality standardisation (see permod_scale); a single global scale()
# leaves the log1p/CLR offset intact so PC1 encodes modality, not biology.
X <- permod_scale(X, as.character(b$obs$modality))
k <- min(args$pca_dim, nrow(X)-1L, ncol(X)-1L)
sv <- irlba::irlba(X, nv=k)
P <- sv$u %*% diag(sv$d, k, k)
rownames(P) <- paste0(b$obs$modality, "__", b$obs$cell_id)
meta <- b$obs
H <- harmony::RunHarmony(P, meta_data=meta, vars_use="batch_modality", theta=args$theta, lambda=args$lambda, sigma=args$sigma, max_iter=args$max_iter, early_stop=TRUE, verbose=FALSE)
Z <- as.matrix(H)[, seq_len(min(args$emb_dim, ncol(H))), drop=FALSE]
write_embedding_csv(Z, b$obs, args$outdir, args$dataset_id)
writeLines(jsonlite::toJSON(list(status="ok", args=as.list(args)), auto_unbox=TRUE, pretty=TRUE), file.path(args$outdir,"method_status.json"))
