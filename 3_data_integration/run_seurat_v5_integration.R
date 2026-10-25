#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(optparse)})
source(file.path(Sys.getenv("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration"), "3_data_integration", "standard_integration_utils.R"))
suppressPackageStartupMessages({library(Seurat)})
options(future.globals.maxSize = 32 * 1024^3)  # Seurat integration exports the full object as a future global; lift the 500 MiB default cap (sequential plan => no cross-worker copy)
opts <- list(
 make_option("--prepared_dir", type="character"), make_option("--outdir", type="character"), make_option("--dataset_id", type="character"),
 make_option("--dims", type="integer", default=30), make_option("--variable_features", type="integer", default=3000),
 make_option("--k_anchor", type="integer", default=5), make_option("--k_filter", type="integer", default=200), make_option("--k_score", type="integer", default=30), make_option("--k_weight", type="integer", default=50),
 make_option("--reduction", type="character", default="cca"),
 make_option("--seed", type="integer", default=42)
)
args <- parse_args(OptionParser(option_list=opts)); set.seed(args$seed)
b <- read_bundle(args$prepared_dir, "combined_raw")
X <- Matrix::t(b$X); rownames(X) <- make.unique(b$features); colnames(X) <- paste0(b$obs$modality, "__", b$obs$cell_id)
meta <- b$obs; rownames(meta) <- colnames(X)
so <- CreateSeuratObject(counts=X, meta.data=meta, min.cells=0, min.features=0)
objs <- SplitObject(so, split.by="modality")
# Per-modality normalization, following each assay's documented method.
# mod1 is RNA -> LogNormalize. mod2 is ADT (CITE) -> CLR margin=2 per Seurat's
# CITE-seq vignette. mod2 of a methylome dataset is already a bounded activity
# score, so it is left as-is rather than log-normalized.
.mod2_kind <- if (grepl("RNAmC|scnmt|snmc2t", args$dataset_id)) "activity" else
              if (grepl("MULTIOME", args$dataset_id)) "lognorm" else "adt"
objs <- lapply(names(objs), function(nm) {
  o <- objs[[nm]]
  if (nm == "mod2" && .mod2_kind == "adt") {
    o <- NormalizeData(o, normalization.method="CLR", margin=2, verbose=FALSE)
  } else if (nm == "mod2" && .mod2_kind == "activity") {
    o <- SetAssayData(o, layer="data", new.data=GetAssayData(o, layer="counts"))
  } else {
    o <- NormalizeData(o, verbose=FALSE)
  }
  o <- FindVariableFeatures(o, nfeatures=min(args$variable_features, nrow(o)), verbose=FALSE)
  o
})
names(objs) <- c("mod1","mod2")[seq_along(objs)]
features <- SelectIntegrationFeatures(object.list=objs, nfeatures=min(args$variable_features, nrow(so)))
# Seurat v5-compatible anchor workflow; fallback to PCA if anchors fail.
Z <- NULL
status <- "integrated_anchors"

# Seurat's RunCCA -> Standardize is compiled C++ and can SIGSEGV on small or
# ill-conditioned inputs (e.g. cross-modality CITE/MULTIOME where the two
# "batches" barely share features). A segfault is NOT catchable by tryCatch, so
# we run the anchor workflow in a forked child. If the child crashes or errors,
# the parent survives and falls back to a plain joint-PCA embedding.
run_anchor_integration <- function() {
  objs <- lapply(objs, function(o) { o <- ScaleData(o, features=features, verbose=FALSE); o <- RunPCA(o, features=features, npcs=args$dims, verbose=FALSE); o })
  anchors <- FindIntegrationAnchors(object.list=objs, anchor.features=features, dims=1:args$dims, reduction=args$reduction, k.anchor=args$k_anchor, k.filter=min(args$k_filter, ncol(so)-1L), k.score=args$k_score, verbose=FALSE)
  integ <- IntegrateData(anchorset=anchors, dims=1:args$dims, k.weight=min(args$k_weight, ncol(so)-1L), verbose=FALSE)
  DefaultAssay(integ) <- "integrated"
  integ <- ScaleData(integ, verbose=FALSE)
  integ <- RunPCA(integ, npcs=args$dims, verbose=FALSE)
  Embeddings(integ, "pca")[colnames(so), , drop=FALSE]
}

anchor_ok <- FALSE
if (requireNamespace("parallel", quietly=TRUE)) {
  job <- parallel::mcparallel(run_anchor_integration())
  res <- parallel::mccollect(job, wait=TRUE)
  val <- if (is.list(res) && length(res) >= 1L) res[[1L]] else NULL
  if (is.matrix(val) || is.data.frame(val)) {
    Z <- as.matrix(val); anchor_ok <- TRUE
  } else if (inherits(val, "try-error")) {
    status <- paste0("pca_fallback: ", trimws(as.character(val)))
  } else {
    status <- "pca_fallback: anchor subprocess died (likely SIGSEGV in RunCCA/Standardize)"
  }
} else {
  res <- try(run_anchor_integration(), silent=TRUE)
  if (!inherits(res, "try-error")) { Z <- as.matrix(res); anchor_ok <- TRUE } else status <- paste0("pca_fallback: ", trimws(as.character(res)))
}

if (!anchor_ok) {
  so2 <- NormalizeData(so, verbose=FALSE); so2 <- FindVariableFeatures(so2, nfeatures=min(args$variable_features, nrow(so2)), verbose=FALSE); so2 <- ScaleData(so2, verbose=FALSE); so2 <- RunPCA(so2, npcs=args$dims, verbose=FALSE)
  Z <- Embeddings(so2, "pca")
}
write_embedding_csv(Z[, seq_len(min(args$dims, ncol(Z))), drop=FALSE], b$obs, args$outdir, args$dataset_id)
writeLines(jsonlite::toJSON(list(status=status, args=as.list(args)), auto_unbox=TRUE, pretty=TRUE), file.path(args$outdir,"method_status.json"))
