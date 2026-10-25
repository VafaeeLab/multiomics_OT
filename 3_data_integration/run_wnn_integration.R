#!/usr/bin/env Rscript
# Seurat WNN (weighted nearest neighbor) joint integration of two SAME-CELL
# modalities (CITE: RNA+ADT; multiome: RNA+ATAC-gene-activity).
#
# Reads the integration "combined_raw" bundle, splits by modality, pairs cells
# by cell_id, builds a two-assay Seurat object, runs FindMultiModalNeighbors,
# and writes a joint per-cell embedding to embeddings.csv in the same schema as
# the other integration methods (one row per (cell, modality); the joint
# coordinates are written for both modality rows of each paired cell).
#
# NOTE: WNN requires the two modalities to be measured on the same cells. For
# rna-only datasets it is not applicable and exits as skipped_not_applicable.
suppressPackageStartupMessages({library(optparse)})
source(file.path(Sys.getenv("PROJECT_ROOT", "/g/data/yr31/mp8713/project_data_integration"),
                 "3_data_integration", "standard_integration_utils.R"))
suppressPackageStartupMessages({library(Seurat); library(Matrix)})

opts <- list(
  make_option("--prepared_dir", type="character"),
  make_option("--outdir", type="character"),
  make_option("--dataset_id", type="character"),
  make_option("--dims", type="integer", default=30),
  make_option("--adt_dims", type="integer", default=18),
  make_option("--mod2_norm", type="character", default="CLR"),  # CLR for ADT; LogNormalize for gene-activity
  make_option("--knn", type="integer", default=20),             # k.nn for FindMultiModalNeighbors (WNN graph)
  make_option("--seed", type="integer", default=42)
)
args <- parse_args(OptionParser(option_list=opts)); set.seed(args$seed)

write_status <- function(outdir, status, msg) {
  dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
  writeLines(jsonlite::toJSON(list(method="WNN", dataset_id=args$dataset_id,
             status=status, message=msg), auto_unbox=TRUE, pretty=TRUE),
             file.path(outdir, "method_status.json"))
}

tryCatch({
  b <- read_bundle(args$prepared_dir, "combined_raw")
  obs <- b$obs
  if (!("modality" %in% colnames(obs)) || length(unique(obs$modality)) < 2) {
    write_status(args$outdir, "skipped_not_applicable",
                 "WNN needs two same-cell modalities; only one modality present.")
    quit(save="no", status=0)
  }
  X <- Matrix::t(b$X); rownames(X) <- make.unique(b$features)
  colnames(X) <- paste0(obs$modality, "__", obs$cell_id)

  m1 <- which(obs$modality == "mod1"); m2 <- which(obs$modality == "mod2")
  id1 <- obs$cell_id[m1]; id2 <- obs$cell_id[m2]
  common <- intersect(id1, id2)
  if (length(common) < 10) {
    write_status(args$outdir, "skipped_not_applicable",
                 sprintf("WNN needs paired cells across modalities; only %d shared cell_ids.", length(common)))
    quit(save="no", status=0)
  }
  # Per-modality feature rows = rows with any signal in that modality's cells.
  X1 <- X[, m1, drop=FALSE]; colnames(X1) <- id1
  X2 <- X[, m2, drop=FALSE]; colnames(X2) <- id2
  f1 <- which(Matrix::rowSums(X1 != 0) > 0); f2 <- which(Matrix::rowSums(X2 != 0) > 0)
  A1 <- X1[f1, common, drop=FALSE]; A2 <- X2[f2, common, drop=FALSE]
  # Disambiguate assay rownames so the two assays never collide.
  rownames(A1) <- make.unique(paste0("m1-", rownames(A1)))
  rownames(A2) <- make.unique(paste0("m2-", rownames(A2)))

  so <- CreateSeuratObject(counts=A1, assay="mod1")
  so[["mod2"]] <- CreateAssayObject(counts=A2)

  DefaultAssay(so) <- "mod1"
  so <- NormalizeData(so, verbose=FALSE)
  so <- FindVariableFeatures(so, nfeatures=min(2000, nrow(A1)), verbose=FALSE)
  so <- ScaleData(so, verbose=FALSE)
  so <- RunPCA(so, npcs=args$dims, reduction.name="pca", verbose=FALSE)

  DefaultAssay(so) <- "mod2"
  VariableFeatures(so) <- rownames(so[["mod2"]])
  if (toupper(args$mod2_norm) == "CLR") {
    so <- NormalizeData(so, normalization.method="CLR", margin=2, verbose=FALSE)
  } else {
    so <- NormalizeData(so, verbose=FALSE)
  }
  adt_dims <- max(2L, min(args$adt_dims, nrow(A2) - 1L))
  so <- ScaleData(so, verbose=FALSE)
  so <- RunPCA(so, npcs=adt_dims, reduction.name="apca", verbose=FALSE)

  so <- FindMultiModalNeighbors(so, reduction.list=list("pca", "apca"),
                                dims.list=list(1:args$dims, 1:adt_dims), k.nn=args$knn, verbose=FALSE)
  so <- RunUMAP(so, nn.name="weighted.nn", reduction.name="wnn.umap",
                n.components=args$dims, verbose=FALSE)
  Zc <- Embeddings(so, "wnn.umap")            # paired cells x dims (one row per cell)
  rownames(Zc) <- common

  # Emit in the per-(cell, modality) schema; joint coords copied to both rows.
  build <- function(idx) {
    ids <- obs$cell_id[idx]; mods <- obs$modality[idx]
    keep <- ids %in% common
    list(ids=ids[keep], mods=mods[keep])
  }
  ord <- which(obs$cell_id %in% common)
  emb <- Zc[obs$cell_id[ord], , drop=FALSE]
  colnames(emb) <- paste0("emb_", seq_len(ncol(emb)) - 1L)
  out <- cbind(obs[ord, c("cell_id", "modality"), drop=FALSE], as.data.frame(emb))
  dir.create(args$outdir, recursive=TRUE, showWarnings=FALSE)
  data.table::fwrite(out, file.path(args$outdir, "embeddings.csv"))

  # ---- persist the WNN graph and everything needed to rebuild/upgrade it ----
  # The wsnn/wknn graphs are the upgrade target; the neighbour lists, modality
  # weights and the two reductions are saved because they cannot be recovered
  # from the graph and would otherwise require re-running WNN.
  tryCatch({
    gdir <- args$outdir
    writeLines(as.character(common), file.path(gdir, "wnn_cells.tsv"))

    for (gname in c("wsnn", "wknn")) {
      if (!is.null(so@graphs[[gname]])) {
        g <- as(as(so@graphs[[gname]], "generalMatrix"), "dgCMatrix")
        Matrix::writeMM(g, file.path(gdir, paste0("wnn_", gname, ".mtx")))
      }
    }

    # weighted.nn keeps the ranked neighbour order and distances that the
    # sparse graph discards
    if (!is.null(so@neighbors[["weighted.nn"]])) {
      wnn <- so@neighbors[["weighted.nn"]]
      data.table::fwrite(as.data.frame(Seurat::Indices(wnn)),
                         file.path(gdir, "wnn_nn_idx.csv"))
      data.table::fwrite(as.data.frame(Seurat::Distances(wnn)),
                         file.path(gdir, "wnn_nn_dist.csv"))
    }

    wcols <- grep("\\.weight$", colnames(so@meta.data), value=TRUE)
    if (length(wcols)) {
      wdf <- data.frame(cell_id=rownames(so@meta.data),
                        so@meta.data[, wcols, drop=FALSE])
      data.table::fwrite(wdf, file.path(gdir, "wnn_modality_weights.csv"))
    }

    for (rn in c("pca", "apca")) {
      if (rn %in% names(so@reductions)) {
        E <- Embeddings(so, rn)
        edf <- data.frame(cell_id=rownames(E), as.data.frame(E))
        data.table::fwrite(edf, file.path(gdir,
          if (rn == "pca") "wnn_reduction_mod1.csv" else "wnn_reduction_mod2.csv"))
      }
    }

    for (asy in c("mod1", "mod2")) {
      vf <- tryCatch(VariableFeatures(so, assay=asy), error=function(e) character(0))
      if (length(vf)) writeLines(vf, file.path(gdir, paste0("wnn_hvf_", asy, ".txt")))
    }

    writeLines(jsonlite::toJSON(list(
        dataset_id=args$dataset_id, n_paired=length(common),
        n_mod1_features=nrow(A1), n_mod2_features=nrow(A2),
        dims=args$dims, adt_dims=adt_dims, knn=args$knn,
        mod2_norm=args$mod2_norm, seed=args$seed,
        graphs=names(so@graphs), neighbors=names(so@neighbors),
        reductions=names(so@reductions)),
      auto_unbox=TRUE, pretty=TRUE), file.path(gdir, "wnn_meta.json"))
  }, error=function(e) {
    message(sprintf("[wnn] graph save failed: %s", conditionMessage(e)))
  })

  write_status(args$outdir, "ok", sprintf("WNN joint embedding for %d paired cells.", length(common)))
}, error=function(e) {
  write_status(args$outdir, "skipped_error", conditionMessage(e))
  quit(save="no", status=1)
})
