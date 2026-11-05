
# ============================================================
# ICE + AUCell Benchmarking
# Six datasets | DeepSAS GitHub reproducibility
# ============================================================

library(ICE)
library(AUCell)
library(Seurat)
library(reticulate)
library(Matrix)

# ============================================================
# 1. Configuration
# ============================================================

OUT <- "ICE+AUCell results"
GENE_DIR <- "Senescence Genes predicted by DeepSAS"

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

datasets <- data.frame(
  ID = c("GSE253338", "GSE233431", "PCLS", "GSE190889", "GSE264648", "GSE172410"),
  H5AD = c("GSE253338_P16_bladder_human.h5ad", "GSE233431_Upar_human.h5ad", "PCLS_adata.h5ad", "GSE190889_IPF_human.h5ad", "GSE264648_adata_AD.h5ad", "GSE172410_muscle_human.h5ad"),
  ICE_CellAge = c("ICE_es_adata_p16_CellAge.csv", "ICE_es_GSE233431.csv", "ICE_es_adata_PCLS_cellage.csv", "ICE_es_GSE190889.csv", "ICE_es_adata_AD.csv", "ICE_es_skeletal_human_version.csv"),
  ICE_DeepSAS = c("ICE_es_adata_p16_deepsas.csv", "ICE_es_adata_sub_uPAR_deepsas.csv", "ICE_es_adata_PCLS_deepsas.csv", "ICE_es_adata_Human_IPF_deepsas.csv", "ICE_es_adata_glia_deepsas.csv", "ICE_es_adata_skeletal_muscle_deepsas.csv"),
  AUCell_DeepSAS = c("AUCELL_es_adata_p16_deepsas.csv", "AUCELL_es_adata_uPAR_deepsas.csv", "AUCELL_PCLS_deepsas.csv", "AUCELL_human_IPF_deepsas.csv", "AUCELL_glia_deepsas.csv", "AUCELL_skeletal_muscle_deepsas.csv"),
  stringsAsFactors = FALSE
)

datasets$DeepSAS_Genes <- file.path(GENE_DIR, paste0(datasets$ID, "_table2ByGene.csv"))

# ============================================================
# 2. Validate all input files before starting
# ============================================================

required_files <- c(datasets$H5AD, datasets$DeepSAS_Genes, file.path(OUT, "senescence_marker_list.csv"))
missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) stop(paste("Missing input files:", paste(missing_files, collapse = ", ")))

# ============================================================
# 3. Load CellAge signature
# ============================================================

markers <- read.csv(file.path(OUT, "senescence_marker_list.csv"), stringsAsFactors = FALSE, check.names = FALSE)

stopifnot("CellAge" %in% colnames(markers))

cellage_signature <- unique(trimws(as.character(markers[["CellAge"]])))
cellage_signature <- cellage_signature[!is.na(cellage_signature) & nzchar(cellage_signature)]

# ============================================================
# 4. Initialize Python interface
# ============================================================

ad <- import("anndata", convert = FALSE)
scipy <- import("scipy.sparse", convert = FALSE)

# ============================================================
# 5. Process all six datasets
# ============================================================

for (i in seq_len(nrow(datasets))) {

  DATASET <- datasets$ID[i]

  cat("\n========================================\n")
  cat("Processing:", DATASET, "\n")
  cat("========================================\n")

  # ----------------------------------------------------------
  # Load original h5ad
  # ----------------------------------------------------------

  adata <- ad$read_h5ad(datasets$H5AD[i])

  barcodes <- as.character(py_to_r(adata$obs_names$tolist()))
  genes <- as.character(py_to_r(adata$var_names$tolist()))

  stopifnot(length(barcodes) > 0)
  stopifnot(length(genes) > 0)
  stopifnot(!anyDuplicated(barcodes))
  stopifnot(!anyDuplicated(genes))

  # ----------------------------------------------------------
  # Extract raw counts
  # ----------------------------------------------------------

  if (py_to_r(adata$layers$`__contains__`("counts"))) {
    counts_py <- adata$layers$get("counts")
  } else {
    counts_py <- adata$X
  }

  counts_py <- scipy$csc_matrix(counts_py)$transpose()$tocsc()
  counts <- py_to_r(counts_py)

  if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")

  rownames(counts) <- genes
  colnames(counts) <- barcodes

  stopifnot(nrow(counts) == length(genes))
  stopifnot(ncol(counts) == length(barcodes))
  stopifnot(all(is.finite(counts@x)))
  stopifnot(all(counts@x >= 0))
  stopifnot(all(abs(counts@x - round(counts@x)) < 1e-6))

  cat("Cells:", ncol(counts), "\n")
  cat("Genes:", nrow(counts), "\n")

  # ----------------------------------------------------------
  # Normalize expression using Seurat
  # ----------------------------------------------------------

  seu <- CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0)
  seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)

  expr <- GetAssayData(seu, assay = "RNA", layer = "data")

  # ----------------------------------------------------------
  # Load DeepSAS gene signature
  # ----------------------------------------------------------

  deepsas_table <- read.csv(datasets$DeepSAS_Genes[i], stringsAsFactors = FALSE, check.names = FALSE)

  stopifnot("gene" %in% colnames(deepsas_table))

  deepsas_signature <- unique(trimws(as.character(deepsas_table[["gene"]])))
  deepsas_signature <- deepsas_signature[!is.na(deepsas_signature) & nzchar(deepsas_signature)]

  cellage_genes <- base::intersect(cellage_signature, rownames(expr))
  deepsas_genes <- base::intersect(deepsas_signature, rownames(expr))

  if (length(cellage_genes) == 0) stop(paste("No matching CellAge genes:", DATASET))
  if (length(deepsas_genes) == 0) stop(paste("No matching DeepSAS genes:", DATASET))

  cat("CellAge genes matched:", length(cellage_genes), "\n")
  cat("DeepSAS genes matched:", length(deepsas_genes), "\n")

  # ==========================================================
  # 6. ICE:CellAge
  # ==========================================================

  normalized_data <- as.matrix(expr)

  ice_cellage <- ICE(normalized_data, list(CellAge = cellage_genes), iteration = TRUE)
  cellage_scores <- ice_cellage[[1]]

  if (is.null(dim(cellage_scores))) {
    if (length(cellage_scores) != length(barcodes)) stop("ICE CellAge score length mismatch")
    if (is.null(names(cellage_scores))) stop("ICE CellAge scores have no barcode identifiers")
    if (!setequal(names(cellage_scores), barcodes)) stop("ICE CellAge barcode mismatch")
    cellage_values <- as.numeric(cellage_scores[barcodes])
  } else {
    cellage_scores <- as.data.frame(cellage_scores, check.names = FALSE)

    if ("CellAge" %in% colnames(cellage_scores) && all(barcodes %in% rownames(cellage_scores))) {
      cellage_values <- as.numeric(cellage_scores[barcodes, "CellAge"])
    } else if ("CellAge" %in% rownames(cellage_scores) && all(barcodes %in% colnames(cellage_scores))) {
      cellage_values <- as.numeric(cellage_scores["CellAge", barcodes])
    } else {
      stop("ICE CellAge output cannot be aligned to original AnnData barcodes")
    }
  }

  if (!all(is.finite(cellage_values))) stop("ICE CellAge contains invalid scores")

  cellage_output <- data.frame(CellAge = cellage_values, row.names = barcodes, check.names = FALSE)

  # ==========================================================
  # 7. ICE:DeepSAS
  # ==========================================================

  ice_deepsas <- ICE(normalized_data, list(DeepSAS_gene = deepsas_genes), iteration = TRUE)
  deepsas_scores <- ice_deepsas[[1]]

  if (is.null(dim(deepsas_scores))) {
    if (length(deepsas_scores) != length(barcodes)) stop("ICE DeepSAS score length mismatch")
    if (is.null(names(deepsas_scores))) stop("ICE DeepSAS scores have no barcode identifiers")
    if (!setequal(names(deepsas_scores), barcodes)) stop("ICE DeepSAS barcode mismatch")
    deepsas_values <- as.numeric(deepsas_scores[barcodes])
  } else {
    deepsas_scores <- as.data.frame(deepsas_scores, check.names = FALSE)

    if ("DeepSAS_gene" %in% colnames(deepsas_scores) && all(barcodes %in% rownames(deepsas_scores))) {
      deepsas_values <- as.numeric(deepsas_scores[barcodes, "DeepSAS_gene"])
    } else if ("DeepSAS_gene" %in% rownames(deepsas_scores) && all(barcodes %in% colnames(deepsas_scores))) {
      deepsas_values <- as.numeric(deepsas_scores["DeepSAS_gene", barcodes])
    } else {
      stop("ICE DeepSAS output cannot be aligned to original AnnData barcodes")
    }
  }

  if (!all(is.finite(deepsas_values))) stop("ICE DeepSAS contains invalid scores")

  deepsas_output <- data.frame(DeepSAS_gene = deepsas_values, row.names = barcodes, check.names = FALSE)

  rm(normalized_data, ice_cellage, ice_deepsas, cellage_scores, deepsas_scores)
  gc()

  # ==========================================================
  # 8. AUCell:DeepSAS only
  # ==========================================================

  rankings <- AUCell_buildRankings(expr, plotStats = FALSE)
  auc <- AUCell_calcAUC(list(DeepSAS_gene = deepsas_genes), rankings)

  auc_matrix <- getAUC(auc)

  if (!"DeepSAS_gene" %in% rownames(auc_matrix)) stop("AUCell DeepSAS signature missing")
  if (!all(barcodes %in% colnames(auc_matrix))) stop("AUCell barcode mismatch")

  auc_values <- as.numeric(auc_matrix["DeepSAS_gene", barcodes])

  if (!all(is.finite(auc_values))) stop("AUCell contains invalid scores")

  auc_output <- data.frame("seu$DeepSAS_AUCell" = auc_values, row.names = barcodes, check.names = FALSE)

  # ==========================================================
  # 9. Validate results before writing
  # ==========================================================

  stopifnot(identical(rownames(cellage_output), barcodes))
  stopifnot(identical(rownames(deepsas_output), barcodes))
  stopifnot(identical(rownames(auc_output), barcodes))

  stopifnot(identical(colnames(cellage_output), "CellAge"))
  stopifnot(identical(colnames(deepsas_output), "DeepSAS_gene"))
  stopifnot(identical(colnames(auc_output), "seu$DeepSAS_AUCell"))

  # ==========================================================
  # 10. Save exact Python-compatible outputs
  # ==========================================================

  write.csv(cellage_output, file.path(OUT, datasets$ICE_CellAge[i]))
  write.csv(deepsas_output, file.path(OUT, datasets$ICE_DeepSAS[i]))
  write.csv(auc_output, file.path(OUT, datasets$AUCell_DeepSAS[i]))

  # ==========================================================
  # 11. Verify saved CSV files
  # ==========================================================

  files_to_check <- c(datasets$ICE_CellAge[i], datasets$ICE_DeepSAS[i], datasets$AUCell_DeepSAS[i])
  columns_to_check <- c("CellAge", "DeepSAS_gene", "seu$DeepSAS_AUCell")

  for (j in seq_along(files_to_check)) {

    result <- read.csv(file.path(OUT, files_to_check[j]), row.names = 1, check.names = FALSE)

    if (!identical(colnames(result), columns_to_check[j])) stop(paste("Column mismatch:", files_to_check[j]))
    if (!identical(rownames(result), barcodes)) stop(paste("Barcode mismatch:", files_to_check[j]))
    if (!all(is.finite(result[[1]]))) stop(paste("Invalid scores:", files_to_check[j]))

    cat("Verified:", files_to_check[j], "\n")
  }

  rm(adata, counts, counts_py, seu, expr, rankings, auc, auc_matrix, auc_values, auc_output, cellage_output, deepsas_output)
  gc()

  cat("COMPLETED:", DATASET, "\n")
}

cat("\nALL SIX DATASETS COMPLETED SUCCESSFULLY\n")
