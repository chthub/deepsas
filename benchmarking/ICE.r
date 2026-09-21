library(ICE)
library(reticulate)

library(Seurat)


ad <- import("anndata", convert = FALSE)
np <- import("numpy", convert = FALSE)
py_config()

py_run_string("
import magic
print(magic.__version__)
")
scipy_sparse <- import("scipy.sparse", convert = FALSE)

markers <- read.csv(
  "/fs/ess/PCON0022/Ahmed/ATAC2HiC/PBMC/scATAC-seq/PBMC_Hg19/Model/sencell-deepsas-v1/senescence_marker_list.csv",##to get CellAGE
  stringsAsFactors = FALSE
)

SeuratObject <- NormalizeData(
  SeuratObject,
  normalization.method = "LogNormalize",
  scale.factor = 10000
)


markers_list <- lapply(markers, function(x) na.omit(unique(x)))
markers_list[3]##cellage
normalized_data<-as.matrix(SeuratObject@assays$RNA$data)
class(normalized_data)
dim(normalized_data)

deepsas_gene<-read.csv('PCLS_Gene_table2_all_batches.csv')
head(deepsas_gene)
deepsas_gene<-list('DeepSAS_gene'=deepsas_gene$gene)
library(AUCell)

genes <- intersect(deepsas_gene$DeepSAS_gene, rownames(SeuratObject))
expr <- GetAssayData(SeuratObject, assay="RNA", layer="data")

rankings <- AUCell_buildRankings(expr, plotStats=FALSE)
auc <- AUCell_calcAUC(list(DeepSAS_gene=genes), rankings)

SeuratObject$DeepSAS_AUCell <- as.numeric(getAUC(auc)["DeepSAS_gene", ])

ICE_result_cellage <- ICE(normalized_data, markers_list[3], iteration = TRUE)
ICE_result_deepsas <- ICE(normalized_data, deepsas_gene, iteration = TRUE)

ICE_es_deepsas <- ICE_result_deepsas[[1]]
ICE_es_cellage <- ICE_result_cellage[[1]]

write.csv(ICE_es_deepsas,'ICE_es_adata_PCLS_DeepSAS.csv')####for ICE:DeepSAS
write.csv(ICE_es_cellage,'ICE_es_adata_PCLS_cellage.csv')####for ICE:CellAge
write.csv(as.data.frame(SeuratObject$DeepSAS_AUCell),'AUCELL_adata_PCLS_deepsas.csv')##AUCell:DeepSAS

