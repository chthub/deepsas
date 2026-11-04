
# conda activate deepscence

# Run from the directory containing the input datasets.
# Outputs: benchmark_scores/*.csv; original h5ad files are never saved.

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("benchmark_scores")
OUT.mkdir(exist_ok=True)
from DeepScence.api import DeepScence
from dca.api import dca

# ===== GSE253338 =====
adata = sc.read_h5ad('GSE253338_P16_bladder_human.h5ad')
assert adata.obs_names.is_unique, "Duplicate barcodes: GSE253338"
adata.obs["b"] = adata.obs['clusters']
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")
assert adata.obs_names.is_unique
adata.obs[["ds"]].rename_axis("barcode").to_csv(OUT / "GSE253338_DeepScence.csv")
print("GSE253338: DeepScence saved", adata.obs["ds"].notna().sum())

# ===== GSE233431 =====
adata = sc.read_h5ad('GSE233431_Upar_human.h5ad')
assert adata.obs_names.is_unique, "Duplicate barcodes: GSE233431"
adata.obs["b"] = adata.obs['clusters']
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")
assert adata.obs_names.is_unique
adata.obs[["ds"]].rename_axis("barcode").to_csv(OUT / "GSE233431_DeepScence.csv")
print("GSE233431: DeepScence saved", adata.obs["ds"].notna().sum())

# ===== PCLS =====
adata = sc.read_h5ad('PCLS_adata.h5ad')
assert adata.obs_names.is_unique, "Duplicate barcodes: PCLS"
sc.pp.filter_genes(adata, min_cells=3)
adata.obs["b"] = adata.obs['Clusters']
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")
assert adata.obs_names.is_unique
adata.obs[["ds"]].rename_axis("barcode").to_csv(OUT / "PCLS_DeepScence.csv")
print("PCLS: DeepScence saved", adata.obs["ds"].notna().sum())

# ===== GSE190889 =====
adata = sc.read_h5ad('GSE190889_IPF_human.h5ad')
assert adata.obs_names.is_unique, "Duplicate barcodes: GSE190889"
adata.obs["b"] = adata.obs['clusters']
sc.pp.filter_genes(adata, min_cells=3)
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")
assert adata.obs_names.is_unique
adata.obs[["ds"]].rename_axis("barcode").to_csv(OUT / "GSE190889_DeepScence.csv")
print("GSE190889: DeepScence saved", adata.obs["ds"].notna().sum())

# ===== GSE264648 =====
adata = sc.read_h5ad('GSE264648_adata_AD.h5ad')
assert adata.obs_names.is_unique, "Duplicate barcodes: GSE264648"
adata.obs["b"] = adata.obs['clusters']
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")
assert adata.obs_names.is_unique
adata.obs[["ds"]].rename_axis("barcode").to_csv(OUT / "GSE264648_DeepScence.csv")
print("GSE264648: DeepScence saved", adata.obs["ds"].notna().sum())

# ===== GSE172410 =====
adata = sc.read_h5ad('GSE172410_muscle_human.h5ad')
assert adata.obs_names.is_unique, "Duplicate barcodes: GSE172410"
adata.obs["b"] = adata.obs['clusters']
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")
assert adata.obs_names.is_unique
adata.obs[["ds"]].rename_axis("barcode").to_csv(OUT / "GSE172410_DeepScence.csv")
print("GSE172410: DeepScence saved", adata.obs["ds"].notna().sum())
