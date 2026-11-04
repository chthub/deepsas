
# conda activate Decoupler
# python AUCell_Benchmarking.py
# Run from the directory containing the input datasets.
# Outputs: benchmark_scores/*.csv

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import decoupler as dc
from sklearn.metrics import roc_auc_score

OUT = Path("benchmark_scores")
OUT.mkdir(exist_ok=True)

markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
net = pd.DataFrame([(col, gene) for col in markers.columns for gene in markers[col].dropna()], columns=["source", "target"]).drop_duplicates()
metrics = ["ds", "senpey_universal", "senpey_specific", "SID_Score", "SnC scores", "CellAge", "GO", "SenMayo", "ICE_cellage", "ICE_deepsas", "aucell_deepsas"]

# ============================================================
# GSE253338: p16 bladder
# ============================================================

adata = sc.read_h5ad("GSE253338_P16_bladder_human.h5ad")
assert adata.obs_names.is_unique, "Duplicate AnnData barcodes"

scores = pd.read_csv(OUT / "GSE253338_DeepScence.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepScence barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepScence barcodes"
assert "ds" in scores.columns
adata.obs["ds"] = pd.to_numeric(scores["ds"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE253338_SenCID.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenCID barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenCID barcodes"
assert "SID_Score" in scores.columns
adata.obs["SID_Score"] = pd.to_numeric(scores["SID_Score"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE253338_SenePy.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenePy barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenePy barcodes"
assert "senpey_universal" in scores.columns
adata.obs["senpey_universal"] = pd.to_numeric(scores["senpey_universal"].reindex(adata.obs_names), errors="raise")
assert "senpey_specific" in scores.columns
adata.obs["senpey_specific"] = pd.to_numeric(scores["senpey_specific"].reindex(adata.obs_names), errors="raise")

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)

assert "CellAge" in adata.obsm["aucell_estimate"].columns
adata.obs["CellAge"] = adata.obsm["aucell_estimate"]["CellAge"].reindex(adata.obs_names)
assert "GO" in adata.obsm["aucell_estimate"].columns
adata.obs["GO"] = adata.obsm["aucell_estimate"]["GO"].reindex(adata.obs_names)
assert "SenMayo" in adata.obsm["aucell_estimate"].columns
adata.obs["SenMayo"] = adata.obsm["aucell_estimate"]["SenMayo"].reindex(adata.obs_names)

# FIX: Read ICE CellAge instead of assigning NaN
scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_p16_CellAge.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE CellAge barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE CellAge barcodes"
assert "CellAge" in scores.columns, "Missing CellAge column"
adata.obs["ICE_cellage"] = pd.to_numeric(scores["CellAge"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_p16_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE DeepSAS barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE DeepSAS barcodes"
assert "DeepSAS_gene" in scores.columns
adata.obs["ICE_deepsas"] = pd.to_numeric(scores["DeepSAS_gene"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/AUCELL_es_adata_p16_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepSAS AUCell barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepSAS AUCell barcodes"
assert "seu$DeepSAS_AUCell" in scores.columns
adata.obs["aucell_deepsas"] = pd.to_numeric(scores["seu$DeepSAS_AUCell"].reindex(adata.obs_names), errors="raise")

adata.obs.index.name = "barcode"
adata.obs.to_csv(OUT / "GSE253338_merged_scores.csv")

# Fibroblast
df = adata.obs.loc[adata.obs["integrated_snn_res.0.1"].astype(str).isin(["0", "3"]), ["orig.ident", *metrics]].copy()
df["label"] = df["orig.ident"].eq("P16_positive").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "GSE253338_fibroblast_AUROC.csv", index=False)
print("GSE253338 / fibroblast: AUROC saved")

# Basal urothelial
df = adata.obs.loc[adata.obs["integrated_snn_res.0.1"].astype(str).eq("1"), ["orig.ident", *metrics]].copy()
df["label"] = df["orig.ident"].eq("P16_positive").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "GSE253338_basal_AUROC.csv", index=False)
print("GSE253338 / basal: AUROC saved")

# ============================================================
# GSE233431: uPAR
# ============================================================

adata = sc.read_h5ad("GSE233431_Upar_human.h5ad")
assert adata.obs_names.is_unique, "Duplicate AnnData barcodes"

scores = pd.read_csv(OUT / "GSE233431_DeepScence.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepScence barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepScence barcodes"
assert "ds" in scores.columns
adata.obs["ds"] = pd.to_numeric(scores["ds"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE233431_SenCID.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenCID barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenCID barcodes"
assert "SID_Score" in scores.columns
adata.obs["SID_Score"] = pd.to_numeric(scores["SID_Score"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE233431_SenePy.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenePy barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenePy barcodes"
assert "senpey_universal" in scores.columns
adata.obs["senpey_universal"] = pd.to_numeric(scores["senpey_universal"].reindex(adata.obs_names), errors="raise")
assert "senpey_specific" in scores.columns
adata.obs["senpey_specific"] = pd.to_numeric(scores["senpey_specific"].reindex(adata.obs_names), errors="raise")

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)

assert "CellAge" in adata.obsm["aucell_estimate"].columns
adata.obs["CellAge"] = adata.obsm["aucell_estimate"]["CellAge"].reindex(adata.obs_names)
assert "GO" in adata.obsm["aucell_estimate"].columns
adata.obs["GO"] = adata.obsm["aucell_estimate"]["GO"].reindex(adata.obs_names)
assert "SenMayo" in adata.obsm["aucell_estimate"].columns
adata.obs["SenMayo"] = adata.obsm["aucell_estimate"]["SenMayo"].reindex(adata.obs_names)

scores = pd.read_csv("ICE+AUCell results/ICE_es_GSE233431.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE CellAge barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE CellAge barcodes"
assert "CellAge" in scores.columns
adata.obs["ICE_cellage"] = pd.to_numeric(scores["CellAge"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_sub_uPAR_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE DeepSAS barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE DeepSAS barcodes"
assert "DeepSAS_gene" in scores.columns
adata.obs["ICE_deepsas"] = pd.to_numeric(scores["DeepSAS_gene"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/AUCELL_es_adata_uPAR_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepSAS AUCell barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepSAS AUCell barcodes"
assert "seu$DeepSAS_AUCell" in scores.columns
adata.obs["aucell_deepsas"] = pd.to_numeric(scores["seu$DeepSAS_AUCell"].reindex(adata.obs_names), errors="raise")

adata.obs.index.name = "barcode"
adata.obs.to_csv(OUT / "GSE233431_merged_scores.csv")

df = adata.obs.loc[adata.obs["seurat_clusters"].astype(str).isin(["0", "1", "3", "7"]) & adata.obs["Sample"].astype(str).str.contains("Male", na=False), ["Sample", *metrics]].copy()
df["label"] = df["Sample"].eq("Male_uPAR_positive").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "GSE233431_uPAR_AUROC.csv", index=False)
print("GSE233431 / uPAR: AUROC saved")

# ============================================================
# PCLS
# ============================================================

adata = sc.read_h5ad("PCLS_adata.h5ad")
assert adata.obs_names.is_unique, "Duplicate AnnData barcodes"

scores = pd.read_csv(OUT / "PCLS_DeepScence.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepScence barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepScence barcodes"
assert "ds" in scores.columns
adata.obs["ds"] = pd.to_numeric(scores["ds"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "PCLS_SenCID.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenCID barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenCID barcodes"
assert "SID_Score" in scores.columns
adata.obs["SID_Score"] = pd.to_numeric(scores["SID_Score"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "PCLS_SenePy.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenePy barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenePy barcodes"
assert "senpey_universal" in scores.columns
adata.obs["senpey_universal"] = pd.to_numeric(scores["senpey_universal"].reindex(adata.obs_names), errors="raise")
assert "senpey_specific" in scores.columns
adata.obs["senpey_specific"] = pd.to_numeric(scores["senpey_specific"].reindex(adata.obs_names), errors="raise")

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)

assert "CellAge" in adata.obsm["aucell_estimate"].columns
adata.obs["CellAge"] = adata.obsm["aucell_estimate"]["CellAge"].reindex(adata.obs_names)
assert "GO" in adata.obsm["aucell_estimate"].columns
adata.obs["GO"] = adata.obsm["aucell_estimate"]["GO"].reindex(adata.obs_names)
assert "SenMayo" in adata.obsm["aucell_estimate"].columns
adata.obs["SenMayo"] = adata.obsm["aucell_estimate"]["SenMayo"].reindex(adata.obs_names)

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_PCLS_cellage.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE CellAge barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE CellAge barcodes"
assert "CellAge" in scores.columns
adata.obs["ICE_cellage"] = pd.to_numeric(scores["CellAge"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_PCLS_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE DeepSAS barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE DeepSAS barcodes"
assert "DeepSAS_gene" in scores.columns
adata.obs["ICE_deepsas"] = pd.to_numeric(scores["DeepSAS_gene"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/AUCELL_PCLS_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepSAS AUCell barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepSAS AUCell barcodes"
assert "seu$DeepSAS_AUCell" in scores.columns
adata.obs["aucell_deepsas"] = pd.to_numeric(scores["seu$DeepSAS_AUCell"].reindex(adata.obs_names), errors="raise")

adata.obs.index.name = "barcode"
adata.obs.to_csv(OUT / "PCLS_merged_scores.csv")

df = adata.obs.loc[adata.obs["Cellinage"].eq("Epithelial"), ["Treatment", *metrics]].copy()
df["label"] = df["Treatment"].eq("Bleomycin").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "PCLS_PCLS_AUROC.csv", index=False)
print("PCLS: AUROC saved")

# ============================================================
# GSE190889: IPF
# ============================================================

adata = sc.read_h5ad("Benchmark Data/GSE190889_IPF_human.h5ad")
assert adata.obs_names.is_unique, "Duplicate AnnData barcodes"

scores = pd.read_csv(OUT / "GSE190889_DeepScence.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepScence barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepScence barcodes"
assert "ds" in scores.columns
adata.obs["ds"] = pd.to_numeric(scores["ds"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE190889_SenCID.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenCID barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenCID barcodes"
assert "SID_Score" in scores.columns
adata.obs["SID_Score"] = pd.to_numeric(scores["SID_Score"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE190889_SenePy.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenePy barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenePy barcodes"
assert "senpey_universal" in scores.columns
adata.obs["senpey_universal"] = pd.to_numeric(scores["senpey_universal"].reindex(adata.obs_names), errors="raise")
assert "senpey_specific" in scores.columns
adata.obs["senpey_specific"] = pd.to_numeric(scores["senpey_specific"].reindex(adata.obs_names), errors="raise")

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)

assert "CellAge" in adata.obsm["aucell_estimate"].columns
adata.obs["CellAge"] = adata.obsm["aucell_estimate"]["CellAge"].reindex(adata.obs_names)
assert "GO" in adata.obsm["aucell_estimate"].columns
adata.obs["GO"] = adata.obsm["aucell_estimate"]["GO"].reindex(adata.obs_names)
assert "SenMayo" in adata.obsm["aucell_estimate"].columns
adata.obs["SenMayo"] = adata.obsm["aucell_estimate"]["SenMayo"].reindex(adata.obs_names)

scores = pd.read_csv("ICE+AUCell results/ICE_es_GSE190889.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE CellAge barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE CellAge barcodes"
assert "CellAge" in scores.columns
adata.obs["ICE_cellage"] = pd.to_numeric(scores["CellAge"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_Human_IPF_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE DeepSAS barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE DeepSAS barcodes"
assert "DeepSAS_gene" in scores.columns
adata.obs["ICE_deepsas"] = pd.to_numeric(scores["DeepSAS_gene"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/AUCELL_human_IPF_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepSAS AUCell barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepSAS AUCell barcodes"
assert "seu$DeepSAS_AUCell" in scores.columns
adata.obs["aucell_deepsas"] = pd.to_numeric(scores["seu$DeepSAS_AUCell"].reindex(adata.obs_names), errors="raise")

adata.obs.index.name = "barcode"
adata.obs.to_csv(OUT / "GSE190889_merged_scores.csv")

df = adata.obs.loc[:, ["health_status", *metrics]].copy()
df["label"] = df["health_status"].eq("IPF").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "GSE190889_IPF_AUROC.csv", index=False)
print("GSE190889 / IPF: AUROC saved")

# ============================================================
# GSE264648: Alzheimer's disease microglia
# ============================================================

adata = sc.read_h5ad("GSE264648_adata_AD.h5ad")
assert adata.obs_names.is_unique, "Duplicate AnnData barcodes"

scores = pd.read_csv(OUT / "GSE264648_DeepScence.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepScence barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepScence barcodes"
assert "ds" in scores.columns
adata.obs["ds"] = pd.to_numeric(scores["ds"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE264648_SenCID.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenCID barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenCID barcodes"
assert "SID_Score" in scores.columns
adata.obs["SID_Score"] = pd.to_numeric(scores["SID_Score"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE264648_SenePy.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenePy barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenePy barcodes"
assert "senpey_universal" in scores.columns
adata.obs["senpey_universal"] = pd.to_numeric(scores["senpey_universal"].reindex(adata.obs_names), errors="raise")
assert "senpey_specific" in scores.columns
adata.obs["senpey_specific"] = pd.to_numeric(scores["senpey_specific"].reindex(adata.obs_names), errors="raise")

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)

assert "CellAge" in adata.obsm["aucell_estimate"].columns
adata.obs["CellAge"] = adata.obsm["aucell_estimate"]["CellAge"].reindex(adata.obs_names)
assert "GO" in adata.obsm["aucell_estimate"].columns
adata.obs["GO"] = adata.obsm["aucell_estimate"]["GO"].reindex(adata.obs_names)
assert "SenMayo" in adata.obsm["aucell_estimate"].columns
adata.obs["SenMayo"] = adata.obsm["aucell_estimate"]["SenMayo"].reindex(adata.obs_names)

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_AD.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE CellAge barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE CellAge barcodes"
assert "CellAge" in scores.columns
adata.obs["ICE_cellage"] = pd.to_numeric(scores["CellAge"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_glia_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE DeepSAS barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE DeepSAS barcodes"
assert "DeepSAS_gene" in scores.columns
adata.obs["ICE_deepsas"] = pd.to_numeric(scores["DeepSAS_gene"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/AUCELL_glia_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepSAS AUCell barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepSAS AUCell barcodes"
assert "seu$DeepSAS_AUCell" in scores.columns
adata.obs["aucell_deepsas"] = pd.to_numeric(scores["seu$DeepSAS_AUCell"].reindex(adata.obs_names), errors="raise")

adata.obs.index.name = "barcode"
adata.obs.to_csv(OUT / "GSE264648_merged_scores.csv")

df = adata.obs.loc[adata.obs["clusters"].eq("Micro") & adata.obs["braak"].astype(str).isin(["0", "6"]), ["diagnosis", *metrics]].copy()
df["label"] = df["diagnosis"].eq("AD").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "GSE264648_Microglia_AUROC.csv", index=False)
print("GSE264648 / Microglia: AUROC saved")

# ============================================================
# GSE172410: skeletal muscle
# ============================================================

adata = sc.read_h5ad("GSE172410_muscle_human.h5ad")
assert adata.obs_names.is_unique, "Duplicate AnnData barcodes"

scores = pd.read_csv(OUT / "GSE172410_DeepScence.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepScence barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepScence barcodes"
assert "ds" in scores.columns
adata.obs["ds"] = pd.to_numeric(scores["ds"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE172410_SenCID.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenCID barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenCID barcodes"
assert "SID_Score" in scores.columns
adata.obs["SID_Score"] = pd.to_numeric(scores["SID_Score"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv(OUT / "GSE172410_SenePy.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate SenePy barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing SenePy barcodes"
assert "senpey_universal" in scores.columns
adata.obs["senpey_universal"] = pd.to_numeric(scores["senpey_universal"].reindex(adata.obs_names), errors="raise")
assert "senpey_specific" in scores.columns
adata.obs["senpey_specific"] = pd.to_numeric(scores["senpey_specific"].reindex(adata.obs_names), errors="raise")

sc.pp.normalize_total(adata, target_sum=1e5)
sc.pp.log1p(adata)
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)

assert "CellAge" in adata.obsm["aucell_estimate"].columns
adata.obs["CellAge"] = adata.obsm["aucell_estimate"]["CellAge"].reindex(adata.obs_names)
assert "GO" in adata.obsm["aucell_estimate"].columns
adata.obs["GO"] = adata.obsm["aucell_estimate"]["GO"].reindex(adata.obs_names)
assert "SenMayo" in adata.obsm["aucell_estimate"].columns
adata.obs["SenMayo"] = adata.obsm["aucell_estimate"]["SenMayo"].reindex(adata.obs_names)

scores = pd.read_csv("ICE+AUCell results/ICE_es_skeletal_human_version.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE CellAge barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE CellAge barcodes"
assert "CellAge" in scores.columns
adata.obs["ICE_cellage"] = pd.to_numeric(scores["CellAge"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/ICE_es_adata_skeletal_muscle_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate ICE DeepSAS barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing ICE DeepSAS barcodes"
assert "DeepSAS_gene" in scores.columns
adata.obs["ICE_deepsas"] = pd.to_numeric(scores["DeepSAS_gene"].reindex(adata.obs_names), errors="raise")

scores = pd.read_csv("ICE+AUCell results/AUCELL_skeletal_muscle_deepsas.csv", index_col=0)
scores.index = scores.index.astype(str)
assert scores.index.is_unique, "Duplicate DeepSAS AUCell barcodes"
assert adata.obs_names.isin(scores.index).all(), "Missing DeepSAS AUCell barcodes"
assert "seu$DeepSAS_AUCell" in scores.columns
adata.obs["aucell_deepsas"] = pd.to_numeric(scores["seu$DeepSAS_AUCell"].reindex(adata.obs_names), errors="raise")

adata.obs.index.name = "barcode"
adata.obs.to_csv(OUT / "GSE172410_merged_scores.csv")

df = adata.obs.loc[adata.obs["clusters"].eq("satellite"), ["age", *metrics]].copy()
df["label"] = df["age"].eq("Old").astype(int)
results = []

for metric in metrics:
    scores = pd.to_numeric(df[metric], errors="coerce")
    valid = scores.notna() & np.isfinite(scores)
    y = df.loc[valid, "label"]
    auc = roc_auc_score(y, scores.loc[valid]) if y.nunique() == 2 else np.nan
    results.append({"metric": metric, "AUROC": auc, "n_cells": int(valid.sum()), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())})

pd.DataFrame(results).to_csv(OUT / "GSE172410_Skeletal_AUROC.csv", index=False)
print("GSE172410 / Skeletal: AUROC saved")

print("All six datasets completed.")
