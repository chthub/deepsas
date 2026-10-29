
# ============================================================
# DeepSAS benchmarking
###the continuous sensescense score predictecetd by DeepSAS are stored in column 'SnC scores' 
# Dataset order:
# 1. GSE253338 - p16 bladder
# 2. GSE233431 - uPAR intestine
# 3. In-house PCLS - lung epithelial
# 4. GSE190889 - human IPF
# 5. GSE264648 - microglia
# 6. GSE172410 - skeletal muscle
# ============================================================

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import senepy as sp
from sklearn.metrics import roc_auc_score
from DeepScence.api import DeepScence
from dca.api import dca
from SenCID.api import SenCID


# ============================================================
# 1. GSE253338 - p16 BLADDER
# ============================================================

adata = sc.read_h5ad("GSE253338_P16_bladder_human.h5ad")

# DeepScence
adata.obs["b"] = adata.obs["clusters"]
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")

# SenCID
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)

# Normalize full expression matrix
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

# AUCell - full normalized matrix before HVG selection
markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
gene_sets = {col: markers[col].dropna().tolist() for col in markers.columns}

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source", "target"])
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

# SenePy - mouse gene symbols
adata_mouse = sc.read_h5ad("GSE253338_P16_bladder_mouse.h5ad")
sc.pp.normalize_total(adata_mouse, target_sum=1e4)
sc.pp.log1p(adata_mouse)

hubs = sp.load_hubs(species="Mouse", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata_mouse)
merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata_mouse.obs["senpey_universal"] = sp.score_hub(adata_mouse, hubs.hubs["Universal"], translator=translator, binarize=False)

# Transfer mouse scores by cell barcode
adata.obs["senpey_universal"] = adata_mouse.obs["senpey_universal"].reindex(adata.obs_names)

# No bladder-specific SenePy hub was calculated in the source script
adata.obs["senpey_specific"] = np.nan

# SenCID downstream processing - HVGs only here
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis=1)
adata.obs["SID_Score"] = pred_dict["SID2"].loc[adata.obs_names, "SID_Score"]

# External ICE and AUCell scores
# Verify these filenames: the original script references muscle files
ICE_result_deepsas = pd.read_csv("ICE+AUCell results/ICE_es_adata_p16_deepsas.csv", index_col="Unnamed: 0")
AUCell_result_deepsas = pd.read_csv("ICE+AUCell results/AUCELL_es_adata_p16_deepsas.csv", index_col="Unnamed: 0")

adata.obs["ICE_deepsas"] = ICE_result_deepsas["DeepSAS_gene"].reindex(adata.obs_names)
adata.obs["aucell_deepsas"] = AUCell_result_deepsas["seu$DeepSAS_gene_UCell"].reindex(adata.obs_names)
adata.obs["ICE_cellage"] = np.nan

# AUROC - Fibroblasts (clusters 0 and 3)
adata_fibroblast = adata[adata.obs["integrated_snn_res.0.1"].astype(str).isin(["0", "3"])].copy()

metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata_fibroblast.obs[["orig.ident", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["orig.ident"] == "P16_positive").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("GSE253338 - Fibroblasts")
print(auc_df)
auc_df.to_csv("GSE253338_fibroblast_AUROC.csv", index=False)

# AUROC - Basal cells (cluster 1)
adata_basal = adata[adata.obs["integrated_snn_res.0.1"].astype(str).isin(["1"])].copy()

metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata_basal.obs[["orig.ident", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["orig.ident"] == "P16_positive").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("GSE253338 - Basal cells")
print(auc_df)
auc_df.to_csv("GSE253338_basal_AUROC.csv", index=False)


# ============================================================
# 2. GSE233431 - uPAR INTESTINE
# ============================================================

adata = sc.read_h5ad("GSE233431_Upar_human.h5ad")

# DeepScence
adata.obs["b"] = adata.obs["clusters"]
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")

# SenCID
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)

# Normalize full expression matrix
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

# AUCell
markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
gene_sets = {col: markers[col].dropna().tolist() for col in markers.columns}

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source", "target"])
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

# SenePy - mouse gene symbols
adata_mouse = sc.read_h5ad("GSE233431_Upar_mouse.h5ad")
sc.pp.normalize_total(adata_mouse, target_sum=1e4)
sc.pp.log1p(adata_mouse)

hubs = sp.load_hubs(species="Mouse", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata_mouse)

key = ("Large_Intestine", "epithelial cell ")
intestine_hub = hubs.hubs[key]
adata_mouse.obs["senpey_specific"] = sp.score_hub(adata_mouse, intestine_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata_mouse.obs["senpey_universal"] = sp.score_hub(adata_mouse, hubs.hubs["Universal"], translator=translator, binarize=False)

# Transfer mouse scores by barcode
adata.obs["senpey_universal"] = adata_mouse.obs["senpey_universal"].reindex(adata.obs_names)
adata.obs["senpey_specific"] = adata_mouse.obs["senpey_specific"].reindex(adata.obs_names)
adata.obs["senpey_intestine"] = adata.obs["senpey_specific"]

# SenCID downstream processing
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis=1)
adata.obs["SID_Score"] = pred_dict["SID1"].loc[adata.obs_names, "SID_Score"]

# External scores
ICE_result_cellage = pd.read_csv("ICE+AUCell results/ICE_es_GSE233431.csv", index_col="Unnamed: 0")
ICE_result_deepsas = pd.read_csv("ICE+AUCell results/ICE_es_adata_sub_uPAR_deepsas.csv", index_col="Unnamed: 0")
AUCell_result_deepsas = pd.read_csv("ICE+AUCell results/AUCELL_es_adata_uPAR_deepsas.csv", index_col="Unnamed: 0")

adata.obs["ICE_cellage"] = ICE_result_cellage["CellAge"].reindex(adata.obs_names)
adata.obs["ICE_deepsas"] = ICE_result_deepsas["DeepSAS_gene"].reindex(adata.obs_names)
adata.obs["aucell_deepsas"] = AUCell_result_deepsas["seu$DeepSAS_gene_UCell"].reindex(adata.obs_names)

# Epithelial cells, male samples
adata_eval = adata[adata.obs["seurat_clusters"].astype(str).isin(["0", "1", "3", "7"])].copy()
adata_eval = adata_eval[adata_eval.obs["Sample"].astype(str).str.contains("Male", na=False)].copy()

# AUROC
metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata_eval.obs[["Sample", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["Sample"] == "Male_uPAR_positive").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("GSE233431 - uPAR")
print(auc_df)
auc_df.to_csv("uPAR_AUROC.csv", index=False)


# ============================================================
# 3. IN-HOUSE PCLS - LUNG EPITHELIAL
# ============================================================

adata = sc.read_h5ad("PCLS_adata.h5ad")

# DeepScence
adata.obs["b"] = adata.obs["Clusters"]
adata = DeepScence(adata, binarize=True, species="human")

# SenCID
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)

# Normalize full expression matrix
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

# AUCell
markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
gene_sets = {col: markers[col].dropna().tolist() for col in markers.columns}

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source", "target"])
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

# SenePy - human lung
hubs = sp.load_hubs(species="Human", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("lung", "basal cell")
lung_basal_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, lung_basal_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)
adata.obs["senpey_lung_basal"] = adata.obs["senpey_specific"]

# SenCID downstream processing
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis=1)
adata.obs["SID_Score"] = pred_dict["SID2"].loc[adata.obs_names, "SID_Score"]

# External scores
ICE_result_cellage = pd.read_csv("ICE+AUCell results/ICE_es_adata_PCLS_cellage.csv", index_col="Unnamed: 0")
ICE_result_deepsas = pd.read_csv("ICE+AUCell results/ICE_es_adata_PCLS_deepsas.csv", index_col="Unnamed: 0")
AUCell_result_deepsas = pd.read_csv("ICE+AUCell results/AUCELL_PCLS_deepsas.csv", index_col="Unnamed: 0")

adata.obs["ICE_cellage"] = ICE_result_cellage["CellAge"].reindex(adata.obs_names)
adata.obs["ICE_deepsas"] = ICE_result_deepsas["DeepSAS_gene"].reindex(adata.obs_names)
adata.obs["aucell_deepsas"] = AUCell_result_deepsas["seu$DeepSAS_AUCell"].reindex(adata.obs_names)

# Epithelial cells
adata_eval = adata[adata.obs["clusters"] == "Epithelial"].copy()

# AUROC
metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata_eval.obs[["Treatment", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["Treatment"] == "Bleomycin").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("In-house PCLS")
print(auc_df)
auc_df.to_csv("PCLS_AUROC.csv", index=False)


# ============================================================
# 4. GSE190889 - HUMAN IPF
# ============================================================

adata = sc.read_h5ad("GSE190889_IPF_human.h5ad")

# DeepScence
adata.obs["b"] = adata.obs["clusters"]
sc.pp.filter_genes(adata, min_cells=1)
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")

# SenCID
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)

# Normalize full expression matrix
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

# AUCell
markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
gene_sets = {col: markers[col].dropna().tolist() for col in markers.columns}

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source", "target"])
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

# SenePy
hubs = sp.load_hubs(species="Human", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("lung", "basal cell")
lung_basal_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, lung_basal_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)
adata.obs["senpey_lung_basal"] = adata.obs["senpey_specific"]

# SenCID downstream processing
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis=1)
adata.obs["SID_Score"] = pred_dict["SID2"].loc[adata.obs_names, "SID_Score"]

# External scores
ICE_result_cellage = pd.read_csv("ICE+AUCell results/ICE_es_GSE190889.csv", index_col="Unnamed: 0")
ICE_result_deepsas = pd.read_csv("ICE+AUCell results/ICE_es_adata_Human_IPF_deepsas.csv", index_col="Unnamed: 0")
AUCell_result_deepsas = pd.read_csv("ICE+AUCell results/AUCELL_human_IPF_deepsas.csv", index_col="Unnamed: 0")

adata.obs["ICE_cellage"] = ICE_result_cellage["CellAge"].reindex(adata.obs_names)
adata.obs["ICE_deepsas"] = ICE_result_deepsas["DeepSAS_gene"].reindex(adata.obs_names)
adata.obs["aucell_deepsas"] = AUCell_result_deepsas["seu$DeepSAS_AUCell"].reindex(adata.obs_names)

# AUROC
metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata.obs[["health_status", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["health_status"] == "IPF").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("GSE190889 - Human IPF")
print(auc_df)
auc_df.to_csv("IPF_AUROC.csv", index=False)


# ============================================================
# 5. GSE264648 - MICROGLIA
# ============================================================

adata = sc.read_h5ad("GSE264648_adata_AD.h5ad")

# DeepScence
adata.obs["b"] = adata.obs["clusters"]
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")

# SenCID
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)

# Normalize full expression matrix
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

# AUCell
markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
gene_sets = {col: markers[col].dropna().tolist() for col in markers.columns}

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source", "target"])
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

# SenePy
hubs = sp.load_hubs(species="Human", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("hippocampus", "microglia")
microglia_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, microglia_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)
adata.obs["senpey_microglia"] = adata.obs["senpey_specific"]

# SenCID downstream processing
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis=1)
adata.obs["SID_Score"] = pred_dict["SID3"].loc[adata.obs_names, "SID_Score"]

# External scores
ICE_result_cellage = pd.read_csv("ICE+AUCell results/ICE_es_adata_AD.csv", index_col="Unnamed: 0")
ICE_result_deepsas = pd.read_csv("ICE+AUCell results/ICE_es_adata_glia_deepsas.csv", index_col="Unnamed: 0")
AUCell_result_deepsas = pd.read_csv("ICE+AUCell results/AUCELL_glia_deepsas.csv", index_col="Unnamed: 0")

adata.obs["ICE_cellage"] = ICE_result_cellage["CellAge"].reindex(adata.obs_names)
adata.obs["ICE_deepsas"] = ICE_result_deepsas["DeepSAS_gene"].reindex(adata.obs_names)
adata.obs["aucell_deepsas"] = AUCell_result_deepsas["seu$DeepSAS_AUCell"].reindex(adata.obs_names)

# Microglia, Braak stages 0 and 6
adata_eval = adata[adata.obs["clusters"] == "Micro"].copy()
adata_eval = adata_eval[adata_eval.obs["braak"].astype(str).isin(["0", "6"])].copy()

# AUROC
metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata_eval.obs[["diagnosis", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["diagnosis"] == "AD").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("GSE264648 - Microglia")
print(auc_df)
auc_df.to_csv("Microglia_AUROC.csv", index=False)


# ============================================================
# 6. GSE172410 - SKELETAL MUSCLE
# ============================================================

adata = sc.read_h5ad("GSE172410_muscle_human.h5ad")

# DeepScence
adata.obs["b"] = adata.obs["clusters"]
dca(adata)
adata = DeepScence(adata, binarize=True, species="human")

# SenCID
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)

# Normalize full expression matrix
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

# AUCell
markers = pd.read_csv("ICE+AUCell results/senescence_marker_list.csv")
gene_sets = {col: markers[col].dropna().tolist() for col in markers.columns}

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source", "target"])
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(mat=adata, net=net, source="source", target="target", use_raw=False)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

# SenePy - mouse gene symbols
adata_mouse = sc.read_h5ad("GSE172410_muscle_mouse.h5ad")
sc.pp.normalize_total(adata_mouse, target_sum=1e4)
sc.pp.log1p(adata_mouse)

hubs = sp.load_hubs(species="Mouse", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata_mouse)

key = ("Diaphragm", "skeletal muscle satellite cell")
satellite_hub = hubs.hubs[key]
adata_mouse.obs["senpey_specific"] = sp.score_hub(adata_mouse, satellite_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata_mouse.obs["senpey_universal"] = sp.score_hub(adata_mouse, hubs.hubs["Universal"], translator=translator, binarize=False)

# Transfer mouse scores by cell barcode
adata.obs["senpey_universal"] = adata_mouse.obs["senpey_universal"].reindex(adata.obs_names)
adata.obs["senpey_specific"] = adata_mouse.obs["senpey_specific"].reindex(adata.obs_names)
adata.obs["senpey_skeletal"] = adata.obs["senpey_specific"]

# SenCID downstream processing
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis=1)
adata.obs["SID_Score"] = pred_dict["SID3"].loc[adata.obs_names, "SID_Score"]

# External scores
ICE_result_cellage = pd.read_csv("ICE+AUCell results/ICE_es_skeletal_human_version.csv", index_col="Unnamed: 0")
ICE_result_deepsas = pd.read_csv("ICE+AUCell results/ICE_es_adata_skeletal_muscle_deepsas.csv", index_col="Unnamed: 0")
AUCell_result_deepsas = pd.read_csv("ICE+AUCell results/AUCELL_skeletal_muscle_deepsas.csv", index_col="Unnamed: 0")

adata.obs["ICE_cellage"] = ICE_result_cellage["CellAge"].reindex(adata.obs_names)
adata.obs["ICE_deepsas"] = ICE_result_deepsas["DeepSAS_gene"].reindex(adata.obs_names)
adata.obs["aucell_deepsas"] = AUCell_result_deepsas["seu$DeepSAS_AUCell"].reindex(adata.obs_names)

# Satellite cells
adata_eval = adata[adata.obs["clusters"] == "satellite"].copy()

# AUROC
metrics = [
    "ds",                 # DeepScence
    "senpey_universal",   # SenePy universal score
    "senpey_specific",    # SenePy cell-type-specific score
    "SID_Score",          # SenCID score
    "SnC scores",         # DeepSAS score
    "CellAge",            # AUCell: CellAge
    "GO",                 # AUCell: GO
    "SenMayo",            # AUCell: SenMayo
    "ICE_cellage",        # ICE: CellAge
    "ICE_deepsas",        # ICE: DeepSAS
    "aucell_deepsas"      # AUCell: DeepSAS
]

df = adata_eval.obs[["age", *metrics]].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
df["label"] = (df["age"] == "Old").astype(int)
auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(), "label"], df.loc[df[m].notna(), m]) if df.loc[df[m].notna(), "label"].nunique() == 2 else np.nan for m in metrics]})
print("GSE172410 - Skeletal muscle")
print(auc_df)
auc_df.to_csv("Skeletal_AUROC.csv", index=False)
