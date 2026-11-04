# ============================================================
# SenePy benchmarking
# conda activate senepy
# python SenePy.py
# ============================================================

import scanpy as sc
import senepy as sp
import pandas as pd
import numpy as np
from pathlib import Path

Path("benchmark_scores").mkdir(exist_ok=True)


# ============================================================
# 1. GSE253338 - p16 BLADDER
# ============================================================

adata = sc.read_h5ad("GSE253338_P16_bladder_mouse.h5ad")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

hubs = sp.load_hubs(species="Mouse", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)
adata.obs["senpey_specific"] = np.nan

adata.obs[["senpey_universal", "senpey_specific"]].to_csv("benchmark_scores/GSE253338_SenePy.csv", index_label="barcode")

print("GSE253338 completed")


# ============================================================
# 2. GSE233431 - uPAR INTESTINE
# ============================================================

adata = sc.read_h5ad("GSE233431_Upar_mouse.h5ad")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

hubs = sp.load_hubs(species="Mouse", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("Large_Intestine", "epithelial cell ")
intestine_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, intestine_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)

adata.obs[["senpey_universal", "senpey_specific"]].to_csv("benchmark_scores/GSE233431_SenePy.csv", index_label="barcode")

print("GSE233431 completed")


# ============================================================
# 3. IN-HOUSE PCLS - LUNG EPITHELIAL
# ============================================================

adata = sc.read_h5ad("PCLS_adata.h5ad")

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

hubs = sp.load_hubs(species="Human", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("lung", "basal cell")
lung_basal_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, lung_basal_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)

adata.obs[["senpey_universal", "senpey_specific"]].to_csv("benchmark_scores/PCLS_SenePy.csv", index_label="barcode")

print("PCLS completed")


# ============================================================
# 4. GSE190889 - HUMAN IPF
# ============================================================

adata = sc.read_h5ad("GSE190889_IPF_human.h5ad")

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

hubs = sp.load_hubs(species="Human", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("lung", "basal cell")
lung_basal_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, lung_basal_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)

adata.obs[["senpey_universal", "senpey_specific"]].to_csv("benchmark_scores/GSE190889_SenePy.csv", index_label="barcode")

print("GSE190889 completed")


# ============================================================
# 5. GSE264648 - MICROGLIA
# ============================================================

adata = sc.read_h5ad("GSE264648_adata_AD.h5ad")

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)

hubs = sp.load_hubs(species="Human", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("hippocampus", "microglia")
microglia_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, microglia_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)

adata.obs[["senpey_universal", "senpey_specific"]].to_csv("benchmark_scores/GSE264648_SenePy.csv", index_label="barcode")

print("GSE264648 completed")


# ============================================================
# 6. GSE172410 - SKELETAL MUSCLE
# ============================================================

adata = sc.read_h5ad("GSE172410_muscle_mouse.h5ad")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

hubs = sp.load_hubs(species="Mouse", sig_type="cell_type")
translator = sp.translator(hub=hubs.hubs, data=adata)

key = ("Diaphragm", "skeletal muscle satellite cell")
satellite_hub = hubs.hubs[key]
adata.obs["senpey_specific"] = sp.score_hub(adata, satellite_hub, translator=translator, binarize=False)

merge_results = hubs.merge_hubs(hubs.metadata, new_name="Universal", calculate_thresh=True, p_thres=0.05)
adata.obs["senpey_universal"] = sp.score_hub(adata, hubs.hubs["Universal"], translator=translator, binarize=False)

adata.obs[["senpey_universal", "senpey_specific"]].to_csv("benchmark_scores/GSE172410_SenePy.csv", index_label="barcode")

print("GSE172410 completed")