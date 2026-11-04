# conda activate SenCID
# Run from the directory containing the input datasets.
# Outputs: benchmark_scores/*.csv
# Original h5ad files are never modified or saved.

import h5py
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from SenCID.api import SenCID

OUT = Path("benchmark_scores")
OUT.mkdir(exist_ok=True)

# ============================================================
# GSE253338: SID2
# ============================================================

with h5py.File("GSE253338_P16_bladder_human.h5ad", "r") as f:
    x = f["X"]
    X = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]
    obs = pd.DataFrame(index=pd.Index(f["obs"][f["obs"].attrs["_index"]][()].astype(str)))
    var = pd.DataFrame(index=pd.Index(f["var"][f["var"].attrs["_index"]][()].astype(str)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for key in f["layers"]:
        x = f["layers"][key]
        adata.layers[key] = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]

assert adata.obs_names.is_unique, "Duplicate barcodes: GSE253338"
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)
score = pred_dict["SID2"]["SID_Score"]
assert score.index.is_unique, "Duplicate SenCID score barcodes"
assert adata.obs_names.isin(score.index).all(), "Missing SenCID score barcodes"
adata.obs["SID_Score"] = pd.to_numeric(score.reindex(adata.obs_names), errors="raise")
adata.obs[["SID_Score"]].rename_axis("barcode").to_csv(OUT / "GSE253338_SenCID.csv")
print("GSE253338: SenCID saved", adata.obs["SID_Score"].notna().sum())

# ============================================================
# GSE233431: SID1
# ============================================================

with h5py.File("GSE233431_Upar_human.h5ad", "r") as f:
    x = f["X"]
    X = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]
    obs = pd.DataFrame(index=pd.Index(f["obs"][f["obs"].attrs["_index"]][()].astype(str)))
    var = pd.DataFrame(index=pd.Index(f["var"][f["var"].attrs["_index"]][()].astype(str)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for key in f["layers"]:
        x = f["layers"][key]
        adata.layers[key] = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]

assert adata.obs_names.is_unique, "Duplicate barcodes: GSE233431"
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)
score = pred_dict["SID1"]["SID_Score"]
assert score.index.is_unique, "Duplicate SenCID score barcodes"
assert adata.obs_names.isin(score.index).all(), "Missing SenCID score barcodes"
adata.obs["SID_Score"] = pd.to_numeric(score.reindex(adata.obs_names), errors="raise")
adata.obs[["SID_Score"]].rename_axis("barcode").to_csv(OUT / "GSE233431_SenCID.csv")
print("GSE233431: SenCID saved", adata.obs["SID_Score"].notna().sum())

# ============================================================
# PCLS: SID2
# ============================================================

with h5py.File("PCLS_adata.h5ad", "r") as f:
    x = f["X"]
    X = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]
    obs = pd.DataFrame(index=pd.Index(f["obs"][f["obs"].attrs["_index"]][()].astype(str)))
    var = pd.DataFrame(index=pd.Index(f["var"][f["var"].attrs["_index"]][()].astype(str)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for key in f["layers"]:
        x = f["layers"][key]
        adata.layers[key] = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]

assert adata.obs_names.is_unique, "Duplicate barcodes: PCLS"
sc.pp.filter_genes(adata, min_cells=3)
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)
score = pred_dict["SID2"]["SID_Score"]
assert score.index.is_unique, "Duplicate SenCID score barcodes"
assert adata.obs_names.isin(score.index).all(), "Missing SenCID score barcodes"
adata.obs["SID_Score"] = pd.to_numeric(score.reindex(adata.obs_names), errors="raise")
adata.obs[["SID_Score"]].rename_axis("barcode").to_csv(OUT / "PCLS_SenCID.csv")
print("PCLS: SenCID saved", adata.obs["SID_Score"].notna().sum())

# ============================================================
# GSE190889: SID2
# ============================================================

with h5py.File("GSE190889_IPF_human.h5ad", "r") as f:
    x = f["X"]
    X = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]
    obs = pd.DataFrame(index=pd.Index(f["obs"][f["obs"].attrs["_index"]][()].astype(str)))
    var = pd.DataFrame(index=pd.Index(f["var"][f["var"].attrs["_index"]][()].astype(str)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for key in f["layers"]:
        x = f["layers"][key]
        adata.layers[key] = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]

assert adata.obs_names.is_unique, "Duplicate barcodes: GSE190889"
sc.pp.filter_genes(adata, min_cells=3)
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)
score = pred_dict["SID2"]["SID_Score"]
assert score.index.is_unique, "Duplicate SenCID score barcodes"
assert adata.obs_names.isin(score.index).all(), "Missing SenCID score barcodes"
adata.obs["SID_Score"] = pd.to_numeric(score.reindex(adata.obs_names), errors="raise")
adata.obs[["SID_Score"]].rename_axis("barcode").to_csv(OUT / "GSE190889_SenCID.csv")
print("GSE190889: SenCID saved", adata.obs["SID_Score"].notna().sum())

# ============================================================
# GSE264648: SID3
# ============================================================

with h5py.File("GSE264648_adata_AD.h5ad", "r") as f:
    x = f["X"]
    X = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]
    obs = pd.DataFrame(index=pd.Index(f["obs"][f["obs"].attrs["_index"]][()].astype(str)))
    var = pd.DataFrame(index=pd.Index(f["var"][f["var"].attrs["_index"]][()].astype(str)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for key in f["layers"]:
        x = f["layers"][key]
        adata.layers[key] = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]

assert adata.obs_names.is_unique, "Duplicate barcodes: GSE264648"
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)
score = pred_dict["SID3"]["SID_Score"]
assert score.index.is_unique, "Duplicate SenCID score barcodes"
assert adata.obs_names.isin(score.index).all(), "Missing SenCID score barcodes"
adata.obs["SID_Score"] = pd.to_numeric(score.reindex(adata.obs_names), errors="raise")
adata.obs[["SID_Score"]].rename_axis("barcode").to_csv(OUT / "GSE264648_SenCID.csv")
print("GSE264648: SenCID saved", adata.obs["SID_Score"].notna().sum())

# ============================================================
# GSE172410: SID3
# ============================================================

with h5py.File("GSE172410_muscle_human.h5ad", "r") as f:
    x = f["X"]
    X = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]
    obs = pd.DataFrame(index=pd.Index(f["obs"][f["obs"].attrs["_index"]][()].astype(str)))
    var = pd.DataFrame(index=pd.Index(f["var"][f["var"].attrs["_index"]][()].astype(str)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    for key in f["layers"]:
        x = f["layers"][key]
        adata.layers[key] = sparse.csr_matrix((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"])) if isinstance(x, h5py.Group) else x[()]

assert adata.obs_names.is_unique, "Duplicate barcodes: GSE172410"
pred_dict, recSID, tmpfiles = SenCID(adata=adata, sidnums=[1, 2, 3, 4, 5, 6], denoising=True, binarize=True, threads=5, savetmp=True)
score = pred_dict["SID3"]["SID_Score"]
assert score.index.is_unique, "Duplicate SenCID score barcodes"
assert adata.obs_names.isin(score.index).all(), "Missing SenCID score barcodes"
adata.obs["SID_Score"] = pd.to_numeric(score.reindex(adata.obs_names), errors="raise")
adata.obs[["SID_Score"]].rename_axis("barcode").to_csv(OUT / "GSE172410_SenCID.csv")
print("GSE172410: SenCID saved", adata.obs["SID_Score"].notna().sum())