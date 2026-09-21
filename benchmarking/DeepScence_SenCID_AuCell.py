#load modules
import scanpy as sc
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

###DeepScence 
from DeepScence.api import DeepScence
from dca.api import dca

##SenCID
from SenCID.api import SenCID
import scanpy as sc
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix

#Inhouse- Alveolar epithelial   
adata=sc.read_h5ad('/fs/ess/PAS2148/Ahmed/pcls_DeepSAS/PCLS_adata_Majorcell.h5ad')

########## DeepScence
adata.obs["b"]=adata.obs['Clusters']#to control for cell type variation when running DeepScence

adata = DeepScence(adata, binarize=True,species='human')

##########SenCID
pred_dict, recSID, tmpfiles = SenCID(adata = adata, 
                    sidnums = [1,2,3,4,5,6], 
                    denoising = True, 
                    binarize = True, 
                    threads = 5, 
                    savetmp = True)

adata.raw = adata

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var['highly_variable']]
#sc.pp.regress_out(adata, ['n_counts', 'percent_mito'] ) #, n_jobs=args.threads)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack') 

adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis = 1)
markers = ['rec_SID1', 'rec_SID2', 'rec_SID3', 'rec_SID4', 'rec_SID5', 'rec_SID6']
sc.pl.violin(adata, keys = markers)

#the SID with the highest recommendation score is considered to be most suitable for senescence evaluation

adata.obs['SID_Score'] = pred_dict['SID6'].loc[adata.obs_names, 'SID_Score']

####GSE190889---Human IPF
 
adata=sc.read_h5ad('/fs/ess/PAS2148/Ahmed/pcls_DeepSAS/GSE190889_IPF_human_sub')

adata.obs["b"]=adata.obs['clusters']#to control for cell type variation when running DeepScence
sc.pp.filter_genes(adata, min_cells=1)
dca(adata)
adata = DeepScence(adata, binarize=True,species='human')


pred_dict, recSID, tmpfiles = SenCID(adata = adata, 
                    sidnums = [1,2,3,4,5,6], 
                    denoising = True, 
                    binarize = True, 
                    threads = 5, 
                    savetmp = True)

adata.raw = adata

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var['highly_variable']]
#sc.pp.regress_out(adata, ['n_counts', 'percent_mito'] ) #, n_jobs=args.threads)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack') 
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis = 1)
markers = ['rec_SID1', 'rec_SID2', 'rec_SID3', 'rec_SID4', 'rec_SID5', 'rec_SID6']
sc.pl.violin(adata, keys = markers)

adata.obs['SID_Score'] = pred_dict['SID2'].loc[adata.obs_names, 'SID_Score']


####GSE233431
adata.obs["b"]=adata.obs['clusters']#to control for cell type variation when running DeepScence

dca(adata)
adata = DeepScence(adata, binarize=True,species='human')

pred_dict, recSID, tmpfiles = SenCID(adata = adata, 
                    sidnums = [1,2,3,4,5,6], 
                    denoising = True, 
                    binarize = True, 
                    threads = 5, 
                    savetmp = True)
adata.raw = adata

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var['highly_variable']]
#sc.pp.regress_out(adata, ['n_counts', 'percent_mito'] ) #, n_jobs=args.threads)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack') 
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis = 1)
markers = ['rec_SID1', 'rec_SID2', 'rec_SID3', 'rec_SID4', 'rec_SID5', 'rec_SID6']
sc.pl.violin(adata, keys = markers)

adata.obs['SID_Score'] = pred_dict['SID1'].loc[adata.obs_names, 'SID_Score']


adata_sub = adata[adata.obs["seurat_clusters"].isin(["0","1","3","7"])].copy()##Epithelial
adata_male = adata_sub[
    adata_sub.obs["Sample"].str.contains("Male")
].copy()

##GSE264648 Microglia

adata=sc.read_h5ad('adata_AD.h5ad')
adata.obs["b"]=adata.obs['clusters']#to control for cell type variation when running DeepScence

dca(adata)
adata = DeepScence(adata, binarize=True,species='human')

pred_dict, recSID, tmpfiles = SenCID(adata = adata, 
                    sidnums = [1,2,3,4,5,6], 
                    denoising = True, 
                    binarize = True, 
                    threads = 5, 
                    savetmp = True)

adata.raw = adata

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var['highly_variable']]
#sc.pp.regress_out(adata, ['n_counts', 'percent_mito'] ) #, n_jobs=args.threads)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack') 
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis = 1)
markers = ['rec_SID1', 'rec_SID2', 'rec_SID3', 'rec_SID4', 'rec_SID5', 'rec_SID6']
sc.pl.violin(adata, keys = markers)

adata.obs['SID_Score'] = pred_dict['SID3'].loc[adata.obs_names, 'SID_Score']

adata_oligo = adata[
    adata.obs["clusters"].isin(["Micro"])
].copy()

adata_oligo = adata_oligo[adata_oligo.obs["braak"].isin(['0', '6'])].copy()

##GSE253338_UPAR
adata=sc.read_h5ad('GSE253338_UPAR.h5ad')

adata.obs["b"]=adata.obs['clusters']#to control for cell type variation when running DeepScence


dca(adata)
adata = DeepScence(adata, binarize=True,species='human')

pred_dict, recSID, tmpfiles = SenCID(adata = adata, 
                    sidnums = [1,2,3,4,5,6], 
                    denoising = True, 
                    binarize = True, 
                    threads = 5, 
                    savetmp = True)

adata.raw = adata

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e5)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var['highly_variable']]
#sc.pp.regress_out(adata, ['n_counts', 'percent_mito'] ) #, n_jobs=args.threads)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack') 
adata.obs = pd.concat([adata.obs, recSID.loc[adata.obs_names, :]], axis = 1)
markers = ['rec_SID1', 'rec_SID2', 'rec_SID3', 'rec_SID4', 'rec_SID5', 'rec_SID6']
sc.pl.violin(adata, keys = markers)

adata.obs['SID_Score'] = pred_dict['SID2'].loc[adata.obs_names, 'SID_Score']

adata_sub = adata[
    adata.obs["integrated_snn_res.0.1"].isin(["0","3"])#"0","3" fibroblast, 1 basal
].copy()











#########################AUCell for CellAge and SenMayo
markers=pd.read_csv('/fs/ess/PCON0022/Ahmed/ATAC2HiC/PBMC/scATAC-seq/PBMC_Hg19/Model/sencell-deepsas-v1/senescence_marker_list.csv')
import pandas as pd

gene_sets = {
    col: markers[col].dropna().tolist()
    for col in markers.columns
}

import pandas as pd

net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source","target"])
import decoupler as dc
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell(
    mat=adata,
    net=net,
    source="source",
    target="target",use_raw=False
)
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])
