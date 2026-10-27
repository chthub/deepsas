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

##Senepy
import senepy as sp


#Inhouse- Alveolar epithelial   
adata=sc.read_h5ad('/fs/ess/PAS2148/Ahmed/pcls_DeepSAS/PCLS_adata_Majorcell.h5ad')

########## DeepScence
adata.obs["b"]=adata.obs['Clusters']#to control for cell type variation when running DeepScence

adata = DeepScence(adata, binarize=True,species='human')
adata.obs['ds']##is the contioous score for DeepScence
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

adata.obs['SID_Score'] = pred_dict['SID2'].loc[adata.obs_names, 'SID_Score']

####Senepy
hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ("lung", "basal cell")

lung_basal_hub = hubs.hubs[key]
adata.obs['senpey_specific'] = sp.score_hub(adata, lung_basal_hub,translator=translator,binarize=False)

hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

merge_results = hubs.merge_hubs(hubs.metadata, new_name = 'Universal',
                calculate_thresh = True, p_thres = 0.05)


adata.obs['senpey_universal'] = sp.score_hub(adata, hubs.hubs['Universal'],translator=translator,binarize=False)

####AUCell SenMayo+CellAge
markers=pd.read_csv('/fs/ess/PCON0022/Ahmed/ATAC2HiC/PBMC/scATAC-seq/PBMC_Hg19/Model/sencell-deepsas-v1/senescence_marker_list.csv')
import pandas as pd

gene_sets = {col: markers[col].dropna().tolist()for col in markers.columns}


net = []
for k, genes in gene_sets.items():
    for g in genes:
        net.append([k, g])

net = pd.DataFrame(net, columns=["source","target"])

import decoupler as dc
net = net.drop_duplicates(subset=["source", "target"])
dc.run_aucell( mat=adata, net=net, source="source", target="target")
adata.obs = adata.obs.join(adata.obsm["aucell_estimate"])

###ICE results
ICE_result_cellage=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_PCLS_cellage.csv',index_col='Unnamed: 0')
ICE_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_PCLS_deepsas.csv',index_col='Unnamed: 0')
AUCell_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/AUCELL_PCLS_deepsas.csv',index_col='Unnamed: 0')


adata.obs['ICE_cellage']=ICE_result_cellage.CellAge.values
adata.obs['ICE_deepsas']=ICE_result_deepsas.DeepSAS_gene.values
adata.obs['aucell_deepsas']=AUCell_result_deepsas['seu$DeepSAS_AUCell'].values
adata.obs['senpey_universal']
adata.obs['senpey_specific'] 
#################################################AUROC
from sklearn.metrics import roc_auc_score
adata_epi = adata[adata.obs["clusters"] == "Epithelial"].copy()###comapre the epithelial cells 
metrics = ["ds",                 ##DeepScence
           "senpey_universal",   ##Senpey universal score
           "senpey_specific",    ##cell specific senpey
           "SID_Score",          ##SenCID score
           "SnC scores",         ##DeepSAS score
           "CellAge",            ##AUCell:CellAge
           "GO",                 ##AUCell:GO
           "SenMayo",            ##AUCell:SenMayo
           "ICE_cellage",        ##ICE:CellAge
           "ICE_deepsas",        ##ICE:DeepSAS
           "aucell_deepsas"      ##AUCell:DeepSAS
           ]

df = adata_epi.obs[["Treatment", *metrics]].copy()
df["label"] = df["Treatment"].eq("Bleomycin").astype(int)

auc_df = pd.DataFrame({
    "metric": metrics,
    "AUROC": [roc_auc_score(df["label"], df[m]) for m in metrics]
}).sort_values("AUROC", ascending=False)

auc_df

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

##GSE253338
adata=sc.read_h5ad('GSE253338_P16_bladder_sub.h5ad')

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


########GSE172410

adata=sc.read_h5ad('skeletal_human_version.h5ad')

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


# subset satellite cells
adata = adata[adata.obs['clusters'] == 'satellite'].copy()



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
