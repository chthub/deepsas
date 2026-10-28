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

##DeepScence
adata.obs["b"]=adata.obs['clusters']#to control for cell type variation when running DeepScence
sc.pp.filter_genes(adata, min_cells=1)
dca(adata)
adata = DeepScence(adata, binarize=True,species='human')

##SenCID
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

##Senpey
hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ("lung", "basal cell")

lung_basal_hub = hubs.hubs[key]
adata.obs['senpey'] = sp.score_hub(adata, lung_basal_hub,translator=translator,binarize=False)

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


##other scores 
ICE_result_cellage=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_GSE190889.csv',index_col='Unnamed: 0')
ICE_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_Human_IPF_deepsas.csv',index_col='Unnamed: 0')
AUCell_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/AUCELL_human_IPF_deepsas.csv',index_col='Unnamed: 0')

adata.obs['ICE_cellage']=ICE_result_cellage.CellAge.values
adata.obs['ICE_deepsas']=ICE_result_deepsas.DeepSAS_gene.values
adata.obs['aucell_deepsas']=AUCell_result_deepsas['seu$DeepSAS_AUCell'].values


####run AUROC
from sklearn.metrics import roc_auc_score
df=adata.obs[['health_status','SnC scores','ds','Universal_score','SID_Score','CellAge', 'GO', 'SenMayo','ICE_cellage', 
              'ICE_deepsas','aucell_deepsas','senpey_specific','senpey_universal']]
df = df.dropna()

df["label"] = (df["health_status"]=="IPF").astype(int)

aucs = []
metrics=[metrics = ["ds",                 ##DeepScence
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
           ]]
for m in metrics:
    auc = roc_auc_score(df["label"], df[m])
    aucs.append([m, auc])

auc_df = pd.DataFrame(aucs, columns=["metric","AUROC"])
auc_df

####GSE233431
adata=sc.read_h5ad('data/GSE233431_Upar_human.h5ad')
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

###Senepy using the mouse gene symbols
adata_mouse=sc.read_h5ad('data/GSE233431_Upar_mouse.h5ad')
sc.pp.normalize_total(adata_mouse, target_sum=1e4)
sc.pp.log1p(adata_mouse)

hubs= sp.load_hubs(species='Mouse',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata_mouse)

key = ('Large_Intestine', 'epithelial cell ')

intestine_hub = hubs.hubs[key]
adata_mouse.obs['senpey_celltype'] = sp.score_hub(adata_mouse, intestine_hub,translator=translator,binarize=False)

hubs= sp.load_hubs(species='Mouse',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata_mouse)

merge_results = hubs.merge_hubs(hubs.metadata, new_name = 'Universal',
                calculate_thresh = True, p_thres = 0.05)

adata_mouse.obs['senpey_universal'] = sp.score_hub(adata_mouse, hubs.hubs['Universal'],translator=translator,binarize=False)

adata.obs['senpey_universal']=adata_mouse.obs['senpey_universal'].values
adata.obs['senpey_celltype']=adata_mouse.obs['senpey_celltype'].values

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

ICE_result_cellage=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_GSE233431.csv',index_col='Unnamed: 0')
ICE_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_sub_uPAR_deepsas.csv',index_col='Unnamed: 0')
aucell_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/AUCELL_es_adata_uPAR_deepsas.csv',index_col='Unnamed: 0')


adata.obs['ICE_cellage']=ICE_result_cellage.CellAge.values
adata.obs['ICE_deepsas']=ICE_result_deepsas.DeepSAS_gene.values
adata.obs['aucell_deepsas']=aucell_result_deepsas['seu$DeepSAS_gene_UCell'].values


adata_sub = adata[adata.obs["seurat_clusters"].isin(["0","1","3","7"])].copy()##Epithelial
adata_male = adata_sub[
    adata_sub.obs["Sample"].str.contains("Male")
].copy()

from sklearn.metrics import roc_auc_score

df=adata_male.obs[['Sample','SnC scores','ds','Universal_score','SID_Score','CellAge', 'GO', 'SenMayo','ICE_cellage','ICE_deepsas', 'aucell_deepsas','SenePy_intestine']]
df = df.dropna()

df["label"] = (df["Sample"]=="Male_uPAR_positive").astype(int)

aucs = []
metrics=['SnC scores','ds', 'Universal_score','SID_Score','CellAge',  'GO', 'SenMayo','ICE_cellage','ICE_deepsas', 'aucell_deepsas','SenePy_intestine']
for m in metrics:
    auc = roc_auc_score(df["label"], df[m])
    aucs.append([m, auc])

auc_df = pd.DataFrame(aucs, columns=["metric","AUROC"])
auc_df
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

hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ('hippocampus', 'microglia')

microglia_hub = hubs.hubs[key]
adata.obs['senpey_basal'] = sp.score_hub(adata, microglia_hub,translator=translator,binarize=False)

hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

merge_results = hubs.merge_hubs(hubs.metadata, new_name = 'Universal',
                calculate_thresh = True, p_thres = 0.05)


adata.obs['senpey_universal'] = sp.score_hub(adata, hubs.hubs['Universal'],translator=translator,binarize=False)

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

ICE_result_=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_AD.csv',index_col='Unnamed: 0')
ICE_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_glia_deepsas.csv',index_col='Unnamed: 0')
AUCell_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/AUCELL_glia_deepsas.csv',index_col='Unnamed: 0')


adata.obs['ICE_cellage']=ICE_result_.CellAge.values
adata.obs['ICE_deepsas']=ICE_result_deepsas.DeepSAS_gene.values
adata.obs['aucell_deepsas']=AUCell_result_deepsas['seu$DeepSAS_AUCell'].values


adata_oligo = adata[
    adata.obs["clusters"].isin(["Micro"])
].copy()

adata_oligo = adata_oligo[adata_oligo.obs["braak"].isin(['0', '6'])].copy()
from sklearn.metrics import roc_auc_score


df=adata_oligo.obs[['diagnosis','SnC scores','ds','Universal_score','SID_Score','CellAge', 'GO', 'SenMayo','ICE_cellage',
                    'ICE_deepsas', 'aucell_deepsas']]
df = df.dropna()

df["label"] = (df["diagnosis"]=="AD").astype(int)

aucs = []
metrics=['SnC scores','ds', 'Universal_score','SID_Score','CellAge',  'GO', 'SenMayo','ICE_cellage','ICE_deepsas', 'aucell_deepsas']
for m in metrics:
    auc = roc_auc_score(df["label"], df[m])
    aucs.append([m, auc])

auc_df = pd.DataFrame(aucs, columns=["metric","AUROC"])
auc_df
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

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata_mouse=sc.read_h5ad('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/GSE253338/adata_P16_muscle.h5ad')
hubs= sp.load_hubs(species='Mouse',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata_mouse)

merge_results = hubs.merge_hubs(hubs.metadata, new_name = 'Universal',
                calculate_thresh = True, p_thres = 0.05)


adata_mouse.obs['senpey_universal'] = sp.score_hub(adata_mouse, hubs.hubs['Universal'],translator=translator,binarize=False)

###then transfer the value to the adata (human version)

(you do it )

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


ICE_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/ICE_es_adata_sub_balanced_muscle_deepsas.csv',index_col='Unnamed: 0')
aucell_result_deepsas=pd.read_csv('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/AUCELL_es_adata_balanced_muscle_deepsas.csv',index_col='Unnamed: 0')


adata.obs['ICE_deepsas']=ICE_result_deepsas.DeepSAS_gene.values
adata.obs['aucell_deepsas']=aucell_result_deepsas['seu$DeepSAS_gene_UCell'].values
adata_sub = adata[
    adata.obs["integrated_snn_res.0.1"].isin(["0","3"])#"0","3" fibroblast, "1" basal
].copy()

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

metrics = ['SnC scores','ds','Universal_score','SID_Score','CellAge','GO','SenMayo','ICE_cellage','ICE_deepsas', 'aucell_deepsas']

df = adata_sub.obs[['orig.ident'] + metrics].copy()
df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')
df["label"] = (df["orig.ident"] == "P16_positive").astype(int)

auc_df = pd.DataFrame({"metric": metrics, "AUROC": [roc_auc_score(df.loc[df[m].notna(),"label"], df.loc[df[m].notna(),m]) if df[m].notna().sum() > 0 and df.loc[df[m].notna(),"label"].nunique() == 2 else np.nan for m in metrics]})

auc_df (make one for fibroblast/make one for basal cells

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



