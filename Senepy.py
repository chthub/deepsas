#load modules
import senepy as sp
import scanpy as sc
import pandas as pd

#Inhouse- Alveolar epithelial   
adata=sc.read_h5ad('/fs/ess/PAS2148/Ahmed/pcls_DeepSAS/PCLS_adata_Majorcell.h5ad')

hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ("lung", "basal cell")

lung_basal_hub = hubs.hubs[key]
adata.obs['senpey'] = sp.score_hub(adata, lung_basal_hub,translator=translator,binarize=False)

####GSE190889---Human IPF
hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ("lung", "basal cell")

lung_basal_hub = hubs.hubs[key]
adata.obs['senpey'] = sp.score_hub(adata, lung_basal_hub,translator=translator,binarize=False)

#GSE172410--Skeletal_muscle
adata=sc.read_h5ad('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/GSE172410/adata_skeletal_muscle.h5ad')

hubs= sp.load_hubs(species='Mouse',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ("Diaphragm", "skeletal muscle satellite cell")

satellite_hub = hubs.hubs[key]
adata.obs['senpey'] = sp.score_hub(adata, satellite_hub,translator=translator,binarize=False)

adata = adata[adata.obs['clusters'] == 'satellite'].copy()

##GSE253338 Upar

adata=sc.read_h5ad('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/GSE233431/adata_colon_uPAR.h5ad')

hubs= sp.load_hubs(species='Mouse',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ('Large_Intestine', 'epithelial cell ')

intestine_hub = hubs.hubs[key]
adata.obs['senpey'] = sp.score_hub(adata, intestine_hub,translator=translator,binarize=False)

adata_sub = adata[adata.obs["seurat_clusters"].isin(["0","1","3","7"])].copy()#Epithelial
adata_male = adata_sub[
    adata_sub.obs["Sample"].str.contains("Male")
].copy()

##GSE264648 Microglia
adata=sc.read_h5ad('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/GSE264648/adata_AD.h5ad')

hubs= sp.load_hubs(species='Human',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

key = ('hippocampus', 'microglia')

microglia_hub = hubs.hubs[key]
adata.obs['senpey'] = sp.score_hub(adata, microglia_hub,translator=translator,binarize=False)

adata_oligo = adata[
    adata.obs["clusters"].isin(["Micro"])
].copy()

adata_oligo = adata_oligo[adata_oligo.obs["braak"].isin(['0', '6'])].copy()

######p16 bladder GSE253338


adata=sc.read_h5ad('/fs/ess/PAS1475/Ahmed/DeepSAS/Benchmark/GSE253338/adata_P16_muscle.h5ad')

hubs= sp.load_hubs(species='Mouse',sig_type='cell_type')
translator = sp.translator(hub = hubs.hubs, data = adata)

merge_results = hubs.merge_hubs(hubs.metadata, new_name = 'Universal',
                calculate_thresh = True, p_thres = 0.05)


adata.obs['senpey'] = sp.score_hub(adata, hubs.hubs['Universal'],translator=translator,binarize=False)

adata_sub = adata[
    adata.obs["integrated_snn_res.0.1"].isin(["0","3"])#"0","3" fibroblast,!1 is the basal cells xoxox
].copy()

