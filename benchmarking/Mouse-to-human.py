adata=sc.read_h5ad('GSE253338_P16_bladder_mouse.h5ad')

from gprofiler import GProfiler
gp = GProfiler(return_dataframe=True)

res = gp.orth(
    organism="mmusculus",
    target="hsapiens",
    query=adata.var_names.tolist()
)

# KEEP everything that mapped (no strict filtering)
res = res[res['name'].notna()]
# keep first occurrence per human gene
res = res.drop_duplicates(subset=['name'])
mapping = dict(zip(res['incoming'], res['name']))
adata = adata[:, adata.var_names.isin(mapping.keys())].copy()

adata.var['human_gene'] = adata.var_names.map(mapping)
adata.var_names = adata.var['human_gene']
adata.var_names_make_unique()

import numpy as np

# fix fake NA
adata.var["human_gene"] = adata.var["human_gene"].replace("N/A", np.nan)

# filter genes (correct axis)
adata = adata[:, adata.var["human_gene"].notna()].copy()

# set gene names
adata.var_names = adata.var["human_gene"]
adata.var_names_make_unique()

adata.write_h5ad('GSE253338_P16_bladder_human.h5ad')
