import scanpy as sc
import numpy as np
from gprofiler import GProfiler
from pathlib import Path

gp = GProfiler(return_dataframe=True)

datasets = [
    ("GSE172410_muscle_mouse.h5ad", "GSE172410_muscle_human.h5ad"),
    ("GSE233431_Upar_mouse.h5ad", "GSE233431_Upar_human.h5ad"),
    ("GSE253338_P16_bladder_mouse.h5ad", "GSE253338_P16_bladder_human.h5ad")
]

for input_file, output_file in datasets:
    print(f"\nProcessing: {input_file}")

    if not Path(input_file).exists():
        print(f"Missing input: {input_file}")
        continue

    adata = sc.read_h5ad(input_file)

    res = gp.orth(organism="mmusculus", target="hsapiens", query=adata.var_names.tolist())

    res = res[res["name"].notna() & res["incoming"].notna()]
    res = res[~res["name"].isin(["N/A", "None", ""])]
    res = res.drop_duplicates(subset=["name"])

    mapping = dict(zip(res["incoming"], res["name"]))

    adata = adata[:, adata.var_names.isin(mapping.keys())].copy()
    adata.var["human_gene"] = adata.var_names.map(mapping)
    adata = adata[:, adata.var["human_gene"].notna()].copy()
    adata.var_names = adata.var["human_gene"].astype(str)
    adata.var_names_make_unique()

    adata.write_h5ad(output_file)

    print(f"Saved: {output_file}")
    print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

print("\nMouse-to-human conversion completed.")
