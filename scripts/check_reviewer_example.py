"""Check the documented example after training and table generation."""

import argparse
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch


def check_example(output_dir, exp_name="example"):
    output_dir = Path(output_dir)
    with (output_dir / f"{exp_name}_run_summary.json").open() as handle:
        summary = json.load(handle)
    status = summary["status"]
    assert status in {"converged", "max_iter"}, f"Unsuccessful candidate state: {status}"

    cells, genes, _, _ = torch.load(
        output_dir / f"{exp_name}_sencellgene-final.data",
        map_location="cpu", weights_only=False,
    )
    cell_ids = {int(value) for value in cells}
    gene_ids = [int(value) for value in genes]
    assert cell_ids, "The reference example must identify at least one SnC"
    assert gene_ids, "The reference example must retain at least one SnG"
    assert len(gene_ids) == len(set(gene_ids)), "Duplicate final SnG IDs"
    assert summary["snc_count"] == len(cell_ids), "Summary/final SnC count mismatch"
    assert summary["sng_count"] == len(gene_ids), "Summary/final SnG count mismatch"

    data = anndata.read_h5ad(output_dir / f"{exp_name}_new_data.h5ad", backed="r")
    try:
        n_cells, n_genes = data.shape
        cell_type_col = summary["config"]["cell_type_col"]
        assert all(n_genes <= value < n_genes + n_cells for value in cell_ids)
        assert all(0 <= value < n_genes for value in gene_ids)
        table_dir = output_dir / "Senescent_Tables"
        table1 = pd.read_csv(table_dir / "Cell_Table1_SnC_scores.csv",
                             dtype={"cell_name": str, "cell_type": str})
        assert len(table1) == n_cells, "Cell table row count differs from processed data"
        assert table1["cell_id"].tolist() == list(range(n_cells)), "Cell ID order changed"
        assert table1["cell_name"].tolist() == list(data.obs_names.astype(str)), "Cell names changed"
        expected_types = data.obs[cell_type_col].astype(str).tolist()
        assert table1["cell_type"].astype(str).tolist() == expected_types, "Cell types changed"
        expected_flags = [int(n_genes + row in cell_ids) for row in range(n_cells)]
        assert table1["ifSnCs"].tolist() == expected_flags, "Exported SnC IDs differ from final result"
        assert np.isfinite(table1["SnC scores"].to_numpy()).all(), "Nonfinite SnC scores"

        counts = pd.read_csv(table_dir / "Cell_Table2_SnCs_per_ct.csv", dtype={"cell_type": str})
        expected_counts = table1.groupby("cell_type").agg(
            number_of_cells=("cell_id", "size"), number_of_SnCs=("ifSnCs", "sum"),
        )
        actual_counts = counts.set_index("cell_type")[list(expected_counts.columns)]
        pd.testing.assert_frame_equal(
            actual_counts.sort_index(), expected_counts.sort_index(), check_dtype=False,
        )

        gene_table = pd.read_csv(table_dir / "Gene_Table1_SnG_scores_per_ct.csv", index_col=0)
        assert len(gene_table) == len(gene_ids), "SnG table row count mismatch"
        assert gene_table.index.astype(str).tolist() == list(data.var_names[gene_ids].astype(str)), "SnG names changed"
        assert np.isfinite(gene_table.to_numpy(dtype=float)).all(), "Nonfinite SnG scores"
        row_counts = {"Cell_Table1_SnC_scores.csv": len(table1),
                      "Cell_Table2_SnCs_per_ct.csv": len(counts),
                      "Gene_Table1_SnG_scores_per_ct.csv": len(gene_table)}
        for filename, required in {
            "Gene_Table2_DEG_ct_SnG_score.csv": {"gene", "cell_type", "SnG_score", "p_val", "logFC", "p_val_adj"},
            "Gene_Table3_gene_ct_count.csv": {"gene", "cell_type", "hallmarker"},
            "Gene_newTable3_gene_ct_count.csv": {"gene", "cell_type", "hallmarker"},
            "table2ByCelltype.csv": {"cell_type", "gene_count", "gene_list"},
            "table2ByGene.csv": {"gene", "cell_type_count", "cell_types"},
        }.items():
            table = pd.read_csv(table_dir / filename)
            assert required.issubset(table.columns), f"Missing columns in {filename}"
            row_counts[filename] = len(table)
    finally:
        data.file.close()

    evidence = {
        "workflow_status": "passed", "stopping_status": status,
        "converged": status == "converged", "final_epoch": summary["final_epoch"],
        "snc_count": len(cell_ids), "sng_count": len(gene_ids),
        "processed_cells": n_cells, "processed_genes": n_genes, "table_rows": row_counts,
    }
    evidence_path = output_dir / "validation_summary.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    print(f"reviewer example complete; evidence={evidence_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Experiment directory containing the final result")
    parser.add_argument("--exp-name", default="example")
    args = parser.parse_args()
    check_example(args.output_dir, args.exp_name)
