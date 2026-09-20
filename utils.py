import argparse
import os

import numpy as np
import numpy.ma as ma
import pandas as pd
import scanpy as sp
import torch
from torch_geometric.data import Data as Graphdata
from torch_geometric.utils import to_undirected
import networkx as nx
from tabulate import tabulate
from scipy import sparse as scsp


# ============================================================================
#                                  ARGS
# ============================================================================
def parse_args():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepSAS: senescent cell and gene identification from scRNA-seq')

    # ---- I/O ----
    parser.add_argument('--input_data_count', type=str,
                        default='example_data/example_data.h5ad',
                        help='Path to input data (h5ad format). '
                             'Default is the bundled example dataset.')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Base output directory')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Table generation only: load a specific iteration; '
                             'default loads the final result')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (used for output directory naming)')
    parser.add_argument('--device_index', type=int, default=0,
                        help='CUDA device index to use')
    parser.add_argument('--retrain', action='store_true',
                        help='Whether to retrain the GAT or load the saved one')
    # ---- Dataset-specific column names (new) ----
    parser.add_argument('--cell_type_col', type=str, default='clusters',
                        help="AnnData .obs column containing cell-type labels "
                             "(default: 'clusters')")
    parser.add_argument('--batch_col', type=str, default='Sample',
                        help="AnnData .obs column containing batch labels for "
                             "ComBat correction (default: 'Sample')")
    parser.add_argument('--batch_remove', action='store_true',
                        help='Run ComBat batch correction on --batch_col '
                             'before UMAP. Default off; set this flag to enable.')

    # ---- Optional phenotype-aware extension ----
    parser.add_argument('--phenotype_aware', action='store_true',
                        help='Also select SnCs within phenotype groups. The '
                             'phenotype entry-point enables this automatically.')
    parser.add_argument('--phenotype_col', type=str, default='Condition',
                        help="AnnData .obs phenotype column (default: 'Condition')")
    parser.add_argument('--use_hvg_deg', action='store_true',
                        help='Before DeepSAS gene selection, retain the union of '
                             'per-cell-type HVGs, marker genes, and L-R genes')
    parser.add_argument('--phenotype_hvg_count', type=int, default=1000,
                        help='HVGs selected per cell type when --use_hvg_deg is '
                             'enabled (default: 1000)')
    parser.add_argument('--min_snc_per_phenotype', type=int, default=1,
                        help='Minimum above-fence SnCs required to retain a '
                             'phenotype group (default: 1)')
    parser.add_argument('--phenotype_zscore_threshold', type=float, default=2.0,
                        help='Z-score cutoff for phenotype-specific SnG reporting '
                             '(default: 2.0)')

    # ---- Preprocessing ----
    parser.add_argument('--min_genes_per_cell', type=int, default=200,
                        help='Discard cells with fewer than this many detected '
                             'genes (default: 200)')
    parser.add_argument('--min_cells_per_gene', type=int, default=10,
                        help='Discard genes detected in fewer than this many '
                             'cells (default: 10)')
    parser.add_argument('--normalization_target_sum', type=float, default=1e4,
                        help='Target total count per observation before log1p '
                             'normalization (default: 10000)')
    parser.add_argument('--scale_max_value', type=float, default=10.0,
                        help='Clip scaled values above this value before PCA '
                             '(default: 10)')
    parser.add_argument('--umap_n_neighbors', type=int, default=10,
                        help='Number of neighbors in the graph used for UMAP '
                             '(default: 10)')
    parser.add_argument('--umap_n_pcs', type=int, default=40,
                        help='Number of principal components used to build the '
                             'UMAP neighbor graph (default: 40)')

    # ---- Model configuration ----
    parser.add_argument('--seed', type=int, default=40,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_genes', type=str, default='full',
                        help="Number of HVGs to use ('full', '8000', '3000', ...)")
    parser.add_argument('--ccc', type=str, default='type1',
                        choices=['type1', 'type3'],
                        help='Cell-cell edge type: type1 (binary), '
                             'type3 (no cell-cell edges)')
    parser.add_argument('--ccc_threshold', type=float, default=0.8,
                        help='CCC score threshold phi for binary cell-cell edges')
    parser.add_argument('--gene_set', type=str, default='full',
                        help="Senescence gene sets used to seed the SnG "
                             "candidates: 'full' for every gene set in "
                             '--marker_list, or one of its column names, '
                             "matched case-insensitively and combined with '+' "
                             "(e.g. 'senmayo', 'senmayo+fridman+cellage' for "
                             'the bundled list)')
    parser.add_argument('--emb_size', type=int, default=12,
                        help='Embedding dimension size')
    projection_group = parser.add_mutually_exclusive_group()
    projection_group.add_argument(
        '--type_specific_projections', dest='type_specific_projections',
        action='store_true',
        help='Use separate learned input projections for gene and cell nodes '
             '(default)')
    projection_group.add_argument(
        '--no_type_specific_projections', dest='type_specific_projections',
        action='store_false',
        help='Feed gene and cell embeddings directly to the shared GAT layers '
             'instead of the default type-specific projections')
    parser.set_defaults(type_specific_projections=True)
    parser.add_argument('--lr_panel', type=str, default=None,
                        help='Optional path to a user-supplied L-R panel CSV with '
                             'columns "ligand" and "receptor" (one row per L-R '
                             'pair). When omitted, the built-in 8-ligand SASP '
                             'panel (Supplementary Table S19) is used.')

    # ---- Senescence marker list and species ----
    parser.add_argument('--marker_list', type=str, default=None,
                        help='Path to the senescence marker CSV used to seed the '
                             'SnG candidates: one column per gene set, one gene '
                             'symbol per row, shorter columns padded with empty '
                             'cells. Column names are the values accepted by '
                             '--gene_set. Default: the bundled human '
                             'senescence_marker_list.csv.')
    parser.add_argument('--species', type=str, default='human',
                        choices=['human', 'mouse', 'rat', 'other'],
                        help='Species of the input data. The data and the marker '
                             'list must use the same gene-symbol nomenclature '
                             '(human HGNC symbols are uppercase, mouse and rat '
                             'MGI symbols are capitalized); this is checked right '
                             "after the data is loaded (default: 'human')")
    parser.add_argument('--min_marker_overlap', type=float, default=0.05,
                        help='Minimum fraction of marker genes that must be found '
                             'in the input data. Below this fraction the run stops '
                             'and asks for ortholog mapping or a species-matched '
                             '--marker_list (default: 0.05)')
    parser.add_argument('--skip_marker_check', action='store_true',
                        help='Report a failing marker/data symbol check as a '
                             'warning instead of stopping the run')

    # ---- Training parameters ----
    parser.add_argument('--gat_epoch', type=int, default=30,
                        help='Number of epochs to train the GAT model')
    parser.add_argument('--gat_learning_rate', type=float, default=0.001,
                        help='Adam learning rate for the graph autoencoder')
    parser.add_argument('--gat_hidden_size', type=int, default=32,
                        help='Hidden width of the two-layer, single-head GAT')
    parser.add_argument('--gat_dropout', type=float, default=0.6,
                        help='Attention dropout during GAT training')
    parser.add_argument('--cell_optim_epoch', type=int, default=50,
                        help='Number of epochs for cell embedding optimization')
    parser.add_argument('--cell_hidden_size', type=int, default=128,
                        help='Hidden width of the cell embedding network')
    parser.add_argument('--distance_levels', type=float, nargs=3,
                        default=(0.0, 0.0, 4.0),
                        metavar=('WITHIN_SNC', 'BETWEEN_SNC', 'WITHIN_NON_SNC'),
                        help='Initial learnable target distances for within-type '
                             'SnC, between-type SnC, and same-type non-SnC terms')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Adam weight decay for cell embedding optimization')
    parser.add_argument('--lr_decay', type=float, default=0.85,
                        help='Learning-rate multiplier after each outer iteration')
    parser.add_argument('--iqr_multiplier', type=float, default=1.5,
                        help='Upper-fence multiplier for SnC scores and for SnG '
                             'scores when --sng_update_mode=iqr')
    parser.add_argument('--sng_update_mode', type=str, default='iqr',
                        choices=['iqr', 'fixed10'],
                        help='SnG candidate update rule: dynamic IQR-based '
                             'replacement (default) or a fixed-count policy '
                             'requesting 10 replacements per iteration')
    parser.add_argument('--min_snc_per_type', type=int, default=1,
                        help='Minimum candidate SnCs retained in a cell type')
    # ---- Contrastive-learning convergence (new) ----
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Hard ceiling on the number of contrastive '
                             'refinement iterations; reaching this limit is '
                             'reported separately from convergence.')
    parser.add_argument('--convergence_tol', type=float, default=0.9,
                        help='Jaccard tolerance for SnC and SnG sets between '
                             'adjacent iterations. The loop exits when both '
                             'Jaccard(SnC_t-1, SnC_t) >= tol and '
                             'Jaccard(SnG_t-1, SnG_t) >= tol.')

    # ---- Downstream reporting (does not alter SnC/SnG prediction) ----
    parser.add_argument('--deg_min_snc', type=int, default=6,
                        help='Minimum SnCs required in a cell type for the '
                             'downstream DEG comparison (default: 6)')
    parser.add_argument('--deg_min_control', type=int, default=2,
                        help='Minimum non-SnC controls required in a cell type '
                             'for the downstream DEG comparison (default: 2)')
    parser.add_argument('--deg_min_logfc', type=float, default=0.25,
                        help='Minimum log fold change retained in the combined '
                             'DEG/SnG table (default: 0.25)')

    args = parser.parse_args()

    if args.emb_size <= 0:
        parser.error('Embedding size must be positive')
    if args.gat_epoch <= 0 or args.cell_optim_epoch <= 0:
        parser.error('Number of epochs must be positive')
    if not (0.0 < args.convergence_tol <= 1.0):
        parser.error('--convergence_tol must be in (0, 1]')
    if args.max_iter < 1:
        parser.error('--max_iter must be >= 1')
    if args.epoch is not None and args.epoch < 0:
        parser.error('--epoch must be nonnegative')
    if not (0 <= args.ccc_threshold <= 1):
        parser.error('--ccc_threshold must be in [0, 1]')
    if not (0 <= args.min_marker_overlap <= 1):
        parser.error('--min_marker_overlap must be in [0, 1]')
    if not (0 <= args.gat_dropout < 1):
        parser.error('--gat_dropout must be in [0, 1)')
    if (args.gat_hidden_size < 1 or args.cell_hidden_size < 1
            or args.min_snc_per_type < 1):
        parser.error('Hidden sizes and minimum SnCs per type must be positive')
    if (args.min_genes_per_cell < 1 or args.min_cells_per_gene < 1
            or args.umap_n_neighbors < 1 or args.umap_n_pcs < 1
            or args.phenotype_hvg_count < 1):
        parser.error('Preprocessing count parameters must be positive')
    if args.deg_min_snc < 1 or args.deg_min_control < 1:
        parser.error('DEG minimum group sizes must be positive')
    if args.min_snc_per_phenotype < 1:
        parser.error('--min_snc_per_phenotype must be positive')
    if not np.isfinite(args.distance_levels).all():
        parser.error('--distance_levels values must be finite')
    for name in ('learning_rate', 'gat_learning_rate', 'lr_decay',
                 'normalization_target_sum', 'scale_max_value'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error('--' + name + ' must be finite and positive')
    for name in ('iqr_multiplier', 'weight_decay'):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error('--' + name + ' must be finite and nonnegative')
    if not np.isfinite(args.deg_min_logfc):
        parser.error('--deg_min_logfc must be finite')
    if not np.isfinite(args.phenotype_zscore_threshold):
        parser.error('--phenotype_zscore_threshold must be finite')

    return args


# ============================================================================
#                             DATA LOADERS
# ============================================================================
def _build_cluster_index(adata, ct_name):
    """Shared helper: build cluster_cell_ls and cell_cluster_arr from a column."""
    if ct_name not in adata.obs.columns:
        raise KeyError(
            f"Cell-type column '{ct_name}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}. "
            f"Pass --cell_type_col to point DeepSAS at the correct column."
        )

    celltype_names = list(adata.obs[ct_name].value_counts().index)
    print(f"Number of cell types: {len(celltype_names)}")
    print(f"Cell-type names: {celltype_names}")

    cluster_cell_ls = []
    cell_cluster_arr = np.zeros(adata.shape[0], dtype=np.int64)
    all_indexs = np.arange(adata.shape[0])
    for i, name in enumerate(celltype_names):
        cell_indexs = all_indexs[adata.obs[ct_name] == name]
        cluster_cell_ls.append(cell_indexs)
        cell_cluster_arr[cell_indexs] = i

    outputs = [[celltype_names[i], len(j)] for i, j in enumerate(cluster_cell_ls)]
    print(tabulate(outputs))
    return cluster_cell_ls, cell_cluster_arr, celltype_names


def load_example_data(path='example_data/example_data.h5ad', ct_name='clusters',
                      min_genes_per_cell=200, min_cells_per_gene=10):
    """Load the bundled small public example dataset."""
    print(f'Load example data from {path} ...')
    adata = sp.read_h5ad(path)
    sp.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sp.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    print(f'\tNumber of cells: {adata.shape[0]}\n\tNumber of genes: {adata.shape[1]}')
    cluster_cell_ls, cell_cluster_arr, celltype_names = \
        _build_cluster_index(adata, ct_name)
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data1(path, ct_name='clusters', min_genes_per_cell=200,
               min_cells_per_gene=10):
    """Generic AnnData loader. Path must be supplied explicitly."""
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            f"Pass --input_data_count <path> on the command line."
        )
    print(f'Loading data from {path} ...')
    adata = sp.read_h5ad(path)
    sp.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sp.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    print(f'Number of cells: {adata.shape[0]}\nNumber of genes: {adata.shape[1]}')
    cluster_cell_ls, cell_cluster_arr, celltype_names = \
        _build_cluster_index(adata, ct_name)
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_rep(path, ct_name='clusters', min_genes_per_cell=200,
                  min_cells_per_gene=10):
    """Loader for subsampled replicate datasets used in the robustness analysis.

    The path layout is now configurable; callers must pass an explicit path
    (e.g. './data4_robust_test/rep1_subsample_half.h5ad').
    """
    return load_data1(path, ct_name=ct_name,
                      min_genes_per_cell=min_genes_per_cell,
                      min_cells_per_gene=min_cells_per_gene)


# ============================================================================
#                        L-R PANEL (configurable now)
# ============================================================================
# Built-in SASP-focused 8-ligand panel (Supplementary Table S19).
# Kept identical to the original v1 panel for backward compatibility.
_BUILTIN_LR_PANEL = {
    "IL6":      ["IL6ST", "IL6R", "HRH1", "F3"],
    "CXCL10":   ["DPP4", "CCR3", "GRM7", "ADRA2A", "SDC4", "CXCR3", "MTNR1A"],
    "IL1B":     ["ADRB2", "SIGIRR", "IL1RAP", "IL1R2", "IL1R1"],
    "CCL2":     ["ACKR1", "CCR2", "ACKR4", "CCR10", "CCR3", "CCR5", "CCR1",
                 "CCR4", "ACKR2"],
    "CCL5":     ["SDC1", "GPR75", "ADRA2A", "CCR3", "CCRL2", "CCR5", "GRM7",
                 "SDC4", "ACKR1", "MTNR1A", "CCR1", "CCR4", "CXCR3", "ACKR4",
                 "ACKR2"],
    "HMGB1":    ["TLR4", "TLR9", "AGER", "CXCR4", "SDC1", "THBD", "TLR2", "CD163"],
    "TNF":      ["PTPRS", "CELSR2", "RIPK1", "FLT4", "TRAF2", "FAS", "ICOS",
                 "NOTCH1", "TNFRSF1B", "TRADD", "TRPM2", "FFAR2", "VSIR",
                 "TNFRSF21", "TNFRSF1A"],
    "SERPINE1": ["LRP1", "ITGAV", "PLAUR", "ITGB5", "LRP2"],
}


def get_ccc_markers(lr_panel_path=None):
    """Return (ligand_receptor_dict, gene_set).

    If `lr_panel_path` is provided, the panel is loaded from a CSV with two
    columns 'ligand' and 'receptor' (one row per L-R pair, repeating ligands
    permitted). Otherwise the built-in 8-ligand SASP panel is returned.
    """
    if lr_panel_path is None:
        ligand_receptor_dict = {k: list(v) for k, v in _BUILTIN_LR_PANEL.items()}
    else:
        if not os.path.exists(lr_panel_path):
            raise FileNotFoundError(f'L-R panel CSV not found: {lr_panel_path}')
        df = pd.read_csv(lr_panel_path)
        if not {'ligand', 'receptor'}.issubset(df.columns):
            raise ValueError(
                f'L-R panel CSV must contain columns "ligand" and "receptor". '
                f'Got: {list(df.columns)}'
            )
        ligand_receptor_dict = {}
        for _, row in df.iterrows():
            l, r = str(row['ligand']).strip(), str(row['receptor']).strip()
            if not l or not r:
                continue
            ligand_receptor_dict.setdefault(l, []).append(r)
        print(f'Loaded {len(df)} L-R pairs ({len(ligand_receptor_dict)} ligands) '
              f'from {lr_panel_path}')

    new_gene_set = set()
    for ligand, receptors in ligand_receptor_dict.items():
        new_gene_set.add(ligand)
        new_gene_set.update(receptors)
    return ligand_receptor_dict, new_gene_set


# Built-in legacy marker seed retained unchanged from the original DeepSAS v1
# workflow for result compatibility. It is used only when ``gene_set=full``.
_BUILTIN_CELL_CYCLE_MARKERS = (
    "CDKN1A", "CDKN2A", "TP53", "GADD45A", "IGFBP7", "SERPINE1", "GLB1",
    "IL6", "IL8", "MMP1", "MMP3",
)


def get_cellcyle_markers():
    """Return a copy of the built-in legacy cell-cycle marker seed."""
    return list(_BUILTIN_CELL_CYCLE_MARKERS)


# Bundled human marker table. A user-supplied CSV that carries these columns
# keeps the original v1 grouping: GO is reported but not used for seeding, and
# the legacy cell-cycle seed above is appended.
BUILTIN_MARKER_FILE = 'senescence_marker_list.csv'
_BUILTIN_MARKER_COLUMNS = ('SenMayo', 'FRIDMAN', 'CellAge', 'GO')
_BUILTIN_SEED_COLUMNS = ('SenMayo', 'FRIDMAN', 'CellAge')
_CELL_CYCLE_GROUP_NAME = 'Cell-cycle markers'


def resolve_marker_list_path(path=None):
    """Resolve the senescence marker CSV.

    ``None`` selects the bundled human list: the copy in the working directory
    when there is one, otherwise the copy shipped next to this module, so the
    pipeline can also be started from another directory.
    """
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Senescence marker list CSV not found: {path}. '
                'Pass --marker_list <path> with a readable CSV.')
        return path
    if os.path.exists(BUILTIN_MARKER_FILE):
        return BUILTIN_MARKER_FILE
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        BUILTIN_MARKER_FILE)


def read_marker_list(path):
    """Read a marker CSV into an ordered ``{gene-set name: [genes]}`` mapping.

    One column per gene set, one gene symbol per row, shorter columns padded
    with empty cells. Duplicates and blank cells are dropped.
    """
    table = pd.read_csv(path)
    groups = {}
    for col_name, column in table.items():
        genes = [str(g).strip() for g in column[column.notnull()]]
        genes = [g for g in genes if g]
        if not genes:
            continue
        groups[str(col_name).strip()] = list(dict.fromkeys(genes))
    if not groups:
        raise ValueError(
            f'No marker genes found in {path}. The file must be a CSV with one '
            'column per gene set and one gene symbol per row.')
    return groups


def load_markers(args):
    """Return the marker gene sets that seed the initial SnG candidates."""
    path = resolve_marker_list_path(getattr(args, 'marker_list', None))
    groups = read_marker_list(path)
    species = getattr(args, 'species', 'human')
    print(f'Senescence marker list: {path} (species: {species})')

    if set(_BUILTIN_MARKER_COLUMNS).issubset(groups):
        # Bundled layout: GO is not used for seeding, and the legacy human
        # cell-cycle seed of DeepSAS v1 is appended.
        groups = {name: groups[name] for name in _BUILTIN_SEED_COLUMNS}
        if species == 'human':
            groups[_CELL_CYCLE_GROUP_NAME] = get_cellcyle_markers()
        else:
            print(f'Species is {species}: the built-in human cell-cycle marker '
                  'seed is not added. Include the orthologs in --marker_list if '
                  'they are wanted.')

    sel = getattr(args, 'gene_set', 'full')
    if sel != 'full':
        available = {name.lower(): name for name in groups}
        selected = {}
        for wanted in str(sel).split('+'):
            name = available.get(wanted.strip().lower())
            if name is None:
                raise ValueError(
                    f'--gene_set {sel!r} requests the gene set {wanted.strip()!r}, '
                    f'which is not a column of {path}. Available gene sets: '
                    f"full, {', '.join(groups)} (combine several with '+').")
            selected[name] = groups[name]
        groups = selected

    print('Number of genes in each marker list:')
    print(tabulate([list(groups), [len(genes) for genes in groups.values()]],
                   headers='firstrow'))
    return [list(genes) for genes in groups.values()]


# ============================================================================
#                      SPECIES / GENE-SYMBOL COMPATIBILITY
# ============================================================================
# Gene symbols are species specific: human HGNC symbols are uppercase
# (CDKN1A), mouse and rat MGI/RGD symbols are capitalized (Cdkn1a). A marker
# list written for one species matches almost nothing in data from another, so
# the run is stopped as soon as the data is loaded instead of collapsing to an
# empty senescence seed much later.
_SPECIES_SYMBOL_STYLE = {
    'human': 'upper',
    'mouse': 'title',
    'rat': 'title',
    'other': None,
}

_STYLE_LABEL = {
    'upper': 'UPPERCASE symbols, e.g. CDKN1A (human HGNC style)',
    'title': 'Capitalized symbols, e.g. Cdkn1a (mouse/rat MGI style)',
    'lower': 'lowercase symbols, e.g. cdkn1a',
    'mixed': 'no dominant capitalization convention',
}


def _symbol_style(symbol):
    letters = [c for c in str(symbol) if c.isalpha()]
    if not letters:
        return 'mixed'
    if all(c.isupper() for c in letters):
        return 'upper'
    if all(c.islower() for c in letters):
        return 'lower'
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return 'title'
    return 'mixed'


def _dominant_symbol_style(symbols, min_fraction=0.6):
    """Most common capitalization style, or 'mixed' when none dominates."""
    counts = {}
    for symbol in symbols:
        style = _symbol_style(symbol)
        counts[style] = counts.get(style, 0) + 1
    if not counts:
        return 'mixed'
    style, count = max(counts.items(), key=lambda item: item[1])
    if count < min_fraction * sum(counts.values()):
        return 'mixed'
    return style


def check_lr_panel_match(data_symbols, args):
    """Report how much of the ligand-receptor panel is present in the data.

    Ligand-receptor symbols are species specific just like the marker list, so
    a human panel used on mouse data yields no CCC edges at all. This is a
    warning rather than an error, because cell-cell edges are optional.
    """
    lr_panel_path = getattr(args, 'lr_panel', None)
    species = getattr(args, 'species', 'human')
    min_overlap = float(getattr(args, 'min_marker_overlap', 0.05))
    data_set = set(data_symbols)
    lr_genes = sorted(get_ccc_markers(lr_panel_path)[1])
    matched = [gene for gene in lr_genes if gene in data_set]
    overlap = len(matched) / len(lr_genes) if lr_genes else 0.
    uses_ccc = getattr(args, 'ccc', 'type1') != 'type3'

    print(f'Ligand-receptor panel genes found in the data: {len(matched)}/'
          f'{len(lr_genes)} ({overlap:.1%})')
    if uses_ccc and overlap < min_overlap:
        panel = ('the built-in human ligand-receptor panel'
                 if lr_panel_path is None else f'the panel in {lr_panel_path}')
        print(f'WARNING: only {len(matched)} of {len(lr_genes)} genes of {panel} '
              f'({overlap:.1%}) are present in the input data, so --ccc '
              f'{getattr(args, "ccc", "type1")} will add few or no cell-cell '
              f'edges. Pass --lr_panel with a panel whose gene symbols match '
              f'the data ({species}), or --ccc type3 to run without cell-cell '
              'edges.')
    return {
        'lr_panel': lr_panel_path,
        'lr_panel_genes': len(lr_genes),
        'lr_matched_genes': len(matched),
        'lr_matched_fraction': overlap,
    }


def check_species_gene_match(adata, markers_ls, args):
    """Verify that the data and the marker list use the same gene symbols.

    Returns a dict of match statistics. Raises ``ValueError`` when fewer than
    ``--min_marker_overlap`` of the marker genes are present in the data,
    unless ``--skip_marker_check`` downgrades the failure to a warning.
    """
    species = getattr(args, 'species', 'human')
    min_overlap = float(getattr(args, 'min_marker_overlap', 0.05))
    marker_symbols = sorted({str(g) for group in markers_ls for g in group})
    if not marker_symbols:
        raise ValueError('The senescence marker list contains no genes.')

    data_symbols = [str(g) for g in adata.var_names]
    data_set = set(data_symbols)
    matched = [g for g in marker_symbols if g in data_set]
    overlap = len(matched) / len(marker_symbols)

    data_upper = {g.upper() for g in data_set}
    matched_ignoring_case = [g for g in marker_symbols if g.upper() in data_upper]

    data_style = _dominant_symbol_style(data_symbols)
    marker_style = _dominant_symbol_style(marker_symbols)
    expected_style = _SPECIES_SYMBOL_STYLE.get(species)

    print(f'Marker genes found in the data: {len(matched)}/{len(marker_symbols)} '
          f'({overlap:.1%}); required: {min_overlap:.1%}')
    print(f'\tData gene symbols: {_STYLE_LABEL[data_style]}')
    print(f'\tMarker gene symbols: {_STYLE_LABEL[marker_style]}')

    stats = {
        'species': species,
        'marker_genes': len(marker_symbols),
        'matched_genes': len(matched),
        'matched_fraction': overlap,
        'matched_ignoring_case': len(matched_ignoring_case),
        'data_symbol_style': data_style,
        'marker_symbol_style': marker_style,
    }
    # The L-R panel is a second species-specific gene list, so it is reported
    # here as well: a panel that does not match the data leaves the CCC part of
    # the graph empty without stopping the run.
    stats.update(check_lr_panel_match(data_set, args))

    if overlap >= min_overlap:
        if expected_style is not None and data_style != expected_style:
            print(f'WARNING: --species {species} expects '
                  f'{_STYLE_LABEL[expected_style]}, but the data uses '
                  f'{_STYLE_LABEL[data_style]}. Check that --species is correct.')
        return stats

    lines = [
        f'Only {len(matched)} of {len(marker_symbols)} senescence marker genes '
        f'({overlap:.1%}) are present in the input data, which is below '
        f'--min_marker_overlap ({min_overlap:.1%}).',
        f'  --species            : {species}'
        + (f' (expects {_STYLE_LABEL[expected_style]})'
           if expected_style is not None else ''),
        f'  data gene symbols    : {_STYLE_LABEL[data_style]}, '
        f'{len(data_set)} genes, e.g. {", ".join(data_symbols[:5])}',
        f'  marker gene symbols  : {_STYLE_LABEL[marker_style]}, '
        f'e.g. {", ".join(marker_symbols[:5])}',
    ]
    if len(matched_ignoring_case) > len(matched):
        lines.append(
            f'  Ignoring capitalization, {len(matched_ignoring_case)} marker '
            'genes would match, so the two sets differ in nomenclature rather '
            'than in content. DeepSAS does not rename genes for you, because '
            'capitalization alone is not a valid ortholog mapping.')
    lines += [
        'How to proceed:',
        '  1. Map the data to the species of the marker list (for example with '
        'Ensembl BioMart, pyensembl or babelgene) and rerun with the mapped '
        'symbols; or',
        '  2. Pass --marker_list <csv> with a senescence hallmark gene list for '
        'this species (one column per gene set, one gene symbol per row), '
        'together with --species and, for cell-cell edges, --lr_panel; or',
        '  3. Pass --skip_marker_check to continue anyway; the senescence seed '
        'will then be nearly empty and the run is expected to return no SnCs.',
    ]
    message = '\n'.join(lines)

    if getattr(args, 'skip_marker_check', False):
        print('WARNING: --skip_marker_check is set; continuing despite the '
              'marker/data mismatch.')
        print(message)
        return stats
    raise ValueError(message)


# ============================================================================
#                       GENE / GRAPH CONSTRUCTION
# ============================================================================
def get_highly_genes(adata, n_genes, normalization_target_sum=1e4):
    new_data = adata.copy()
    sp.pp.normalize_total(new_data, target_sum=normalization_target_sum)
    sp.pp.log1p(new_data)
    sp.pp.highly_variable_genes(new_data, n_top_genes=n_genes)
    return list(new_data.var[new_data.var['highly_variable'] == True].index)


def combine_genes(adata, markers_ls, args):
    markers_set = set(g for markers in markers_ls for g in markers)
    print('Total marker genes: ', len(markers_set))

    if args.n_genes == 'full':
        print('Using all genes!')
        highly_genes = list(adata.var.index)
    else:
        print(f'Using {args.n_genes} highly variable genes!')
        highly_genes = get_highly_genes(
            adata, int(args.n_genes),
            getattr(args, 'normalization_target_sum', 1e4))
    print('Highly variable gene count: ', len(highly_genes))

    highly_genes = sorted(list(set(highly_genes) - markers_set))
    print('Highly variable genes after dropping duplicates with senescence markers: ',
          len(highly_genes))

    sen_gene_ls = []
    if scsp.issparse(adata.X):
        cell_gene = adata.X.toarray()
    else:
        cell_gene = adata.X
    gene_names = list(adata.var.index)
    for gene in markers_set:
        if gene in gene_names:
            gene_index = gene_names.index(gene)
            if max(cell_gene[:, gene_index]) != 0:
                sen_gene_ls.append(gene)

    filtered_highly_genes = []
    for gene in highly_genes:
        if gene in gene_names:
            gene_index = gene_names.index(gene)
            if max(cell_gene[:, gene_index]) != 0:
                filtered_highly_genes.append(gene)
    if len(filtered_highly_genes) != len(highly_genes):
        print('Some highly variable genes have zero expression in all cells.')

    sen_gene_ls = sorted(sen_gene_ls)
    gene_names = filtered_highly_genes + sen_gene_ls
    print('Total gene num:', len(gene_names))

    adata_gene_names = list(adata.var.index)
    gene_indexs = [adata_gene_names.index(name) for name in gene_names]
    new_data = adata[:, gene_indexs]
    if gene_names != list(new_data.var_names):
        raise AssertionError('Gene-index mismatch')

    markers_index = []
    for markers in markers_ls:
        indexs = [gene_names.index(m) for m in markers if m in gene_names]
        markers_index.append(indexs)

    sen_gene_ls = [gene_names.index(i) for i in sen_gene_ls]
    nonsen_gene_ls = [gene_names.index(i) for i in filtered_highly_genes]
    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def process_data(adata, cluster_cell_ls, cell_cluster_arr, args, markers_ls=None):
    if markers_ls is None:
        markers_ls = load_markers(args)
    return combine_genes(adata, markers_ls, args)


def build_graph_nx(adata, gene_cell, cell_cluster_arr, sen_gene_ls,
                   nonsen_gene_ls, gene_names, args):
    """Construct the binary cell-gene graph and optional binary CCC edges."""
    lr_panel_path = getattr(args, 'lr_panel', None)
    g_index, c_index = np.nonzero(gene_cell)
    print('Cell-gene graph, the number of edges:', len(g_index))
    gene_num = gene_cell.shape[0]
    c_index = c_index + gene_num

    if args.ccc == 'type1':
        print('Adding cell-cell edges with binary weights ...')
        adj_matrix = build_ccc_graph(
            gene_cell, gene_names, lr_panel_path, args.ccc_threshold)
        i1, i2 = np.nonzero(adj_matrix)
        print('CCC graph, the number of edges:', len(i1))
        i1, i2 = i1 + gene_num, i2 + gene_num
        edge_index = torch.tensor(np.array([np.concatenate([g_index, i1]),
                                            np.concatenate([c_index, i2])]),
                                  dtype=torch.long)
    else:
        print('No cell-cell edges (type3) ...')
        edge_index = torch.tensor(np.array([g_index, c_index]), dtype=torch.long)

    graph_nx = nx.Graph(edge_index.T.tolist())
    for i in range(gene_num):
        graph_nx.nodes[i]['type'] = 'g'
        graph_nx.nodes[i]['index'] = i
        graph_nx.nodes[i]['name'] = gene_names[i]
        graph_nx.nodes[i]['is_sen'] = i in sen_gene_ls
    cell_names = list(adata.obs.index)
    for i in range(gene_cell.shape[1]):
        graph_nx.nodes[i + gene_num]['type'] = 'c'
        graph_nx.nodes[i + gene_num]['cluster'] = cell_cluster_arr[i]
        graph_nx.nodes[i + gene_num]['index'] = i + gene_num
        graph_nx.nodes[i + gene_num]['name'] = cell_names[i]
    return graph_nx, edge_index


def add_nx_embedding(graph_nx, gene_embed, cell_embed):
    for i in range(gene_embed.shape[0]):
        graph_nx.nodes[i]['emb'] = gene_embed[i].detach().cpu()
    for i in range(cell_embed.shape[0]):
        graph_nx.nodes[i + gene_embed.shape[0]]['emb'] = cell_embed[i].detach().cpu()
    return graph_nx


def build_graph_pyg(gene_cell, gene_embed, cell_embed, edge_indexs):
    print('Building PyG graph')
    y = [True] * gene_cell.shape[0] + [False] * gene_cell.shape[1]
    y = torch.tensor(y)
    print('edge index: ', edge_indexs.shape)
    x = torch.cat([gene_embed, cell_embed]).detach()
    print('node feature: ', x.shape)

    edge_index = to_undirected(edge_indexs)
    graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)

    print('Pyg graph:', graph_pyg)
    print('graph.is_directed():', graph_pyg.is_directed())
    return graph_pyg


# ============================================================================
#                       L-R PROBABILITY (SoptSC-adapted)
# ============================================================================
def build_ccc_matrix(expression_matrix, gene_names, lr_panel_path):
    """Average per-pair L-R probabilities as defined in Methods Section 1.2."""
    ligand_receptor_dict = get_ccc_markers(lr_panel_path)[0]
    ccc_matrix = None
    pair_count = 0
    for ligand, receptors in ligand_receptor_dict.items():
        if ligand not in gene_names:
            continue
        ligand_exp = expression_matrix[:, gene_names.index(ligand)].reshape(-1, 1)
        receptor_indexs = [gene_names.index(r) for r in receptors if r in gene_names]
        for receptor_index in receptor_indexs:
            receptor_exp = expression_matrix[:, receptor_index].reshape(1, -1)
            pair_score = ligand_exp * receptor_exp
            masked = ma.masked_where(pair_score == 0, pair_score)
            pair_score = np.exp(-1 / masked).filled(0)
            ccc_matrix = (pair_score if ccc_matrix is None
                          else ccc_matrix + pair_score)
            pair_count += 1
    if pair_count:
        ccc_matrix = ccc_matrix / pair_count
    return ccc_matrix


def convert_to_adj(ccc_matrix, threshold):
    symmetric = 0.5 * (ccc_matrix + ccc_matrix.T)
    mask = symmetric >= threshold
    return np.where(mask, symmetric, 0)


def build_ccc_graph(gene_cell, gene_names, lr_panel_path, threshold):
    """gene_cell is (gene x cell). Internally transposed to (cell x gene)."""
    ccc_matrix = build_ccc_matrix(gene_cell.T, gene_names, lr_panel_path)
    if ccc_matrix is None:
        # Edge case: none of the L-R genes found in the dataset
        n = gene_cell.shape[1]
        return np.zeros((n, n))
    adj_matrix = convert_to_adj(ccc_matrix, threshold)
    return adj_matrix


# ============================================================================
#               SET-OVERLAP HELPERS (legacy + new Jaccard)
# ============================================================================
def get_sencell_cover(old_sencell_dict, sencell_dict):
    """Legacy: |old ∩ new| / |new|. Retained for backward compatibility."""
    set1 = set(old_sencell_dict.keys())
    set2 = set(sencell_dict.keys())
    inter = set1.intersection(set2)
    if len(set2) == 0:
        return 0.0
    cover = len(inter) / len(set2)
    print('sencell cover:', cover)
    return cover


def get_sengene_cover(old_sengene_ls, sengene_ls):
    """Legacy: |old ∩ new| / |new|."""
    if isinstance(old_sengene_ls, torch.Tensor):
        old_sengene_ls = old_sengene_ls.tolist()
    if isinstance(sengene_ls, torch.Tensor):
        sengene_ls = sengene_ls.tolist()
    set1, set2 = set(old_sengene_ls), set(sengene_ls)
    inter = set1.intersection(set2)
    if len(set2) == 0:
        return 0.0
    cover = len(inter) / len(set2)
    print('sengene cover:', cover)
    return cover


def jaccard(set_a, set_b):
    """Symmetric set similarity, used for the new convergence check."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def get_sencell_jaccard(old_sencell_dict, sencell_dict):
    return jaccard(old_sencell_dict.keys(), sencell_dict.keys())


def get_sengene_jaccard(old_sengene_ls, sengene_ls):
    if isinstance(old_sengene_ls, torch.Tensor):
        old_sengene_ls = old_sengene_ls.tolist()
    if isinstance(sengene_ls, torch.Tensor):
        sengene_ls = sengene_ls.tolist()
    return jaccard(old_sengene_ls, sengene_ls)


def get_sencell_intersection(old_sencell_dict, sencell_dict):
    set1 = set(old_sencell_dict.keys())
    set2 = set(sencell_dict.keys())
    return {i: sencell_dict[i] for i in (set1 & set2)}


# ============================================================================
#                                Misc utils
# ============================================================================
def save_objs(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print('obj saved', path)


def load_objs(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def caculate_GSEA(adata, args, use_onemarker=False, one_marker=None):
    import gseapy as gp
    gene_expression_df = pd.DataFrame(
        adata.X.T, index=adata.var.index, columns=adata.obs.index)
    if use_onemarker:
        all_marker_genes = [one_marker]
    else:
        all_marker_genes = load_markers(args)
        all_marker_genes = list(set(j for i in all_marker_genes for j in i))
    gene_sets = {'GeneSet1': all_marker_genes}
    ssgsea_results = gp.ssgsea(data=gene_expression_df, gene_sets=gene_sets,
                               sample_norm_method='rank', outdir=None)
    return ssgsea_results.res2d
