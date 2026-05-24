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
from numba import jit


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
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (used for output directory naming)')
    parser.add_argument('--device_index', type=int, default=0,
                        help='CUDA device index to use')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='Whether to retrain the GAT or load the saved one')
    parser.add_argument('--timestamp', type=str, default='',
                        help='Optional timestamp tag for the output directory')

    # ---- Dataset-specific column names (new) ----
    parser.add_argument('--cell_type_col', type=str, default='clusters',
                        help="AnnData .obs column containing cell-type labels "
                             "(default: 'clusters')")
    parser.add_argument('--batch_col', type=str, default='Sample',
                        help="AnnData .obs column containing batch labels for "
                             "ComBat correction (default: 'Sample')")
    parser.add_argument('--batch_remove', action='store_true', default=False,
                        help='Run ComBat batch correction on --batch_col '
                             'before UMAP. Default off; set this flag to enable.')

    # ---- Model configuration ----
    parser.add_argument('--seed', type=int, default=40,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_genes', type=str, default='full',
                        help="Number of HVGs to use ('full', '8000', '3000', ...)")
    parser.add_argument('--ccc', type=str, default='type1',
                        choices=['type1', 'type3'],
                        help='Cell-cell edge type: type1 (binary), '
                             'type3 (no cell-cell edges)')
    parser.add_argument('--gene_set', type=str, default='full',
                        help='Senescence gene-set source (full, senmayo, fridman, '
                             'cellage, senmayo+fridman, senmayo+cellage, '
                             'senmayo+fridman+cellage)')
    parser.add_argument('--emb_size', type=int, default=12,
                        help='Embedding dimension size')
    parser.add_argument('--use_autoencoder', action='store_true', default=False,
                        help='Use the optional reduction autoencoder for initial '
                             'embeddings instead of the default UMAP (Methods 1.2)')
    parser.add_argument('--lr_panel', type=str, default=None,
                        help='Optional path to a user-supplied L-R panel CSV with '
                             'columns "ligand" and "receptor" (one row per L-R '
                             'pair). When omitted, the built-in 8-ligand SASP '
                             'panel (Supplementary Table S19) is used.')

    # ---- Training parameters ----
    parser.add_argument('--gat_epoch', type=int, default=30,
                        help='Number of epochs to train the GAT model')
    parser.add_argument('--sencell_num', type=int, default=600,
                        help='Number of senescent cells to use')
    parser.add_argument('--sengene_num', type=int, default=200,
                        help='Number of senescence-associated genes to use')
    parser.add_argument('--sencell_epoch', type=int, default=40,
                        help='Number of epochs to train the Sencell model')
    parser.add_argument('--cell_optim_epoch', type=int, default=50,
                        help='Number of epochs for cell embedding optimization')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--batch_id', type=int, default=0,
                        help='Batch ID for processing')

    # ---- Contrastive-learning convergence (new) ----
    parser.add_argument('--max_iter', type=int, default=10,
                        help='Hard ceiling on the number of contrastive '
                             'refinement iterations (Methods 1.5). The loop '
                             'normally exits well before this on the convergence '
                             'criterion.')
    parser.add_argument('--convergence_tol', type=float, default=0.99,
                        help='Jaccard tolerance for SnC and SnG sets between '
                             'adjacent iterations. The loop exits when both '
                             'Jaccard(SnC_t-1, SnC_t) >= tol and '
                             'Jaccard(SnG_t-1, SnG_t) >= tol.')

    args = parser.parse_args()

    if args.emb_size <= 0:
        parser.error('Embedding size must be positive')
    if args.gat_epoch <= 0 or args.sencell_epoch <= 0 or args.cell_optim_epoch <= 0:
        parser.error('Number of epochs must be positive')
    if not (0.0 < args.convergence_tol <= 1.0):
        parser.error('--convergence_tol must be in (0, 1]')
    if args.max_iter < 1:
        parser.error('--max_iter must be >= 1')

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


def load_example_data(path='example_data/example_data.h5ad', ct_name='clusters'):
    """Load the bundled small public example dataset."""
    print(f'Load example data from {path} ...')
    adata = sp.read_h5ad(path)
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)
    print(f'\tNumber of cells: {adata.shape[0]}\n\tNumber of genes: {adata.shape[1]}')
    cluster_cell_ls, cell_cluster_arr, celltype_names = \
        _build_cluster_index(adata, ct_name)
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data1(path, ct_name='clusters'):
    """Generic AnnData loader. Path must be supplied explicitly."""
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            f"Pass --input_data_count <path> on the command line."
        )
    print(f'Loading data from {path} ...')
    adata = sp.read_h5ad(path)
    sp.pp.filter_cells(adata, min_genes=200)
    sp.pp.filter_genes(adata, min_cells=10)
    print(f'Number of cells: {adata.shape[0]}\nNumber of genes: {adata.shape[1]}')
    cluster_cell_ls, cell_cluster_arr, celltype_names = \
        _build_cluster_index(adata, ct_name)
    return adata, cluster_cell_ls, cell_cluster_arr, celltype_names


def load_data_rep(path, ct_name='clusters'):
    """Loader for subsampled replicate datasets used in the robustness analysis.

    The path layout is now configurable; callers must pass an explicit path
    (e.g. './data4_robust_test/rep1_subsample_half.h5ad').
    """
    return load_data1(path, ct_name=ct_name)


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


def get_cellcyle_markers():
    return ["CDKN1A", "CDKN2A", "TP53", "GADD45A", "IGFBP7", "SERPINE1", "GLB1",
            "IL6", "IL8", "MMP1", "MMP3"]


def load_markers(args):
    markers = pd.read_csv('senescence_marker_list.csv')
    markers_ls = []
    for col_name, data in markers.items():
        markers_ls.append(list(data[data.notnull()]))

    lr_panel_path = getattr(args, 'lr_panel', None)
    markers5 = list(get_ccc_markers(lr_panel_path)[1])
    markers_ls.append(markers5)
    markers6 = get_cellcyle_markers()
    markers_ls.append(markers6)

    print('Number of genes in each marker list:')
    print(tabulate([
        ['SenMayo', 'FRIDMAN', 'CellAge', 'Cell-cycle markers'],
        [len(markers_ls[0]), len(markers_ls[1]), len(markers_ls[2]), len(markers_ls[5])]
    ], headers='firstrow'))

    # Drop GO and L-R markers from the active set used for senescence-gene seeding
    markers_ls = [markers_ls[0], markers_ls[1], markers_ls[2], markers_ls[5]]

    sel = args.gene_set
    if sel == 'full':
        return markers_ls
    if sel == 'senmayo':
        return [markers_ls[0]]
    if sel == 'fridman':
        return [markers_ls[1]]
    if sel == 'cellage':
        return [markers_ls[2]]
    if sel == 'senmayo+cellage':
        return [markers_ls[0], markers_ls[2]]
    if sel == 'senmayo+fridman':
        return [markers_ls[0], markers_ls[1]]
    if sel == 'senmayo+fridman+cellage':
        return [markers_ls[0], markers_ls[1], markers_ls[2]]
    return markers_ls


def load_nonsenmarkers(adata):
    nonsen_markers = ["CCNB1", "CDK1", "CDC25C", "WEE1", "CHK1", "CCNA2", "PCNA",
                      "MCM", "RPA", "DHFR", "CCNB1", "AURKA", "AURKB", "PLK1",
                      "H3S10ph", "BUB1"]
    return [g for g in nonsen_markers if g in adata.var_names]


# ============================================================================
#                       GENE / GRAPH CONSTRUCTION
# ============================================================================
def get_highly_genes(adata, n_genes):
    new_data = adata.copy()
    sp.pp.normalize_total(new_data, target_sum=1e4)
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
        highly_genes = get_highly_genes(adata, int(args.n_genes))
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
    assert gene_names[100] == new_data.var.index[100], 'Gene-index mismatch'

    markers_index = []
    for markers in markers_ls:
        indexs = [gene_names.index(m) for m in markers if m in gene_names]
        markers_index.append(indexs)

    sen_gene_ls = [gene_names.index(i) for i in sen_gene_ls]
    nonsen_gene_ls = [gene_names.index(i) for i in filtered_highly_genes]
    return new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names


def process_data(adata, cluster_cell_ls, cell_cluster_arr, args):
    markers_ls = load_markers(args)
    return combine_genes(adata, markers_ls, args)


def build_graph_nx(adata, gene_cell, cell_cluster_arr, sen_gene_ls,
                   nonsen_gene_ls, gene_names, args):
    """Construct the heterogeneous cell-gene graph, optionally augmented with
    cell-cell edges from the L-R-derived CCC matrix.

    The reviewer correctly noted that the type2 (continuous) branch in v1
    built continuous edge weights but never piped them into the GAT. The
    weights are now consumed by GATv2Conv(edge_dim=1) in the encoder; see
    model_GAT.py.
    """
    lr_panel_path = getattr(args, 'lr_panel', None)
    g_index, c_index = np.nonzero(gene_cell)
    print('Cell-gene graph, the number of edges:', len(g_index))
    gene_num = gene_cell.shape[0]
    c_index = c_index + gene_num

    if args.ccc == 'type1':
        print('Adding cell-cell edges with binary weights ...')
        adj_matrix, _ = build_ccc_graph(gene_cell, gene_names, lr_panel_path)
        i1, i2 = np.nonzero(adj_matrix)
        print('CCC graph, the number of edges:', len(i1))
        i1, i2 = i1 + gene_num, i2 + gene_num
        edge_index = torch.tensor(np.array([np.concatenate([g_index, i1]),
                                            np.concatenate([c_index, i2])]),
                                  dtype=torch.long)
        ccc_matrix = None
    else:
        print('No cell-cell edges (type3) ...')
        edge_index = torch.tensor(np.array([g_index, c_index]), dtype=torch.long)
        ccc_matrix = None

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
    return graph_nx, edge_index, ccc_matrix


def add_nx_embedding(graph_nx, gene_embed, cell_embed):
    for i in range(gene_embed.shape[0]):
        graph_nx.nodes[i]['emb'] = gene_embed[i].detach().cpu()
    for i in range(cell_embed.shape[0]):
        graph_nx.nodes[i + gene_embed.shape[0]]['emb'] = cell_embed[i].detach().cpu()
    return graph_nx


def build_graph_pyg(gene_cell, gene_embed, cell_embed, edge_indexs, ccc_matrix=None):
    print('Building PyG graph')
    y = [True] * gene_cell.shape[0] + [False] * gene_cell.shape[1]
    y = torch.tensor(y)
    print('edge index: ', edge_indexs.shape)
    x = torch.cat([gene_embed, cell_embed]).detach()
    print('node feature: ', x.shape)

    if ccc_matrix is None:
        print('Building PyG graph without edge weights (binary edges).')
        edge_index = to_undirected(edge_indexs)
        graph_pyg = Graphdata(x=x, edge_index=edge_index, y=y)
    else:
        print('Building PyG graph with continuous edge weights (edge_dim=1).')
        flatten_edge_features = ccc_matrix[ccc_matrix != 0]
        min_val = np.min(flatten_edge_features)
        max_val = np.max(flatten_edge_features)
        if max_val > min_val:
            normalized = (flatten_edge_features - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(flatten_edge_features)
        # Cell-gene edges receive weight 1.0 by convention; CCC edges receive
        # the normalized L-R probability.
        edge_attr = np.concatenate([
            np.ones(edge_indexs.shape[1] - len(normalized)),
            normalized
        ])
        undirected_edge_index, undirected_edge_attr = to_undirected(
            edge_indexs,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            reduce='mean'
        )
        graph_pyg = Graphdata(x=x, edge_index=undirected_edge_index,
                              edge_attr=undirected_edge_attr, y=y)

    print('Pyg graph:', graph_pyg)
    print('graph.is_directed():', graph_pyg.is_directed())
    return graph_pyg


# ============================================================================
#                       L-R PROBABILITY (SoptSC-adapted)
# ============================================================================
def build_ccc_matrix(expression_matrix, gene_names, lr_panel_path=None):
    """Adapted from SoptSC's L-R probability term (Wang et al. 2019).

    DeepSAS uses ONLY the L-R-expression term, not the SoptSC target-gene
    weighting (see Methods 1.2 in the revised manuscript for rationale).
    """
    ligand_receptor_dict = get_ccc_markers(lr_panel_path)[0]
    ccc_matrix = None
    for ligand, receptors in ligand_receptor_dict.items():
        if ligand not in gene_names:
            continue
        ligand_exp = expression_matrix[:, gene_names.index(ligand)].reshape(-1, 1)
        receptor_indexs = [gene_names.index(r) for r in receptors if r in gene_names]
        if not receptor_indexs:
            continue
        receptor_exp = expression_matrix[:, receptor_indexs]
        receptor_exp = np.sum(receptor_exp, axis=1).reshape(1, -1)
        result = ligand_exp * receptor_exp
        ccc_matrix = result if ccc_matrix is None else ccc_matrix + result
    return ccc_matrix


def convert_to_adj(ccc_matrix):
    masked_result = ma.masked_where(ccc_matrix == 0, ccc_matrix)
    transformed = np.exp(-1 / masked_result).filled(0)
    symmetric = 0.5 * (transformed + transformed.T)
    mask = symmetric >= 0.8
    return np.where(mask, symmetric, 0)


@jit(nopython=True)
def compute_adj_matrix(ccc_matrix, w=1.0, b=0.0):
    n = ccc_matrix.shape[0]
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff_norm = np.linalg.norm(ccc_matrix[i] - ccc_matrix[j])
            adj_matrix[i, j] = 1 / (1 + np.exp(w * diff_norm ** 2 + b))
    return adj_matrix


def convert_to_adj_v2(ccc_matrix, t=0.8):
    adj_matrix = compute_adj_matrix(ccc_matrix)
    return np.where(adj_matrix >= t, adj_matrix, 0)


def build_ccc_graph(gene_cell, gene_names, lr_panel_path=None):
    """gene_cell is (gene x cell). Internally transposed to (cell x gene)."""
    ccc_matrix = build_ccc_matrix(gene_cell.T, gene_names, lr_panel_path)
    if ccc_matrix is None:
        # Edge case: none of the L-R genes found in the dataset
        n = gene_cell.shape[1]
        return np.zeros((n, n)), np.zeros((n, n))
    adj_matrix = convert_to_adj(ccc_matrix)
    ccc_matrix = ccc_matrix * adj_matrix
    return adj_matrix, ccc_matrix


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
