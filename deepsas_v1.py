import csv
import datetime
import logging
import os
import random

import numpy as np
import scanpy as sp
import torch
from torch.optim.lr_scheduler import ExponentialLR

import utils
from model_AE import reduction_AE
from model_GAT import GAEModel
from model_Sencell import Sencell, cell_optim


# ============================================================================
#                                  SETUP
# ============================================================================
args = utils.parse_args()
print(vars(args))
args.output_dir = f"./outputs/{args.exp_name}"
print("Outputs dir:", args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='# %Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()


# ============================================================================
#                       PART 1: LOAD & PROCESS DATA
# ============================================================================
logger.info("====== Part 1: load and process data ======")
if 'example' in args.exp_name and not os.path.isabs(args.input_data_count) and \
        args.input_data_count == 'example_data/example_data.h5ad':
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = \
        utils.load_example_data(path=args.input_data_count, ct_name=args.cell_type_col)
else:
    adata, cluster_cell_ls, cell_cluster_arr, celltype_names = \
        utils.load_data1(args.input_data_count, ct_name=args.cell_type_col)

new_data, markers_index, sen_gene_ls, nonsen_gene_ls, gene_names = \
    utils.process_data(adata, cluster_cell_ls, cell_cluster_arr, args)
new_data.write_h5ad(os.path.join(args.output_dir, f'{args.exp_name}_new_data.h5ad'))

gene_cell = new_data.X.toarray().T
args.gene_num = gene_cell.shape[0]
args.cell_num = gene_cell.shape[1]
print(f'cell num: {new_data.shape[0]}, gene num: {new_data.shape[1]}')

if args.retrain:
    graph_nx, edge_indexs, ccc_matrix = utils.build_graph_nx(
        new_data, gene_cell, cell_cluster_arr, sen_gene_ls,
        nonsen_gene_ls, gene_names, args)
logger.info("Part 1 done.")


# ============================================================================
#                  PART 2: INITIAL EMBEDDING (UMAP by default)
# ============================================================================
logger.info("====== Part 2: initial embedding ======")
device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
print('device:', device)
args.device = device


def run_scanpy(adata_in, batch_remove=False, batch_name='Sample'):
    """Default initial embedding: PCA -> kNN -> UMAP (Methods 1.2)."""
    a = adata_in.copy()
    sp.pp.normalize_total(a, target_sum=1e4)
    sp.pp.log1p(a)
    sp.pp.scale(a, max_value=10)
    if batch_remove:
        if batch_name not in a.obs.columns:
            raise KeyError(
                f"Batch column '{batch_name}' not found in adata.obs. "
                f"Pass --batch_col or omit --batch_remove."
            )
        print(f"Removing batch effect via ComBat on .obs['{batch_name}'] ...")
        sp.pp.combat(a, key=batch_name)
    else:
        print('Skipping batch correction.')
    sp.tl.pca(a, svd_solver='arpack')
    sp.pp.neighbors(a, n_neighbors=10, n_pcs=40)
    sp.tl.umap(a, n_components=args.emb_size)
    return a.obsm['X_umap']


if args.use_autoencoder:
    # Optional dimensionality-reduction autoencoder
    if args.retrain:
        gene_embed, cell_embed = reduction_AE(gene_cell, device)
        print(gene_embed.shape, cell_embed.shape)
        torch.save(gene_embed, os.path.join(args.output_dir, f'{args.exp_name}_gene.emb'))
        torch.save(cell_embed, os.path.join(args.output_dir, f'{args.exp_name}_cell.emb'))
    else:
        print('Skipping AE training; loading saved embeddings.')
        gene_embed = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_gene.emb'))
        cell_embed = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_cell.emb'))

    if args.retrain:
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed,
                                          edge_indexs, ccc_matrix)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
    else:
        graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
else:
    # Default: UMAP-based initial embeddings (matches Methods 1.2)
    if args.retrain:
        cell_embed = run_scanpy(new_data.copy(),
                                batch_remove=args.batch_remove,
                                batch_name=args.batch_col)
        print('Cell embedding generated.')
        gene_embed = run_scanpy(new_data.copy().T,
                                batch_remove=False,
                                batch_name=args.batch_col)
        print('Gene embedding generated.')
        cell_embed = torch.tensor(cell_embed)
        gene_embed = torch.tensor(gene_embed)
        graph_nx = utils.add_nx_embedding(graph_nx, gene_embed, cell_embed)
        graph_pyg = utils.build_graph_pyg(gene_cell, gene_embed, cell_embed,
                                          edge_indexs, ccc_matrix)
        torch.save(graph_nx, os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        torch.save(graph_pyg, os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
        print('graph_nx and graph_pyg saved.')
    else:
        print('Loading saved graph_nx and graph_pyg ...')
        graph_nx = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphnx.data'))
        graph_pyg = torch.load(os.path.join(args.output_dir, f'{args.exp_name}_graphpyg.data'))
logger.info("Part 2 done.")


# ============================================================================
#                       PART 3: GAT TRAINING
# ============================================================================
logger.info("====== Part 3: GAT training ======")
data = graph_pyg.to(device)
torch.cuda.empty_cache()

edge_dim = None

if args.retrain:
    model = GAEModel(args.emb_size, args.emb_size, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train():
        model.train()
        optimizer.zero_grad()
        if model.use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
        else:
            z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(args.gat_epoch):
        loss = train()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    GAT_path = os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
    torch.save(model, GAT_path)
    print(f'GAT model saved! {GAT_path}')
else:
    GAT_path = os.path.join(args.output_dir, f'{args.exp_name}_GAT.pt')
    print(f'Loading GAT from {GAT_path}')
    model = torch.load(GAT_path).to(device)
torch.cuda.empty_cache()
logger.info("Part 3 done.")


# ============================================================================
#                  PART 4: CONTRASTIVE LEARNING (convergence-controlled)
# ============================================================================
logger.info("====== Part 4: Contrastive learning ======")


def check_celltypes(predicted_cell_indexs, graph_nx, celltype_names):
    cell_types = []
    for i in predicted_cell_indexs:
        cluster = graph_nx.nodes[int(i)]['cluster']
        cell_types.append(celltype_names[cluster])
    from collections import Counter
    print("SnC in different cell types: ", Counter(cell_types))


def build_cell_dict(gene_cell, predicted_cell_indexs, GAT_embeddings, graph_nx):
    sencell_dict = {}
    nonsencell_dict = {}
    for i in range(gene_cell.shape[0], gene_cell.shape[0] + gene_cell.shape[1]):
        target = sencell_dict if i in predicted_cell_indexs else nonsencell_dict
        target[i] = [GAT_embeddings[i], graph_nx.nodes[int(i)]['cluster'], 0, i]
    return sencell_dict, nonsencell_dict


def identify_sengene_v1(sencell_dict, gene_cell, edge_index_selfloop,
                        attention_scores, sen_gene_ls):
    if len(sencell_dict) == 0:
        return torch.as_tensor(sen_gene_ls, dtype=torch.long)

    cell_index = torch.tensor(list(sencell_dict.keys()), dtype=torch.long)
    cell_mask = torch.zeros(gene_cell.shape[0] + gene_cell.shape[1], dtype=torch.bool)
    cell_mask[cell_index] = True
    res = []
    score_sengene_ls = []
    for gene_index in range(gene_cell.shape[0]):
        connected_cells = edge_index_selfloop[0][edge_index_selfloop[1] == gene_index]
        masked = connected_cells[cell_mask[connected_cells]]
        if masked.numel() == 0:
            res.append(0)
        else:
            tmp = attention_scores[edge_index_selfloop[1] == gene_index]
            attention_edge = torch.sum(tmp[cell_mask[connected_cells]], dim=1)
            res.append(torch.mean(attention_edge).item())
        if gene_index in sen_gene_ls:
            score_sengene_ls.append(res[-1])
    num = 10
    res1 = torch.tensor(res)
    new_genes = torch.argsort(res1)[-num:]
    score_sengene_ls = torch.tensor(score_sengene_ls)
    if isinstance(sen_gene_ls, torch.Tensor):
        new_sen_gene_ls = sen_gene_ls[torch.argsort(score_sengene_ls)[num:].tolist()]
    else:
        new_sen_gene_ls = torch.tensor(sen_gene_ls)[torch.argsort(score_sengene_ls)[num:].tolist()]
    return torch.cat((new_sen_gene_ls, new_genes))


def generate_ct_specific_scores(sen_gene_ls, gene_cell, edge_index_selfloop,
                                attention_scores, graph_nx, celltype_names):
    gene_index = torch.as_tensor(sen_gene_ls, dtype=torch.long)
    gene_mask = torch.zeros(gene_cell.shape[0] + gene_cell.shape[1], dtype=torch.bool)
    gene_mask[gene_index] = True
    ct_specific_scores = {}
    for cell_index in range(gene_cell.shape[0], gene_cell.shape[0] + gene_cell.shape[1]):
        connected_genes = edge_index_selfloop[0][edge_index_selfloop[1] == cell_index]
        if len(connected_genes[gene_mask[connected_genes]]) == 0:
            continue
        attention_edge = torch.sum(
            attention_scores[edge_index_selfloop[1] == cell_index][gene_mask[connected_genes]],
            axis=1)
        attention_s = torch.mean(attention_edge)
        cluster = graph_nx.nodes[int(cell_index)]['cluster']
        ct_specific_scores.setdefault(cluster, []).append([float(attention_s), int(cell_index)])
    return ct_specific_scores


def calculate_outliers_v1(scores_index):
    arr = np.array(scores_index)
    scores, indexs = arr[:, 0], arr[:, 1]
    Q1, Q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    upper = Q3 + 1.5 * (Q3 - Q1)
    snc_index, outliers = [], []
    for i, s in enumerate(scores):
        if s > upper:
            snc_index.append(indexs[i])
            outliers.append([s, indexs[i]])
    return len(snc_index), snc_index, outliers


def extract_cell_indexs(ct_specific_scores):
    snc_indexs = []
    for key, values in ct_specific_scores.items():
        arr = np.array(values)
        counts, snc_index, _ = calculate_outliers_v1(arr)
        if counts >= 10:
            snc_indexs.extend(snc_index)
    return snc_indexs


# --- model + optimizer setup ---
cellmodel = Sencell(args.emb_size).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(cellmodel.parameters(),
                             lr=args.learning_rate, weight_decay=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.85)
sencell_dict = None

edge_index_selfloop_cell, attention_scores_cell = model.get_attention_scores(data)
attention_scores_cell = attention_scores_cell.cpu().detach()
edge_index_selfloop_cell = edge_index_selfloop_cell.cpu().detach()
model.eval()

# --- convergence-controlled loop ---
convergence_log_path = os.path.join(
    args.output_dir, f'{args.exp_name}_convergence.csv')
with open(convergence_log_path, 'w', newline='') as f:
    csv.writer(f).writerow(['iter', 'snc_count', 'sng_count',
                            'jaccard_snc', 'jaccard_sng', 'converged'])

epoch = 0
converged = False
old_sencell_dict = None
old_sengene_ls = None

while not converged and epoch < args.max_iter:
    print(f'{datetime.datetime.now()}: Contrastive learning iter {epoch:03d}')

    # ---- cell selection from current attention scores ----
    ct_specific_scores = generate_ct_specific_scores(
        sen_gene_ls, gene_cell, edge_index_selfloop_cell,
        attention_scores_cell, graph_nx, celltype_names)
    predicted_cell_indexs = extract_cell_indexs(ct_specific_scores)
    check_celltypes(predicted_cell_indexs, graph_nx, celltype_names)

    GAT_embeddings = model(data.x, data.edge_index).detach()
    sencell_dict, nonsencell_dict = build_cell_dict(
        gene_cell, predicted_cell_indexs, GAT_embeddings, graph_nx)

    jacc_cell = (utils.get_sencell_jaccard(old_sencell_dict, sencell_dict)
                 if old_sencell_dict is not None else float('nan'))

    # ---- contrastive optimization of cell embeddings ----
    cellmodel, sencell_dict, nonsencell_dict = cell_optim(
        cellmodel, optimizer, sencell_dict, nonsencell_dict, None,
        args, train=True)
    scheduler.step()
    print('current lr:', optimizer.param_groups[0]['lr'])

    new_GAT_embeddings = GAT_embeddings
    for key in sencell_dict:
        new_GAT_embeddings[key] = sencell_dict[key][2].detach()
    for key in nonsencell_dict:
        new_GAT_embeddings[key] = nonsencell_dict[key][2].detach()
    data.x = new_GAT_embeddings
    edge_index_selfloop, attention_scores = model.get_attention_scores(data)
    attention_scores = attention_scores.cpu()
    edge_index_selfloop = edge_index_selfloop.cpu()

    # ---- gene selection from updated attention ----
    new_sen_gene_ls = identify_sengene_v1(
        sencell_dict, gene_cell, edge_index_selfloop,
        attention_scores, sen_gene_ls)
    jacc_gene = (utils.get_sengene_jaccard(old_sengene_ls, new_sen_gene_ls)
                 if old_sengene_ls is not None else float('nan'))

    # ---- save iteration outputs (preserves v1's per-iter file naming) ----
    torch.save([sencell_dict, new_sen_gene_ls, attention_scores, edge_index_selfloop],
               os.path.join(args.output_dir,
                            f'{args.exp_name}_sencellgene-epoch{epoch}.data'))

    # ---- convergence check ----
    if epoch > 0 and jacc_cell >= args.convergence_tol \
            and jacc_gene >= args.convergence_tol:
        converged = True
        print(f'Converged at iter {epoch}: '
              f'Jaccard(SnC)={jacc_cell:.4f}, Jaccard(SnG)={jacc_gene:.4f}')

    # ---- log ----
    with open(convergence_log_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch, len(sencell_dict),
            int(len(new_sen_gene_ls)),
            f'{jacc_cell:.6f}' if not np.isnan(jacc_cell) else '',
            f'{jacc_gene:.6f}' if not np.isnan(jacc_gene) else '',
            converged
        ])

    # ---- roll forward ----
    old_sencell_dict = sencell_dict
    old_sengene_ls = (new_sen_gene_ls.tolist()
                      if isinstance(new_sen_gene_ls, torch.Tensor)
                      else list(new_sen_gene_ls))
    sen_gene_ls = new_sen_gene_ls
    edge_index_selfloop_cell = edge_index_selfloop
    attention_scores_cell = attention_scores
    epoch += 1

if not converged:
    print(f'Reached --max_iter ({args.max_iter}) without meeting convergence '
          f'tolerance {args.convergence_tol}. Last iteration outputs retained.')
logger.info("Part 4 done.")
