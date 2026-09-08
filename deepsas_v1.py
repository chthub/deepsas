import csv
import datetime
import json
import logging
import os
import random
import time

import numpy as np
import scanpy as sp
import torch
from torch.optim.lr_scheduler import ExponentialLR

import utils
from model_AE import reduction_AE
from model_GAT import GAEModel
from model_Sencell import Sencell, cell_optim
from refinement import (candidate_convergence, mean_attention_by_target,
                        update_gene_candidates)


# ============================================================================
#                                  SETUP
# ============================================================================
args = utils.parse_args()
print(vars(args))
run_config = vars(args).copy()
args.output_dir = os.path.join(args.output_dir, args.exp_name)
print("Outputs dir:", args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
summary_path = os.path.join(args.output_dir, f'{args.exp_name}_run_summary.json')
run_summary = {'status': 'running', 'config': run_config}
with open(summary_path, 'w') as f:
    json.dump(run_summary, f, indent=2)

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
run_summary['initial_sng_indices'] = list(sen_gene_ls)

gene_cell = (new_data.X.toarray() if hasattr(new_data.X, 'toarray') else new_data.X).T
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
run_config['device'] = str(device)


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
    sp.tl.pca(a, svd_solver='arpack', random_state=args.seed)
    sp.pp.neighbors(a, n_neighbors=10, n_pcs=40)
    sp.tl.umap(a, n_components=args.emb_size, random_state=args.seed)
    return a.obsm['X_umap']


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

# Methods 1.2 uses binary connectivity. There is no independent edge feature
# vector; phi affects which CCC edges exist, rather than their GAT edge_dim.
edge_dim = None

if args.retrain:
    model = GAEModel(args.emb_size, args.emb_size, edge_dim=edge_dim,
                     hidden_size=args.gat_hidden_size,
                     dropout=args.gat_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gat_learning_rate)

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

    gat_started = time.monotonic()
    for epoch in range(args.gat_epoch):
        loss = train()
        elapsed = time.monotonic() - gat_started
        eta = elapsed / (epoch + 1) * (args.gat_epoch - epoch - 1)
        print(f'{datetime.datetime.now()}: GAT epoch {epoch + 1}/{args.gat_epoch}, '
              f'loss={loss:.6f}, elapsed={elapsed:.1f}s, ETA={eta:.1f}s')
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
    counts = Counter(cell_types)
    print("SnC in different cell types: ", counts)
    return dict(counts)


def build_cell_dict(gene_cell, predicted_cell_indexs, GAT_embeddings, graph_nx):
    predicted_cell_indexs = set(predicted_cell_indexs)
    sencell_dict = {}
    nonsencell_dict = {}
    for i in range(gene_cell.shape[0], gene_cell.shape[0] + gene_cell.shape[1]):
        target = sencell_dict if i in predicted_cell_indexs else nonsencell_dict
        target[i] = [GAT_embeddings[i], graph_nx.nodes[int(i)]['cluster'], 0, i]
    return sencell_dict, nonsencell_dict


def identify_sengene_v1(sencell_dict, gene_cell, edge_index_selfloop,
                        attention_scores, sen_gene_ls, iqr_multiplier=1.5):
    scores, _ = mean_attention_by_target(
        edge_index_selfloop, attention_scores, sencell_dict.keys(),
        gene_cell.shape[0] + gene_cell.shape[1])
    updated, diagnostics = update_gene_candidates(
        scores[:gene_cell.shape[0]].numpy(), sen_gene_ls, iqr_multiplier)
    return torch.as_tensor(updated, dtype=torch.long), diagnostics


def generate_ct_specific_scores(sen_gene_ls, gene_cell, edge_index_selfloop,
                                attention_scores, graph_nx, celltype_names):
    scores, connected = mean_attention_by_target(
        edge_index_selfloop, attention_scores, sen_gene_ls,
        gene_cell.shape[0] + gene_cell.shape[1])
    ct_specific_scores = {}
    for cell_index in range(gene_cell.shape[0], gene_cell.shape[0] + gene_cell.shape[1]):
        if not connected[cell_index]:
            continue
        cluster = graph_nx.nodes[int(cell_index)]['cluster']
        ct_specific_scores.setdefault(cluster, []).append([float(scores[cell_index]), int(cell_index)])
    return ct_specific_scores


def calculate_outliers_v1(scores_index, iqr_multiplier=1.5):
    if len(scores_index) == 0:
        return 0, [], []
    arr = np.array(scores_index)
    scores, indexs = arr[:, 0], arr[:, 1]
    if not np.isfinite(scores).all():
        raise FloatingPointError('Non-finite SnC attention score')
    Q1, Q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    upper = Q3 + iqr_multiplier * (Q3 - Q1)
    snc_index, outliers = [], []
    for i, s in enumerate(scores):
        if s > upper:
            snc_index.append(int(indexs[i]))
            outliers.append([s, int(indexs[i])])
    return len(snc_index), snc_index, outliers


def extract_cell_indexs(ct_specific_scores, iqr_multiplier=1.5, min_snc_per_type=10):
    snc_indexs = []
    for key, values in ct_specific_scores.items():
        arr = np.array(values)
        counts, snc_index, _ = calculate_outliers_v1(arr, iqr_multiplier)
        print(f'Cell type {key}: {counts} above-fence cells, '
              f'{counts if counts >= min_snc_per_type else 0} retained '
              f'(minimum {min_snc_per_type})')
        if counts >= min_snc_per_type:
            snc_indexs.extend(snc_index)
    return snc_indexs


# --- model + optimizer setup ---
cellmodel = Sencell(args.emb_size).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(cellmodel.parameters(),
                             lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
sencell_dict = None

# Candidate scoring must be deterministic: disable training dropout before the
# very first attention extraction as well as subsequent ones.
model.eval()
initial_features = data.x.detach().clone()
with torch.no_grad():
    edge_index_selfloop_cell, attention_scores_cell = model.get_attention_scores(data)
    # Retain the pretrained embeddings (Methods 1.4). Re-encoding data.x after
    # replacing it with optimized embeddings would feed the frozen encoder's
    # output back into itself and move gene features at every iteration.
    GAT_embeddings = model.encode(data.x, data.edge_index).detach()
attention_scores_cell = attention_scores_cell.cpu().detach()
edge_index_selfloop_cell = edge_index_selfloop_cell.cpu().detach()

# --- convergence-controlled loop ---
convergence_log_path = os.path.join(
    args.output_dir, f'{args.exp_name}_convergence.csv')
log_fields = ['iter', 'snc_count', 'sng_count', 'jaccard_snc', 'jaccard_sng',
              'converged', 'status', 'cell_loss', 'gene_threshold',
              'sng_outlier_count', 'sng_swaps', 'sng_added', 'sng_removed',
              'snc_counts_by_type', 'elapsed_seconds']
with open(convergence_log_path, 'w', newline='') as f:
    csv.DictWriter(f, fieldnames=log_fields).writeheader()

previous_cells = previous_genes = None
refinement_started = time.monotonic()
for epoch in range(args.max_iter):
    print(f'{datetime.datetime.now()}: Contrastive learning iter {epoch:03d}')

    # ---- cell selection from current attention scores ----
    ct_specific_scores = generate_ct_specific_scores(
        sen_gene_ls, gene_cell, edge_index_selfloop_cell,
        attention_scores_cell, graph_nx, celltype_names)
    predicted_cell_indexs = extract_cell_indexs(
        ct_specific_scores, args.iqr_multiplier, args.min_snc_per_type)
    check_celltypes(predicted_cell_indexs, graph_nx, celltype_names)

    sencell_dict, nonsencell_dict = build_cell_dict(
        gene_cell, predicted_cell_indexs, GAT_embeddings, graph_nx)

    cell_loss = None
    gene_diagnostics = dict(gene_threshold=None, sng_outlier_count=0,
                            sng_swaps=0, sng_added=[], sng_removed=[])
    if sencell_dict and len(sen_gene_ls):
        # ---- contrastive optimization of cell embeddings ----
        cellmodel, sencell_dict, nonsencell_dict = cell_optim(
            cellmodel, optimizer, sencell_dict, nonsencell_dict, None,
            args, train=True)
        cell_loss = cellmodel.last_loss
        scheduler.step()
        print('current lr:', optimizer.param_groups[0]['lr'])

        # The frozen GAT was trained on these input gene features. Only cell
        # rows are optimized; replacing gene rows with encoder outputs changes
        # the representation expected by the attention layer.
        updated_features = initial_features.clone()
        for cell_dict in (sencell_dict, nonsencell_dict):
            for key, value in cell_dict.items():
                updated_features[key] = value[2].detach()
        data.x = updated_features
        with torch.no_grad():
            edge_index_selfloop, attention_scores = model.get_attention_scores(data)
        attention_scores = attention_scores.detach().cpu()
        edge_index_selfloop = edge_index_selfloop.detach().cpu()

        # ---- threshold-driven, strictly improving gene replacements ----
        new_sen_gene_ls, gene_diagnostics = identify_sengene_v1(
            sencell_dict, gene_cell, edge_index_selfloop,
            attention_scores, sen_gene_ls, args.iqr_multiplier)

        # Save and compare complete iteration states: cell labels are selected
        # using the same updated attention and genes that are exported below.
        final_scores = generate_ct_specific_scores(
            new_sen_gene_ls, gene_cell, edge_index_selfloop,
            attention_scores, graph_nx, celltype_names)
        final_cells = extract_cell_indexs(
            final_scores, args.iqr_multiplier, args.min_snc_per_type)
        all_cells = nonsencell_dict.copy()
        all_cells.update(sencell_dict)
        sencell_dict = {i: all_cells[i] for i in final_cells}
    else:
        # An empty selection is a diagnostic endpoint, not an optimization
        # target or evidence of convergence. Preserve scores for cell tables.
        new_sen_gene_ls = torch.as_tensor(sen_gene_ls, dtype=torch.long)
        edge_index_selfloop = edge_index_selfloop_cell
        attention_scores = attention_scores_cell

    status, jacc_cell, jacc_gene = candidate_convergence(
        previous_cells, sencell_dict.keys(), previous_genes,
        new_sen_gene_ls.tolist(), args.convergence_tol)
    if status == 'running' and epoch + 1 == args.max_iter:
        status = 'max_iter'
    converged = status == 'converged'
    counts_by_type = check_celltypes(sencell_dict.keys(), graph_nx, celltype_names)

    # ---- save iteration outputs (preserves v1's per-iter file naming) ----
    iteration_result = [sencell_dict, new_sen_gene_ls, attention_scores, edge_index_selfloop]
    torch.save(iteration_result,
               os.path.join(args.output_dir,
                            f'{args.exp_name}_sencellgene-epoch{epoch}.data'))

    # ---- log ----
    elapsed = time.monotonic() - refinement_started
    row = dict(iter=epoch, snc_count=len(sencell_dict), sng_count=len(new_sen_gene_ls),
               jaccard_snc=jacc_cell if np.isfinite(jacc_cell) else '',
               jaccard_sng=jacc_gene if np.isfinite(jacc_gene) else '',
               converged=converged, status=status, cell_loss=cell_loss,
               snc_counts_by_type=json.dumps(counts_by_type), elapsed_seconds=elapsed,
               **gene_diagnostics)
    row['sng_added'] = json.dumps(row['sng_added'])
    row['sng_removed'] = json.dumps(row['sng_removed'])
    with open(convergence_log_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=log_fields).writerow(row)
    eta_cap = elapsed / (epoch + 1) * (args.max_iter - epoch - 1)
    print(f'{datetime.datetime.now()}: Iter {epoch}: status={status}, '
          f'SnCs={len(sencell_dict)}, SnGs={len(new_sen_gene_ls)}, '
          f'swaps={gene_diagnostics["sng_swaps"]}, '
          f'Jaccard(SnC)={jacc_cell:.6f}, Jaccard(SnG)={jacc_gene:.6f}, '
          f'elapsed={elapsed:.1f}s, ETA to iteration cap={eta_cap:.1f}s')
    if status != 'running':
        break

    # ---- roll forward ----
    previous_cells = list(sencell_dict)
    previous_genes = new_sen_gene_ls.tolist()
    sen_gene_ls = new_sen_gene_ls
    edge_index_selfloop_cell = edge_index_selfloop
    attention_scores_cell = attention_scores

final_path = os.path.join(args.output_dir, f'{args.exp_name}_sencellgene-final.data')
torch.save(iteration_result, final_path)
run_summary.update(status=status, converged=converged, final_epoch=epoch,
                   snc_count=len(sencell_dict), sng_count=len(new_sen_gene_ls),
                   snc_counts_by_type=counts_by_type, final_result=os.path.basename(final_path),
                   convergence_log=os.path.basename(convergence_log_path))
with open(summary_path, 'w') as f:
    json.dump(run_summary, f, indent=2)
if status == 'empty_candidates':
    logger.warning('Stopped with empty candidates; this is not convergence. '
                   'See per-type selection counts and the convergence log.')
elif status == 'max_iter':
    logger.warning('Reached max_iter=%d without convergence; final outputs are retained.',
                   args.max_iter)
print(f'Final status: {status}; final iteration: {epoch}; result: {final_path}')
logger.info("Part 4 done.")
