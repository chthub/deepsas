import os
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as F


_OPTIMIZED_EMBEDDING = 2


def get_cluster_cell_dict(sencell_dict, nonsencell_dict):
    # Calculate the mapping table of cells contained in the cluster
    # {cluster: [cell_indexs]}
    cluster_sencell = defaultdict(list)
    cluster_nonsencell = defaultdict(list)

    for key, value in sencell_dict.items():
        cluster_sencell[value[1]].append(key)

    for key, value in nonsencell_dict.items():
        cluster_nonsencell[value[1]].append(key)

    return cluster_sencell, cluster_nonsencell


def getPrototypeEmb(sencell_dict, cluster_sencell):
    # Calculate the prototype embedding, only the prototype of sen cells is calculated
    prototype_emb = {}
    for key, value in cluster_sencell.items():
        embs = []
        for i in value:
            embs.append(sencell_dict[i][_OPTIMIZED_EMBEDDING].view(1, -1))
        embs = torch.cat(embs)
        prototype_emb[key] = torch.mean(embs, 0)

    return prototype_emb


class Sencell(torch.nn.Module):
    """Fixed residual MLP used to refine cell embeddings.

    The reported architecture uses six linear transformations, CELU
    activations, two residual additions, and shared LayerNorm. ``hidden_size``
    and the three initial distance targets remain the tunable dimensions.
    """

    def __init__(self, dim, hidden_size, distance_levels):
        super().__init__()
        if dim < 1 or hidden_size < 1:
            raise ValueError('Embedding and hidden dimensions must be positive')
        levels = torch.as_tensor(distance_levels, dtype=torch.float32)
        if levels.numel() != 3 or not torch.isfinite(levels).all():
            raise ValueError('Exactly three finite distance levels are required')
        self.hidden_size = hidden_size
        self.linear1 = Linear(dim, self.hidden_size)
        self.linear2 = Linear(self.hidden_size, self.hidden_size)
        self.linear21 = Linear(self.hidden_size, self.hidden_size)
        self.linear22 = Linear(self.hidden_size, self.hidden_size)
        self.linear3 = Linear(self.hidden_size, self.hidden_size)
        self.linear4 = Linear(self.hidden_size, dim)
        self.act = torch.nn.CELU()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.levels = torch.nn.Parameter(levels.clone())
        self.device = None

    def catEmbeddings(self, sencell_dict, nonsencell_dict):
        embeddings = []
        for key, value in sencell_dict.items():
            embeddings.append(value[0].view(1, -1))
        for key, value in nonsencell_dict.items():
            embeddings.append(value[0].view(1, -1))
        return torch.cat(embeddings)

    def updateDict(self, x, sencell_dict, nonsencell_dict):
        count = 0
        for key, value in sencell_dict.items():
            sencell_dict[key][2] = x[count]
            count += 1
        for key, value in nonsencell_dict.items():
            nonsencell_dict[key][2] = x[count]
            count += 1

        return sencell_dict, nonsencell_dict

    def forward(self, sencell_dict, nonsencell_dict, device):
        x = self.catEmbeddings(sencell_dict, nonsencell_dict).to(device)
        self.device = device

        x = self.act(self.linear1(x))
        x = x + self.act(self.linear2(x))
        x = self.layer_norm(x)
        x = x + self.act(self.linear22(self.act(self.linear21(x))))
        x = self.layer_norm(x)
        x = self.linear4(self.act(self.linear3(x)))

        result = self.updateDict(x, sencell_dict, nonsencell_dict)
        return result

    def eucliDistance(self, v1, v2):
        # Calculate Euclidean distance
        return F.pairwise_distance(v1.view(1, -1), v2.view(1, -1))

    def get_d1(self, sencell_dict, cluster_sencell, prototype_emb):
        d1 = []
        for cluster, prototype in prototype_emb.items():
            # For the case where there is only one sen cell, the distance is 0, but it is displayed as 1.1314e-05
            distance_ls = []
            for cell_index in cluster_sencell[cluster]:
                cell_emb = sencell_dict[cell_index][_OPTIMIZED_EMBEDDING]
                distance = self.eucliDistance(cell_emb, prototype)
                distance_ls.append(distance)
            d1.append(distance_ls)

        return d1

    def get_d2(self, sencell_dict, cluster_sencell, prototype_emb):
        # d2 represents the distance between sen cells in different clusters
        d2 = []
        for cluster, prototype in prototype_emb.items():
            distance_ls = []
            for another_cluster, cell_indexs in cluster_sencell.items():
                if another_cluster != cluster:
                    for cell_index in cell_indexs:
                        cell_emb = sencell_dict[cell_index][_OPTIMIZED_EMBEDDING]
                        distance = self.eucliDistance(cell_emb, prototype)
                        distance_ls.append(distance)
            d2.append(distance_ls)
        return d2

    def get_d3(self, nonsencell_dict, cluster_nonsencell, prototype_emb):
        # d3 represents the distance between senescent cells and non-senescent cells within the same cell type
        d3 = []
        for cluster, prototype in prototype_emb.items():
            distance_ls = []
            if cluster in cluster_nonsencell:
                for nonsencell_index in cluster_nonsencell[cluster]:
                    nonsencell_emb = nonsencell_dict[nonsencell_index][
                        _OPTIMIZED_EMBEDDING]
                    distance = self.eucliDistance(nonsencell_emb, prototype)
                    distance_ls.append(distance)
            d3.append(distance_ls)
        return d3

    def caculateDistance(self, sencell_dict, nonsencell_dict,
                         cluster_sencell, cluster_nonsencell,
                         prototype_emb):
        # d1: The distance between the prototype and each cell in the sen cell cluster
        # For each cluster with sen cells, there will be d1
        d1 = self.get_d1(sencell_dict, cluster_sencell, prototype_emb)
        # d2 represents the distance between sen cells in different clusters
        d2 = self.get_d2(sencell_dict, cluster_sencell, prototype_emb)
        # d3 represents the distance between sen cells and non-sen cells within the same cell type
        d3 = self.get_d3(nonsencell_dict, cluster_nonsencell, prototype_emb)
        return d1, d2, d3

    def getMultiLevelDistanceLoss(self, distances):
        # Custom ProtoNCE + MDR
        d1, d2, d3 = distances
        result = torch.tensor(0., device=self.device, requires_grad=True)

        def distanceDiff(cluster_d, level):
            # cluster_d is a list of distances of the same type in the same cluster
            count = 0
            result = torch.tensor(0., device=self.device, requires_grad=True)
            for d in cluster_d:
                result = result + (d - level).abs().sum()
                count += 1
            if count==0:
                return torch.tensor(0., device=self.device, requires_grad=True)
            return result/count

        for cluster_d_1, cluster_d_2, cluster_d_3 in zip(d1, d2, d3):
            result = result + distanceDiff(cluster_d_1, self.levels[0])
            result = result + distanceDiff(cluster_d_2, self.levels[1])
            result = result + distanceDiff(cluster_d_3, self.levels[2])
        return result

    def loss(self, sencell_dict, nonsencell_dict):
        # step 1: Calculate the mapping table of cluster and cell
        cluster_sencell, cluster_nonsencell = get_cluster_cell_dict(
            sencell_dict, nonsencell_dict)
        # step 2: Calculate prototype embedding of sen cell clusters
        prototype_emb = getPrototypeEmb(sencell_dict, cluster_sencell)
        # step 3: Calculate distance
        distances = self.caculateDistance(sencell_dict, nonsencell_dict,
                                          cluster_sencell, cluster_nonsencell,
                                          prototype_emb)
        # step 4: Calculate multi-level distance
        loss = self.getMultiLevelDistanceLoss(distances)

        return loss


def cell_optim(cellmodel, optimizer, sencell_dict, nonsencell_dict, args):
    if not sencell_dict:
        raise ValueError('Cell optimization requires nonempty SnC candidates')
    cellmodel.train()

    started = time.monotonic()
    for epoch in range(args.cell_optim_epoch):
        optimizer.zero_grad()
        sencell_dict, nonsencell_dict = cellmodel(
            sencell_dict, nonsencell_dict, args.device)
        loss = cellmodel.loss(sencell_dict, nonsencell_dict)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite cell optimization loss')
        cellmodel.last_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        elapsed = time.monotonic() - started
        eta = elapsed / (epoch + 1) * (args.cell_optim_epoch - epoch - 1)
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}: '
              f'Cell epoch {epoch + 1}/{args.cell_optim_epoch}, '
              f'loss={cellmodel.last_loss:.6f}, elapsed={elapsed:.1f}s, '
              f'ETA={eta:.1f}s')

    # Return embeddings evaluated with the final optimizer update applied.
    cellmodel.eval()
    with torch.no_grad():
        sencell_dict, nonsencell_dict = cellmodel(
            sencell_dict, nonsencell_dict, args.device)

    torch.save(cellmodel, os.path.join(
        args.output_dir, f'{args.exp_name}_cellmodel.pt'))

    return cellmodel, sencell_dict, nonsencell_dict
