import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

from torch_geometric.nn import GATConv, GATv2Conv, GAE
from torch_geometric.nn.models import InnerProductDecoder


class GATEncoder(torch.nn.Module):
    """Two-layer single-head GAT encoder.

    Parameters
    ----------
    in_channels, out_channels : int
        Embedding sizes.
    edge_dim : int or None
        If None: original GATConv (edge weights ignored).
        If an int: GATv2Conv layers that consume `edge_attr` of that dimension.
    """

    def __init__(self, in_channels, out_channels, edge_dim=None,
                 hidden_size=32, dropout=0.6):
        super().__init__()
        self.use_edge_attr = edge_dim is not None
        if self.use_edge_attr:
            self.conv1 = GATv2Conv(in_channels, hidden_size, heads=1, dropout=dropout,
                                   edge_dim=edge_dim)
            self.conv2 = GATv2Conv(hidden_size, out_channels, heads=1, concat=True,
                                   dropout=dropout, edge_dim=edge_dim)
        else:
            self.conv1 = GATConv(in_channels, hidden_size, heads=1, dropout=dropout)
            self.conv2 = GATConv(hidden_size, out_channels, heads=1, concat=True,
                                 dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
        return x


class GAEModel(GAE):
    """Graph autoencoder = GATEncoder + InnerProductDecoder.

    This is the "graph autoencoder" referred to in Methods Section 1.3 of the
    manuscript. It is distinct from the optional dimensionality-reduction
    autoencoder in model_AE.py.
    """

    def __init__(self, in_channels, out_channels, edge_dim=None,
                 hidden_size=32, dropout=0.6):
        encoder = GATEncoder(in_channels, out_channels, edge_dim=edge_dim,
                             hidden_size=hidden_size, dropout=dropout)
        super().__init__(encoder)
        self.use_edge_attr = edge_dim is not None

    def encode(self, x, edge_index, edge_attr=None):
        return self.encoder(x, edge_index, edge_attr=edge_attr)

    def get_attention_scores(self, data):
        x, edge_index = data.x, data.edge_index
        if self.use_edge_attr:
            edge_attr = data.edge_attr
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = edge_attr.float()
            _, (edge_index_selfloop, alpha) = self.encoder.conv1(
                x, edge_index, edge_attr=edge_attr,
                return_attention_weights=True)
        else:
            _, (edge_index_selfloop, alpha) = self.encoder.conv1(
                x, edge_index, return_attention_weights=True)
        return edge_index_selfloop, alpha


# ----------------------------------------------------------------------------
# Auxiliary encoder used in earlier prototyping; preserved for compatibility.
# Not used in the production deepsas_v1.py pipeline.
# ----------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear1 = Linear(dim, dim)
        self.linear2 = Linear(dim, dim)
        self.conv1 = GATConv(dim, dim, add_self_loops=False)
        self.conv2 = GATConv(dim, dim, add_self_loops=False)
        self.act = torch.nn.CELU()

    def _cat(self, x_gene, x_cell, y):
        result = []
        cg = cc = 0
        for i in y:
            if i:
                result.append(x_gene[cg].view(1, -1)); cg += 1
            else:
                result.append(x_cell[cc].view(1, -1)); cc += 1
        return torch.cat(result)

    def forward(self, graph):
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        x_gene = F.relu(self.linear1(x[y, :]))
        x_cell = F.relu(self.linear2(x[~y, :]))
        x = self._cat(x_gene, x_cell, y)
        x = self.act(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.act(self.conv2(x, edge_index))
        return x


class SenGAE(GAE):
    def __init__(self):
        super().__init__(encoder=Encoder(), decoder=InnerProductDecoder())

    def forward(self, graph, split=10):
        return self.encode(graph)
