import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import GATConv, GAE


# Fixed architecture values used by the reported model. They match the PyG
# defaults used before these choices were made explicit.
GAT_HEADS = 1
GAT_CONCAT = True
GAT_ADD_SELF_LOOPS = True
GAT_NEGATIVE_SLOPE = 0.2


def require_projection_mode(model, type_specific_projections):
    """Reject checkpoints whose projection architecture does not match."""
    has_type_specific = (hasattr(model.encoder, 'gene_projection')
                         and hasattr(model.encoder, 'cell_projection'))
    if has_type_specific != type_specific_projections:
        expected = ('type-specific gene/cell projections'
                    if type_specific_projections else 'shared GAT projection')
        raise RuntimeError(
            f'This GAT checkpoint does not use the requested {expected}. '
            'Rerun deepsas_v1.py with --retrain.')


class GATEncoder(torch.nn.Module):
    """Two-layer GAT encoder for the binary DeepSAS graph."""

    def __init__(self, in_channels, out_channels, hidden_size, dropout,
                 type_specific_projections):
        super().__init__()
        self.type_specific_projections = type_specific_projections
        if type_specific_projections:
            self.gene_projection = nn.Linear(in_channels, in_channels)
            self.cell_projection = nn.Linear(in_channels, in_channels)
        # Fixed architecture used in the reported model: two GATConv layers.
        # These settings are architecture choices rather than dataset-tuning
        # parameters. Because DeepSAS builds a binary graph, there is no
        # separate edge-feature interface.
        gat_options = dict(
            heads=GAT_HEADS,
            concat=GAT_CONCAT,
            negative_slope=GAT_NEGATIVE_SLOPE,
            add_self_loops=GAT_ADD_SELF_LOOPS,
            dropout=dropout,
        )
        self.conv1 = GATConv(in_channels, hidden_size, **gat_options)
        self.conv2 = GATConv(hidden_size, out_channels, **gat_options)

    def project_node_types(self, x, is_gene):
        """Project gene and cell nodes into a shared feature space."""
        type_specific = getattr(
            self, 'type_specific_projections',
            hasattr(self, 'gene_projection') and hasattr(self, 'cell_projection'))
        if not type_specific:
            return x
        if is_gene.ndim != 1 or is_gene.numel() != x.shape[0]:
            raise ValueError('Node-type mask must contain one value per node')
        is_gene = is_gene.to(device=x.device, dtype=torch.bool)
        projected = torch.empty_like(x)
        projected[is_gene] = F.relu(self.gene_projection(x[is_gene]))
        projected[~is_gene] = F.relu(self.cell_projection(x[~is_gene]))
        return projected

    def forward(self, x, edge_index, is_gene=None):
        x = self.project_node_types(x, is_gene)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


class GAEModel(GAE):
    """Graph autoencoder = GATEncoder + InnerProductDecoder.

    This is the "graph autoencoder" referred to in Methods Section 1.3 of the
    manuscript.
    """

    def __init__(self, in_channels, out_channels, hidden_size, dropout,
                 type_specific_projections=True):
        encoder = GATEncoder(in_channels, out_channels,
                             hidden_size=hidden_size, dropout=dropout,
                             type_specific_projections=type_specific_projections)
        super().__init__(encoder)
        self.architecture_version = (
            'type-specific-projections-v1' if type_specific_projections
            else 'shared-projection-v1')

    def encode(self, x, edge_index, is_gene=None):
        return self.encoder(x, edge_index, is_gene)

    def get_attention_scores(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder.project_node_types(x, data.y)
        x = self.encoder.conv1(x, edge_index)
        x = F.elu(x)
        _, (edge_index_selfloop, alpha) = self.encoder.conv2(
            x, edge_index, return_attention_weights=True)
        return edge_index_selfloop, alpha
