import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Multi-head graph attention layer.

    Messages flow from parent -> child and child -> parent so that
    each propagation node can incorporate information from its
    local propagation neighborhood.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        if output_dim % heads != 0:
            raise ValueError(
                f"output_dim ({output_dim}) must be divisible by "
                f"heads ({heads})."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = output_dim // heads

        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)

        self.output_projection = nn.Linear(
            output_dim,
            output_dim,
            bias=False,
        )

        self.residual = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x:
                Node features [num_nodes, input_dim]

            edge_index:
                Directed edges [2, num_edges].
                edge_index[0] = source node
                edge_index[1] = destination node

        Returns:
            Node representations [num_nodes, output_dim]
        """

        num_nodes = x.size(0)

        if num_nodes == 0:
            return x.new_zeros((0, self.output_dim))

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(num_nodes, self.heads, self.head_dim)
        k = k.view(num_nodes, self.heads, self.head_dim)
        v = v.view(num_nodes, self.heads, self.head_dim)

        # Add reverse edges so information propagates in both directions.
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]

            src_all = torch.cat([src, dst], dim=0)
            dst_all = torch.cat([dst, src], dim=0)
        else:
            src_all = torch.empty(
                0,
                dtype=torch.long,
                device=x.device,
            )
            dst_all = torch.empty(
                0,
                dtype=torch.long,
                device=x.device,
            )

        # Include self-loops.
        self_nodes = torch.arange(
            num_nodes,
            device=x.device,
            dtype=torch.long,
        )

        src_all = torch.cat([src_all, self_nodes], dim=0)
        dst_all = torch.cat([dst_all, self_nodes], dim=0)

        # Attention score for each edge and head.
        scores = (
            q[dst_all] * k[src_all]
        ).sum(dim=-1) / (self.head_dim ** 0.5)

        # Stable edge-wise softmax.
        attention = torch.zeros_like(scores)

        for node in range(num_nodes):
            mask = dst_all == node

            if mask.any():
                attention[mask] = F.softmax(
                    scores[mask],
                    dim=0,
                )

        messages = v[src_all] * attention.unsqueeze(-1)

        aggregated = torch.zeros(
            num_nodes,
            self.heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        aggregated.index_add_(
            0,
            dst_all,
            messages,
        )

        aggregated = aggregated.reshape(
            num_nodes,
            self.output_dim,
        )

        aggregated = self.output_projection(aggregated)
        aggregated = self.dropout(aggregated)

        # Residual connection.
        aggregated = self.norm(
            aggregated + self.residual(x)
        )

        return F.gelu(aggregated)


class GraphEncoder(nn.Module):
    """
    Propagation-tree graph encoder.

    Input:
        Node-level propagation features.

    Output:
        One fixed-size representation for the entire propagation tree.
    """

    def __init__(
        self,
        input_dim=8,
        hidden_dim=256,
        output_dim=256,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_layer_1 = GraphAttentionLayer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        self.graph_layer_2 = GraphAttentionLayer(
            input_dim=hidden_dim,
            output_dim=output_dim,
            heads=heads,
            dropout=dropout,
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )

        self.output_norm = nn.LayerNorm(output_dim)

        self.output_dim = output_dim

    def forward(
        self,
        node_features,
        edge_index,
    ):
        """
        Args:
            node_features:
                [num_nodes, input_dim]

            edge_index:
                [2, num_edges]

        Returns:
            graph_representation:
                [output_dim]
        """

        if node_features.dim() != 2:
            raise ValueError(
                "node_features must have shape "
                "[num_nodes, input_dim]."
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]."
            )

        x = self.input_projection(node_features)

        x = self.graph_layer_1(
            x,
            edge_index,
        )

        x = self.graph_layer_2(
            x,
            edge_index,
        )

        # Learned attention pooling over propagation nodes.
        scores = self.attention_pool(x).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=0,
        )

        graph_representation = torch.sum(
            x * weights.unsqueeze(-1),
            dim=0,
        )

        graph_representation = self.output_norm(
            graph_representation
        )

        return graph_representation