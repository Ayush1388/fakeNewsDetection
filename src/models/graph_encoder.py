import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Multi-head graph attention layer.

    Messages flow from parent -> child and child -> parent so that
    each propagation node can incorporate information from its
    local propagation neighborhood.

    Attention logits and softmax are computed in float32 to remain
    numerically stable when the surrounding model is running under
    mixed precision.
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
                f"{heads}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = output_dim // heads

        self.query = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

        self.key = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

        self.value = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

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
            return x.new_zeros(
                (0, self.output_dim)
            )

        # ---------------------------------------------------------
        # Project node features.
        # ---------------------------------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(
            num_nodes,
            self.heads,
            self.head_dim,
        )

        k = k.view(
            num_nodes,
            self.heads,
            self.head_dim,
        )

        v = v.view(
            num_nodes,
            self.heads,
            self.head_dim,
        )

        # ---------------------------------------------------------
        # Build bidirectional propagation edges.
        # ---------------------------------------------------------

        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]

            src_all = torch.cat(
                [src, dst],
                dim=0,
            )

            dst_all = torch.cat(
                [dst, src],
                dim=0,
            )

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

        # ---------------------------------------------------------
        # Add self-loops.
        # ---------------------------------------------------------

        self_nodes = torch.arange(
            num_nodes,
            device=x.device,
            dtype=torch.long,
        )

        src_all = torch.cat(
            [src_all, self_nodes],
            dim=0,
        )

        dst_all = torch.cat(
            [dst_all, self_nodes],
            dim=0,
        )

        # ---------------------------------------------------------
        # Attention scores.
        #
        # IMPORTANT:
        # Compute QK^T in float32 even when x is fp16/bf16.
        # This prevents overflow/underflow in the custom
        # attention branch during mixed-precision training.
        # ---------------------------------------------------------

        q_edges = q[dst_all].float()
        k_edges = k[src_all].float()

        scores = (
            q_edges * k_edges
        ).sum(dim=-1) / (
            self.head_dim ** 0.5
        )

        # ---------------------------------------------------------
        # Stable edge-wise softmax, vectorized.
        #
        # This used to loop over every node in Python
        # (`for node in range(num_nodes)`), which is O(num_nodes *
        # num_edges) and became a serious bottleneck (and a plausible
        # cause of apparent training "hangs") on the larger
        # propagation trees in this dataset, some of which have
        # hundreds of nodes. It's replaced with a scatter-based
        # per-destination-node softmax that does the exact same
        # computation (max subtraction per destination node per
        # head, then softmax over incoming edges) in O(num_edges),
        # with no Python-level loop.
        #
        # Softmax is still explicitly performed in float32.
        # ---------------------------------------------------------

        dst_expanded = dst_all.unsqueeze(-1).expand(-1, self.heads)

        max_per_dst = torch.full(
            (num_nodes, self.heads),
            float("-inf"),
            dtype=torch.float32,
            device=x.device,
        )

        max_per_dst.scatter_reduce_(
            0,
            dst_expanded,
            scores,
            reduce="amax",
            include_self=True,
        )

        # Every node has a self-loop, so every row of max_per_dst is
        # guaranteed to have been written at least once (no -inf
        # rows survive to the subtraction below).
        scores_shifted = scores - max_per_dst.gather(0, dst_expanded)

        exp_scores = torch.exp(scores_shifted)

        sum_per_dst = torch.zeros(
            (num_nodes, self.heads),
            dtype=torch.float32,
            device=x.device,
        )

        sum_per_dst.index_add_(0, dst_all, exp_scores)

        attention = exp_scores / (
            sum_per_dst.gather(0, dst_expanded) + 1e-16
        )

        # ---------------------------------------------------------
        # Message passing.
        #
        # Convert attention weights back to the value dtype so
        # aggregation remains compatible with mixed precision.
        # ---------------------------------------------------------

        attention = attention.to(v.dtype)

        messages = (
            v[src_all]
            * attention.unsqueeze(-1)
        )

        aggregated = torch.zeros(
            num_nodes,
            self.heads,
            self.head_dim,
            device=x.device,
            dtype=v.dtype,
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

        # ---------------------------------------------------------
        # Output projection.
        # ---------------------------------------------------------

        aggregated = self.output_projection(
            aggregated
        )

        aggregated = self.dropout(
            aggregated
        )

        # ---------------------------------------------------------
        # Residual connection + normalization.
        # ---------------------------------------------------------

        residual = self.residual(x)

        aggregated = self.norm(
            aggregated + residual
        )

        return F.gelu(aggregated)


class GraphEncoder(nn.Module):
    """
    Propagation-tree graph encoder.

    Input:
        Node-level propagation features.

    Output:
        One fixed-size representation for the entire
        propagation tree.
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
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
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
            nn.Linear(
                output_dim,
                output_dim // 2,
            ),
            nn.Tanh(),
            nn.Linear(
                output_dim // 2,
                1,
            ),
        )

        self.output_norm = nn.LayerNorm(
            output_dim
        )

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

        if (
            edge_index.dim() != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError(
                "edge_index must have shape "
                "[2, num_edges]."
            )

        # ---------------------------------------------------------
        # Input projection.
        # ---------------------------------------------------------

        x = self.input_projection(
            node_features
        )

        # ---------------------------------------------------------
        # Graph message-passing layers.
        # ---------------------------------------------------------

        x = self.graph_layer_1(
            x,
            edge_index,
        )

        x = self.graph_layer_2(
            x,
            edge_index,
        )

        # ---------------------------------------------------------
        # Learned attention pooling over propagation nodes.
        #
        # Pooling scores are also calculated in float32 so the
        # softmax remains stable under mixed precision.
        # ---------------------------------------------------------

        scores = self.attention_pool(
            x
        ).squeeze(-1)

        scores_float = scores.float()

        scores_float = (
            scores_float
            - scores_float.max(
                dim=0,
                keepdim=True,
            ).values
        )

        weights = torch.softmax(
            scores_float,
            dim=0,
        )

        weights = weights.to(x.dtype)

        graph_representation = torch.sum(
            x * weights.unsqueeze(-1),
            dim=0,
        )

        graph_representation = self.output_norm(
            graph_representation
        )

        return graph_representation