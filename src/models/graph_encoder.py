from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        dim,
        dropout=0.2,
    ):
        super().__init__()

        self.projection = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.source_attention = nn.Linear(
            dim,
            1,
            bias=False,
        )

        self.target_attention = nn.Linear(
            dim,
            1,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        edge_index,
    ):

        h = self.projection(x)

        if edge_index.numel() == 0:
            return self.norm(
                x + self.dropout(
                    F.gelu(h)
                )
            )

        source = edge_index[0]
        target = edge_index[1]

        scores = (
            self.source_attention(
                h[source]
            )
            +
            self.target_attention(
                h[target]
            )
        )

        scores = F.leaky_relu(
            scores.squeeze(-1),
            negative_slope=0.2,
        )

        attention = torch.zeros_like(
            scores
        )

        unique_targets = torch.unique(
            target
        )

        for node in unique_targets:

            mask = (
                target == node
            )

            attention[mask] = torch.softmax(
                scores[mask],
                dim=0,
            )

        messages = torch.zeros_like(h)

        messages.index_add_(
            0,
            target,
            h[source]
            * attention.unsqueeze(-1),
        )

        # Reverse information flow.
        reverse_messages = torch.zeros_like(
            h
        )

        reverse_messages.index_add_(
            0,
            source,
            h[target]
            * attention.unsqueeze(-1),
        )

        output = (
            h
            + messages
            + reverse_messages
        )

        output = F.gelu(
            output
        )

        output = self.dropout(
            output
        )

        return self.norm(
            x + output
        )


class PropagationGraphEncoder(nn.Module):

    def __init__(
        self,
        input_dim=8,
        hidden_dim=256,
        layers=3,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    hidden_dim,
                    dropout,
                )
                for _ in range(layers)
            ]
        )

        self.pool_attention = nn.Sequential(
            nn.Linear(
                hidden_dim,
                64,
            ),
            nn.Tanh(),
            nn.Linear(
                64,
                1,
            ),
        )

        self.output = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        node_features,
        edge_index,
    ):

        x = self.input_projection(
            node_features
        )

        for layer in self.layers:
            x = layer(
                x,
                edge_index,
            )

        scores = self.pool_attention(
            x
        )

        weights = torch.softmax(
            scores.squeeze(-1),
            dim=0,
        )

        graph_representation = torch.sum(
            x * weights.unsqueeze(-1),
            dim=0,
        )

        return self.output(
            graph_representation
        )