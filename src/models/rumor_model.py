from __future__ import annotations

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .feature_encoder import FeatureEncoder
from .graph_encoder import PropagationGraphEncoder
from .temporal_encoder import TemporalPropagationEncoder


class PropagationFusion(nn.Module):

    def __init__(
        self,
        semantic_dim,
        linguistic_dim=256,
        graph_dim=256,
        temporal_dim=128,
        fusion_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.semantic_projection = nn.Linear(
            semantic_dim,
            fusion_dim,
        )

        self.linguistic_projection = nn.Linear(
            linguistic_dim,
            fusion_dim,
        )

        self.graph_projection = nn.Linear(
            graph_dim,
            fusion_dim,
        )

        self.temporal_projection = nn.Linear(
            temporal_dim,
            fusion_dim,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                fusion_dim * 4,
                fusion_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                fusion_dim,
                4,
            ),
        )

        self.norm = nn.LayerNorm(
            fusion_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        semantic,
        linguistic,
        graph,
        temporal,
    ):

        views = torch.stack(
            [
                self.semantic_projection(
                    semantic
                ),

                self.linguistic_projection(
                    linguistic
                ),

                self.graph_projection(
                    graph
                ),

                self.temporal_projection(
                    temporal
                ),
            ],
            dim=1,
        )

        attended, _ = self.attention(
            views,
            views,
            views,
            need_weights=False,
        )

        pooled = attended.mean(
            dim=1
        )

        gate_input = torch.cat(
            [
                views[:, 0],
                views[:, 1],
                views[:, 2],
                views[:, 3],
            ],
            dim=-1,
        )

        gate_weights = torch.softmax(
            self.gate(
                gate_input
            ),
            dim=-1,
        )

        gated = torch.sum(
            attended
            * gate_weights.unsqueeze(-1),
            dim=1,
        )

        fused = self.norm(
            pooled
            + self.dropout(gated)
        )

        return fused, gate_weights


class PropagationRumorModel(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name="microsoft/deberta-v3-base",
        linguistic_dim=14,
        graph_input_dim=8,
        feature_dim=256,
        temporal_dim=128,
        fusion_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            model_name=model_name,
            dropout=dropout,
        )

        self.feature_encoder = FeatureEncoder(
            input_dim=linguistic_dim,
            hidden_dim=feature_dim,
            dropout=dropout,
        )

        self.graph_encoder = PropagationGraphEncoder(
            input_dim=graph_input_dim,
            hidden_dim=feature_dim,
            layers=3,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalPropagationEncoder(
            hidden_dim=temporal_dim,
            dropout=dropout,
        )

        self.fusion = PropagationFusion(
            semantic_dim=self.text_encoder.output_dim,
            linguistic_dim=feature_dim,
            graph_dim=feature_dim,
            temporal_dim=temporal_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                fusion_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                fusion_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        linguistic_features,
        node_features,
        edge_index,
        delays,
    ):

        semantic = self.text_encoder(
            input_ids,
            attention_mask,
        )

        linguistic = self.feature_encoder(
            linguistic_features
        )

        graph_representations = []

        temporal_representations = []

        for nodes, edges, delay in zip(
            node_features,
            edge_index,
            delays,
        ):

            graph_representations.append(
                self.graph_encoder(
                    nodes,
                    edges,
                )
            )

            temporal_representations.append(
                self.temporal_encoder(
                    delay
                )
            )

        graph = torch.stack(
            graph_representations
        )

        temporal = torch.stack(
            temporal_representations
        )

        fused, gate_weights = self.fusion(
            semantic,
            linguistic,
            graph,
            temporal,
        )

        logits = self.classifier(
            fused
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        confidence, _ = probabilities.max(
            dim=-1
        )

        entropy = -torch.sum(
            probabilities
            * torch.log(
                probabilities + 1e-8
            ),
            dim=-1,
        )

        return {
            "logits": logits,
            "probabilities": probabilities,
            "confidence": confidence,
            "entropy": entropy,
            "gate_weights": gate_weights,
        }