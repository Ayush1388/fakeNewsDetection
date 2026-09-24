from __future__ import annotations

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .feature_encoder import FeatureEncoder
from .graph_encoder import GraphEncoder
from .temporal_encoder import TemporalEncoder
from ..data.features import NUM_LINGUISTIC_FEATURES


class PropagationFusion(nn.Module):
    """
    Fuses four complementary views:

        1. Semantic representation from DeBERTa
        2. Linguistic feature representation
        3. Propagation graph representation
        4. Temporal propagation representation

    Cross-view attention is followed by a learned gating mechanism.
    """

    def __init__(
        self,
        semantic_dim,
        linguistic_dim=256,
        graph_dim=256,
        temporal_dim=256,
        fusion_dim=256,
        dropout=0.3,
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
        """
        Args:
            semantic:
                [batch_size, semantic_dim]

            linguistic:
                [batch_size, linguistic_dim]

            graph:
                [batch_size, graph_dim]

            temporal:
                [batch_size, temporal_dim]

        Returns:
            fused:
                [batch_size, fusion_dim]

            gate_weights:
                [batch_size, 4]
        """

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

        # Cross-view self-attention.
        attended, _ = self.attention(
            views,
            views,
            views,
            need_weights=False,
        )

        # Mean-pooled attended representation.
        pooled = attended.mean(
            dim=1
        )

        # Learn how much each view should contribute.
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
    """
    Full fake-news / rumor detection model.

    Architecture:

        Source text
              |
           DeBERTa
              |
        semantic vector
              |
              +----------------------+
                                     |
        Linguistic features          |
              |                      |
        FeatureEncoder               |
              |                      |
              +----------------------+
                                     |
        Propagation tree              |
              |                      |
        GraphEncoder                 |
              |                      |
              +----------------------+
                                     |
        Propagation delays            |
              |                      |
        Temporal statistics           |
              |                      |
        TemporalEncoder              |
              |                      |
              +----------------------+
                                     |
                              PropagationFusion
                                     |
                              Classification head
    """

    def __init__(
        self,
        num_classes,
        model_name="microsoft/deberta-v3-base",
        linguistic_dim=NUM_LINGUISTIC_FEATURES,
        graph_input_dim=8,
        feature_dim=256,
        temporal_dim=256,
        fusion_dim=256,
        dropout=0.3,
        freeze_layers=8,
    ):
        super().__init__()

        # --------------------------------------------------
        # Text branch
        # --------------------------------------------------

        self.text_encoder = TextEncoder(
            model_name=model_name,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )

        # --------------------------------------------------
        # Linguistic feature branch
        # --------------------------------------------------

        self.feature_encoder = FeatureEncoder(
            input_dim=linguistic_dim,
            hidden_dim=feature_dim,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Propagation graph branch
        # --------------------------------------------------

        self.graph_encoder = GraphEncoder(
            input_dim=graph_input_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            heads=4,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Temporal propagation branch
        # --------------------------------------------------

        self.temporal_encoder = TemporalEncoder(
            input_dim=8,
            hidden_dim=128,
            output_dim=temporal_dim,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Multi-view fusion
        # --------------------------------------------------

        self.fusion = PropagationFusion(
            semantic_dim=self.text_encoder.output_dim,
            linguistic_dim=feature_dim,
            graph_dim=feature_dim,
            temporal_dim=temporal_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                fusion_dim,
                fusion_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                fusion_dim,
                num_classes,
            ),
        )

    @staticmethod
    def build_temporal_features(delays):
        """
        Convert raw propagation delays into eight temporal
        statistics expected by TemporalEncoder.

        Features:

            1. Mean log delay
            2. Median log delay
            3. Maximum log delay
            4. Standard deviation of log delay
            5. Fraction of nodes within 1 minute
            6. Fraction of nodes within 5 minutes
            7. Fraction of nodes within 30 minutes
            8. Fraction of nodes within 60 minutes

        Args:
            delays:
                [num_nodes]

        Returns:
            [8]
        """

        delays = delays.float()

        # Prevent invalid negative propagation delays.
        delays = torch.clamp(
            delays,
            min=0.0,
        )

        log_delays = torch.log1p(
            delays
        )

        mean_log_delay = log_delays.mean()

        median_log_delay = torch.median(
            log_delays
        )

        max_log_delay = log_delays.max()

        std_log_delay = (
            log_delays.std(
                unbiased=False
            )
        )

        within_1 = (
            delays <= 1.0
        ).float().mean()

        within_5 = (
            delays <= 5.0
        ).float().mean()

        within_30 = (
            delays <= 30.0
        ).float().mean()

        within_60 = (
            delays <= 60.0
        ).float().mean()

        return torch.stack(
            [
                mean_log_delay,
                median_log_delay,
                max_log_delay,
                std_log_delay,
                within_1,
                within_5,
                within_30,
                within_60,
            ]
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
        """
        Args:
            input_ids:
                [batch_size, sequence_length]

            attention_mask:
                [batch_size, sequence_length]

            linguistic_features:
                [batch_size, 14]

            node_features:
                Batch/list of propagation node tensors.
                Each tree:
                    [num_nodes, 8]

            edge_index:
                Batch/list of propagation edge tensors.
                Each tree:
                    [2, num_edges]

            delays:
                Batch/list of propagation delay tensors.
                Each tree:
                    [num_nodes]

        Returns:
            Dictionary containing logits, probabilities,
            confidence, entropy and fusion gate weights.
        """

        # --------------------------------------------------
        # Text representation
        # --------------------------------------------------

        semantic = self.text_encoder(
            input_ids,
            attention_mask,
        )

        # --------------------------------------------------
        # Match the dtype expected by the custom
        # fusion layers.
        #
        # DeBERTa may run in FP16 on CUDA while the
        # custom layers remain in FP32.
        # --------------------------------------------------

        semantic_dtype = (
            self.fusion
            .semantic_projection
            .weight
            .dtype
        )

        semantic = semantic.to(
            semantic_dtype
        )

        # --------------------------------------------------
        # Linguistic representation
        # --------------------------------------------------

        linguistic = self.feature_encoder(
            linguistic_features
        )

        # --------------------------------------------------
        # Propagation representations
        # --------------------------------------------------

        graph_representations = []
        temporal_representations = []

        for nodes, edges, delay in zip(
            node_features,
            edge_index,
            delays,
        ):

            # Graph representation
            graph_representation = self.graph_encoder(
                nodes,
                edges,
            )

            graph_representations.append(
                graph_representation
            )

            # Build the eight temporal statistics
            temporal_features = (
                self.build_temporal_features(
                    delay
                )
            )

            # Temporal representation
            temporal_representation = (
                self.temporal_encoder(
                    temporal_features
                )
            )

            temporal_representations.append(
                temporal_representation
            )

        graph = torch.stack(
            graph_representations
        )

        temporal = torch.stack(
            temporal_representations
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        fused, gate_weights = self.fusion(
            semantic,
            linguistic,
            graph,
            temporal,
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

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