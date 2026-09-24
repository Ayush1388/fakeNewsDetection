import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .feature_encoder import FeatureEncoder
from .cross_attention import CrossViewAttention
from .mixture_of_experts import AdaptiveMixtureOfExperts
from ..data.features import NUM_LINGUISTIC_FEATURES


class TEGFND(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name="microsoft/deberta-v3-base",
        linguistic_dim=NUM_LINGUISTIC_FEATURES,
        feature_dim=256,
        fusion_dim=256,
        expert_dim=256,
        num_experts=4,
        dropout=0.2,
        freeze_layers=4,
        moe_aux_loss_weight=0.01,
    ):
        super().__init__()

        self.moe_aux_loss_weight = moe_aux_loss_weight

        self.text_encoder = TextEncoder(
            model_name=model_name,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )

        self.feature_encoder = FeatureEncoder(
            input_dim=linguistic_dim,
            hidden_dim=feature_dim,
            dropout=dropout,
        )

        self.cross_attention = CrossViewAttention(
            semantic_dim=self.text_encoder.output_dim,
            feature_dim=feature_dim,
            fusion_dim=fusion_dim,
            heads=4,
            dropout=dropout,
        )

        self.moe = AdaptiveMixtureOfExperts(
            input_dim=fusion_dim,
            expert_dim=expert_dim,
            num_experts=num_experts,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, num_classes),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        linguistic_features,
    ):

        semantic = self.text_encoder(
            input_ids,
            attention_mask,
        )

        linguistic = self.feature_encoder(
            linguistic_features
        )

        fused = self.cross_attention(
            semantic,
            linguistic,
        )

        representation, gate_weights, moe_aux_loss = self.moe(
            fused
        )

        logits = self.classifier(
            representation
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
            "aux_loss": self.moe_aux_loss_weight * moe_aux_loss,
        }
