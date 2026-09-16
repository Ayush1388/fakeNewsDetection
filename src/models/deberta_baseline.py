import torch
import torch.nn as nn

from .text_encoder import TextEncoder


class DeBERTaBaseline(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name="microsoft/deberta-v3-base",
        dropout=0.2,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            model_name=model_name,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(
                self.text_encoder.output_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        linguistic_features=None,
    ):

        semantic = self.text_encoder(
            input_ids,
            attention_mask,
        )

        logits = self.classifier(
            semantic
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
        }