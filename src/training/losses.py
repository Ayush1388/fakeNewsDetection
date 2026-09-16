import torch
import torch.nn as nn


class WeightedCrossEntropy(nn.Module):

    def __init__(self, class_weights=None):

        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            weight=class_weights
        )

    def forward(
        self,
        logits,
        labels,
    ):
        return self.loss(
            logits,
            labels,
        )