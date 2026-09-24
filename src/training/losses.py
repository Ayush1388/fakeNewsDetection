import torch
import torch.nn as nn


class WeightedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with optional class weighting and label
    smoothing.

    Label smoothing (typically 0.05-0.1) is a cheap, well-tested
    regularizer for small, noisily-labeled text classification
    datasets: it prevents the model from becoming over-confident on
    a handful of thousand training examples and tends to improve
    generalization on the held-out test split.
    """

    def __init__(
        self,
        class_weights=None,
        label_smoothing=0.0,
    ):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, labels):
        return self.loss(logits, labels)


def compute_class_weights(labels, num_classes):
    """
    Inverse-frequency class weights, normalized so the mean weight
    is 1.0. Robust to missing classes in a split (assigns weight 1).
    """

    labels = torch.as_tensor(labels, dtype=torch.long)

    counts = torch.bincount(
        labels,
        minlength=num_classes,
    ).float()

    weights = torch.where(
        counts > 0,
        counts.sum() / (num_classes * counts.clamp(min=1)),
        torch.ones_like(counts),
    )

    return weights
