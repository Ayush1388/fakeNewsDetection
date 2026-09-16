import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Encodes temporal propagation statistics into a fixed-size vector.
    """

    def __init__(
        self,
        input_dim=8,
        hidden_dim=128,
        output_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = output_dim

    def forward(self, temporal_features):
        if temporal_features.dim() == 1:
            temporal_features = temporal_features.unsqueeze(0)

        output = self.network(temporal_features)

        # A tree has one temporal-statistics vector, so return [output_dim].
        if output.size(0) == 1:
            return output.squeeze(0)

        return output.mean(dim=0)