import torch.nn as nn


class FeatureEncoder(nn.Module):

    def __init__(
        self,
        input_dim=14,
        hidden_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)