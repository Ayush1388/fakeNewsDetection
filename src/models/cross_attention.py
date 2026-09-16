import torch
import torch.nn as nn


class CrossViewAttention(nn.Module):

    def __init__(
        self,
        semantic_dim,
        feature_dim,
        fusion_dim=256,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.semantic_projection = nn.Linear(
            semantic_dim,
            fusion_dim,
        )

        self.feature_projection = nn.Linear(
            feature_dim,
            fusion_dim,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        semantic,
        features,
    ):

        semantic = self.semantic_projection(
            semantic
        )

        features = self.feature_projection(
            features
        )

        views = torch.stack(
            [semantic, features],
            dim=1,
        )

        attended, _ = self.attention(
            views,
            views,
            views,
        )

        fused = self.norm(
            views + attended
        )

        return fused.mean(dim=1)