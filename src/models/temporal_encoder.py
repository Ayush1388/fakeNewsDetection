from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPropagationEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                8,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        delays,
    ):

        if delays.numel() == 0:

            statistics = torch.zeros(
                8,
                device=delays.device,
            )

        else:

            delays = torch.clamp(
                delays,
                min=0.0,
            )

            log_delays = torch.log1p(
                delays
            )

            statistics = torch.stack(
                [
                    log_delays.mean(),

                    log_delays.median(),

                    log_delays.max(),

                    delays.std()
                    if delays.numel() > 1
                    else torch.tensor(
                        0.0,
                        device=delays.device,
                    ),

                    (delays <= 1.0).float().mean(),

                    (delays <= 5.0).float().mean(),

                    (delays <= 30.0).float().mean(),

                    (delays <= 60.0).float().mean(),
                ]
            )

        return self.network(
            statistics
        )