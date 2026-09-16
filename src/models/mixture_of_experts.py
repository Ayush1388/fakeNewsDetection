import torch
import torch.nn as nn


class Expert(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class AdaptiveMixtureOfExperts(nn.Module):

    def __init__(
        self,
        input_dim,
        expert_dim=256,
        num_experts=3,
        dropout=0.2,
    ):
        super().__init__()

        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                Expert(
                    input_dim,
                    expert_dim,
                    expert_dim,
                    dropout,
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, num_experts),
        )

        self.output_norm = nn.LayerNorm(
            expert_dim
        )

        self.output_dim = expert_dim

    def forward(self, x):

        expert_outputs = torch.stack(
            [
                expert(x)
                for expert in self.experts
            ],
            dim=1,
        )

        gate_logits = self.gate(x)

        gate_weights = torch.softmax(
            gate_logits,
            dim=-1,
        )

        output = torch.sum(
            expert_outputs
            * gate_weights.unsqueeze(-1),
            dim=1,
        )

        return self.output_norm(output), gate_weights