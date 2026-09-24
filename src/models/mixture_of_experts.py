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

    @staticmethod
    def load_balancing_loss(gate_weights):
        """
        Coefficient-of-variation load-balancing loss.

        Vanilla soft-gated MoE has no pressure to actually use all
        experts: the gate can collapse onto a single expert early
        in training, wasting the extra capacity the architecture
        was meant to provide. Penalizing the squared coefficient of
        variation of each expert's average gate weight across the
        batch encourages the router to spread load across experts,
        which is the standard fix used in sparsely/softly gated
        MoE models (e.g. Shazeer et al., 2017).
        """

        importance = gate_weights.sum(dim=0)

        mean = importance.mean()
        var = importance.var(unbiased=False)

        cv_squared = var / (mean ** 2 + 1e-10)

        return cv_squared

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

        aux_loss = self.load_balancing_loss(gate_weights)

        return self.output_norm(output), gate_weights, aux_loss