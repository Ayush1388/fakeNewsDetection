import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.backbone.config.hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(dropout)

        self.output_dim = hidden_size

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = output.last_hidden_state

        # Keep the custom attention layer in the same
        # dtype as the DeBERTa hidden states.
        attention_dtype = hidden.dtype

        scores = self.attention(
            hidden.to(
                self.attention[0].weight.dtype
            )
        ).squeeze(-1)

        scores = scores.to(attention_dtype)

        scores = scores.masked_fill(
            attention_mask == 0,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        pooled = torch.sum(
            hidden * weights.unsqueeze(-1),
            dim=1,
        )

        return self.dropout(pooled)