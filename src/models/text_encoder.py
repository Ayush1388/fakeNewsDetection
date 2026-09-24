import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Transformer text encoder with multi-view pooling.

    Rather than relying on a single attention-pooled vector, the
    encoder concatenates three complementary poolings of the last
    hidden state:

        1. Learned attention pooling (focuses on the most
           discriminative tokens).
        2. Mean pooling (a stable, low-variance summary of the
           whole sequence).
        3. Max pooling (captures salient/extreme token signals,
           e.g. strongly emotional or sensational words).

    This concat-pooling is a standard, cheap way to improve small
    transformer classification datasets without adding parameters
    to the backbone itself.

    ``freeze_layers`` optionally freezes the embeddings and the
    first N transformer layers. Fully fine-tuning a ~86M parameter
    backbone on a ~1-2k example dataset (Twitter15/16, PolitiFact)
    is a major overfitting risk; freezing the lower, more generic
    layers and only adapting the upper layers is a standard
    mitigation for small-data transformer fine-tuning.
    """

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        dropout=0.1,
        freeze_layers=0,
    ):
        super().__init__()

        # Always keep the pretrained backbone parameters in FP32.
        # CUDA AMP will handle FP16 computation during training.
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        self.backbone.float()

        if freeze_layers > 0:
            self._freeze_lower_layers(freeze_layers)

        hidden_size = self.backbone.config.hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)

        self.output_dim = hidden_size

    def _freeze_lower_layers(self, freeze_layers):
        """
        Freeze the embedding module and the first ``freeze_layers``
        transformer encoder layers. Works across the common
        HuggingFace encoder layouts (BERT/DeBERTa/RoBERTa-style
        ``encoder.layer``).
        """

        for param in self.backbone.get_input_embeddings().parameters():
            param.requires_grad = False

        encoder = getattr(self.backbone, "encoder", None)

        layers = None

        if encoder is not None:
            layers = getattr(encoder, "layer", None)

            if layers is None:
                layers = getattr(encoder, "layers", None)

        if layers is None:
            return

        for layer in list(layers)[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

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

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        # ------------------------------------------------------
        # Attention pooling.
        # ------------------------------------------------------

        scores = self.attention(hidden).squeeze(-1)

        scores = scores.masked_fill(
            attention_mask == 0,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_pooled = torch.sum(
            hidden * weights.unsqueeze(-1),
            dim=1,
        )

        # ------------------------------------------------------
        # Mean pooling (mask-aware).
        # ------------------------------------------------------

        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        mean_pooled = summed / counts

        # ------------------------------------------------------
        # Max pooling (mask-aware).
        # ------------------------------------------------------

        masked_hidden = hidden.masked_fill(
            mask == 0,
            torch.finfo(hidden.dtype).min,
        )

        max_pooled, _ = masked_hidden.max(dim=1)

        # ------------------------------------------------------
        # Fuse the three views.
        # ------------------------------------------------------

        pooled = torch.cat(
            [attention_pooled, mean_pooled, max_pooled],
            dim=-1,
        )

        pooled = self.pool_projection(pooled)

        return self.dropout(pooled)
