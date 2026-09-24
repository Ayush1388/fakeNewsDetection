import torch

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:  # pragma: no cover
    get_linear_schedule_with_warmup = None


NO_DECAY_SUBSTRINGS = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")


def build_optimizer(
    model,
    backbone_lr=1e-5,
    head_lr=1e-4,
    weight_decay=0.01,
):
    """
    Two param groups (pretrained backbone vs. everything else),
    each split further into decay / no-decay.

    Two ideas, both standard practice for fine-tuning a pretrained
    transformer on top of a randomly-initialized head:

    1. Differential learning rates. The backbone already encodes
       general language knowledge; it only needs small nudges. The
       fusion/graph/MoE/classifier heads are randomly initialized
       and need a substantially larger learning rate to learn
       anything useful within a handful of epochs. Using a single
       small LR for both (as the original training scripts did)
       undertrains the heads; a single large LR destroys the
       pretrained backbone.
    2. No weight decay on biases and LayerNorm/bias parameters,
       which is the standard BERT/DeBERTa fine-tuning recipe.
    """

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        is_backbone = name.startswith("text_encoder.backbone")
        is_no_decay = any(s in name for s in NO_DECAY_SUBSTRINGS)

        if is_backbone and is_no_decay:
            backbone_no_decay.append(param)
        elif is_backbone:
            backbone_decay.append(param)
        elif is_no_decay:
            head_no_decay.append(param)
        else:
            head_decay.append(param)

    param_groups = []

    if backbone_decay:
        param_groups.append(
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay}
        )

    if backbone_no_decay:
        param_groups.append(
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0}
        )

    if head_decay:
        param_groups.append(
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay}
        )

    if head_no_decay:
        param_groups.append(
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0}
        )

    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer,
    num_training_steps,
    warmup_ratio=0.1,
):
    """
    Linear warmup followed by linear decay to zero, stepped once per
    optimizer step (not per epoch). Fine-tuning a transformer from
    step 1 at full learning rate is a well-known source of training
    instability on small datasets; a short warmup phase fixes this
    cheaply.
    """

    if get_linear_schedule_with_warmup is None:
        return None

    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
