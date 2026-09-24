import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from src.data.loaders import load_dataset
from src.data.graph_dataset import (
    PropagationDataset,
    propagation_collate,
)
from src.data.dataset import FakeNewsDataset
from src.data.features import NUM_LINGUISTIC_FEATURES
from src.models.rumor_model import PropagationRumorModel
from src.models.deberta_baseline import DeBERTaBaseline
from src.models.tegfnd import TEGFND
from src.training.losses import WeightedCrossEntropy, compute_class_weights
from src.training.optim import build_optimizer, build_scheduler
from src.training.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset_name):
    if dataset_name in {"twitter15", "twitter16"}:
        return 4

    if dataset_name == "politifact":
        return 6

    raise ValueError(
        f"Unsupported dataset: {dataset_name}"
    )


def get_dataset_path(dataset_name):
    paths = {
        "twitter15": "data/twitter15",
        "twitter16": "data/twitter16",
        "politifact": (
            "data/politifact/"
            "politifact_factcheck_data.json"
        ),
    }

    if dataset_name not in paths:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    return paths[dataset_name]


def get_tree_dir(dataset_name):
    if dataset_name == "twitter15":
        return "data/twitter15/tree"

    if dataset_name == "twitter16":
        return "data/twitter16/tree"

    raise ValueError(
        "Propagation trees are currently available "
        "for Twitter15 and Twitter16 only."
    )


def build_model(
    model_type,
    num_classes,
    model_name,
    freeze_layers,
):
    if model_type == "deberta":
        return DeBERTaBaseline(
            num_classes=num_classes,
            model_name=model_name,
            freeze_layers=freeze_layers,
        )

    if model_type == "tegfnd":
        return TEGFND(
            num_classes=num_classes,
            model_name=model_name,
            linguistic_dim=NUM_LINGUISTIC_FEATURES,
            freeze_layers=freeze_layers,
        )

    if model_type == "propagation":
        return PropagationRumorModel(
            num_classes=num_classes,
            model_name=model_name,
            linguistic_dim=NUM_LINGUISTIC_FEATURES,
            graph_input_dim=8,
            feature_dim=256,
            temporal_dim=256,
            fusion_dim=256,
            dropout=0.3,
            freeze_layers=freeze_layers,
        )

    raise ValueError(
        f"Unsupported model: {model_type}"
    )


def create_splits(
    dataframe,
    seed,
):
    """
    Stratified 70/15/15 split at the dataframe level.

    The previous implementation used ``torch.utils.data.random_split``
    on the *dataset* object, which ignores class labels entirely.
    On a small dataset (Twitter15/16 have ~1.5k and ~0.8k examples
    across 4 balanced-but-not-guaranteed-balanced classes) an
    unstratified split can easily shift the class balance between
    train/val/test enough to bias both training and the reported
    test accuracy. This matches the stratified-split methodology the
    base papers use (and what the README already documented, but the
    code did not actually do).
    """

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=seed,
        stratify=dataframe["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "twitter15",
            "twitter16",
            "politifact",
        ],
    )

    parser.add_argument(
        "--model",
        type=str,
        default="tegfnd",
        choices=[
            "tegfnd",
            "propagation",
            "deberta",
        ],
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)

    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")

    # A first tegfnd run (text + linguistic features only, no
    # propagation graph) on Twitter15 hit Train F1 0.99 vs Val F1
    # 0.72 by epoch 13-15 -- a textbook small-dataset transformer
    # overfit (DeBERTa-v3-base is ~86M parameters against ~1043
    # training examples). freeze_layers=4 was not nearly enough to
    # control that gap, so the default is raised to 8 (of
    # deberta-v3-base's 12 encoder layers, leaving only the top 4 +
    # the task heads trainable). See README for why --model
    # propagation is the bigger lever here.
    parser.add_argument("--freeze-layers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume from (either the "
            "'*_last.pt' checkpoint saved every epoch, or a "
            "'best' checkpoint). Restores model/optimizer/"
            "scheduler state and continues from the next epoch. "
            "--epochs should be the ORIGINAL total epoch budget "
            "(e.g. still 15, not 'epochs remaining')."
        ),
    )

    args = parser.parse_args(argv)

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset_path = get_dataset_path(
        args.dataset
    )

    dataframe = load_dataset(
        args.dataset,
        dataset_path,
    )

    print(
        f"Loaded samples: {len(dataframe)}"
    )

    # --------------------------------------------------
    # Stratified train / validation / test split
    # --------------------------------------------------

    train_df, val_df, test_df = create_splits(
        dataframe,
        args.seed,
    )

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    model_name = (
        "microsoft/deberta-v3-base"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # --------------------------------------------------
    # Dataset construction
    # --------------------------------------------------

    if args.model == "propagation":

        if args.dataset not in {
            "twitter15",
            "twitter16",
        }:
            raise ValueError(
                "Propagation model currently supports "
                "Twitter15 and Twitter16 only."
            )

        tree_dir = get_tree_dir(args.dataset)

        train_dataset = PropagationDataset(
            train_df, tokenizer, tree_dir, args.max_length,
        )
        val_dataset = PropagationDataset(
            val_df, tokenizer, tree_dir, args.max_length,
        )
        test_dataset = PropagationDataset(
            test_df, tokenizer, tree_dir, args.max_length,
        )

        collate_fn = propagation_collate

    else:

        train_dataset = FakeNewsDataset(
            train_df, tokenizer, max_length=args.max_length,
        )
        val_dataset = FakeNewsDataset(
            val_df, tokenizer, max_length=args.max_length,
        )
        test_dataset = FakeNewsDataset(
            test_df, tokenizer, max_length=args.max_length,
        )

        collate_fn = None

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    num_classes = get_num_classes(
        args.dataset
    )

    model = build_model(
        model_type=args.model,
        num_classes=num_classes,
        model_name=model_name,
        freeze_layers=args.freeze_layers,
    )

    model = model.to(device).float()

    # --------------------------------------------------
    # Optimizer (differential LR + no-decay groups)
    # --------------------------------------------------

    optimizer = build_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------
    # LR schedule: linear warmup -> linear decay,
    # stepped once per optimizer step.
    # --------------------------------------------------

    steps_per_epoch = max(
        1,
        len(train_loader) // args.grad_accum_steps,
    )

    num_training_steps = steps_per_epoch * args.epochs

    scheduler = build_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=args.warmup_ratio,
    )

    # --------------------------------------------------
    # Loss: label smoothing + optional class weights
    # --------------------------------------------------

    class_weights = None

    if args.use_class_weights:
        class_weights = compute_class_weights(
            train_df["label"].to_numpy(),
            num_classes,
        ).to(device)

        print(f"Class weights: {class_weights.tolist()}")

    criterion = WeightedCrossEntropy(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = (
            "results/checkpoints/"
            f"{args.dataset}_"
            f"{args.model}_"
            f"seed{args.seed}.pt"
        )

    print(f"Checkpoint: {save_path}")

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_path=save_path,
        scheduler=scheduler,
        grad_accum_steps=args.grad_accum_steps,
        patience=args.patience,
    )

    # --------------------------------------------------
    # Resume from a checkpoint, if requested.
    # --------------------------------------------------

    start_epoch = 1
    best_val_f1 = -float("inf")

    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")

        start_epoch, best_val_f1 = Trainer.load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        print(
            f"Resuming at epoch {start_epoch}/{args.epochs} "
            f"(best val F1 so far: {best_val_f1:.4f})"
        )

        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint is already past --epochs={args.epochs} "
                f"(checkpoint epoch={start_epoch - 1}). Increase "
                "--epochs to continue training."
            )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        best_val_f1=best_val_f1,
    )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    history_path = save_path.replace(
        ".pt",
        "_history.pt",
    )

    torch.save(
        history,
        history_path,
    )

    print(
        f"\nTraining history saved → "
        f"{history_path}"
    )

    print(
        f"\nBest validation macro-F1: "
        f"{history['best_val_f1']:.4f}"
    )

    print(
        "\nTraining complete. Run scripts/evaluate.py "
        "against the saved checkpoint to get the held-out "
        "test-set accuracy."
    )

    return history


if __name__ == "__main__":
    main()
