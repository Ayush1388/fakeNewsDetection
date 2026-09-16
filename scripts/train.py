import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from src.data.loaders import load_dataset
from src.data.graph_dataset import (
    PropagationDataset,
    propagation_collate,
)
from src.models.rumor_model import PropagationRumorModel
from src.models.deberta_baseline import DeBERTaBaseline
from src.training.losses import WeightedCrossEntropy
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
):
    if model_type == "deberta":
        return DeBERTaBaseline(
            num_classes=num_classes,
            model_name=model_name,
        )

    if model_type == "propagation":
        return PropagationRumorModel(
            num_classes=num_classes,
            model_name=model_name,
            linguistic_dim=14,
            graph_input_dim=8,
            feature_dim=256,
            temporal_dim=256,
            fusion_dim=256,
            dropout=0.2,
        )

    raise ValueError(
        f"Unsupported model: {model_type}"
    )


def create_splits(
    dataset,
    seed,
):
    total_size = len(dataset)

    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )


def main():

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
        default="propagation",
        choices=[
            "propagation",
            "deberta",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

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

    dataframe = load_dataset(
        args.dataset
    )

    print(
        f"Loaded samples: {len(dataframe)}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    model_name = "microsoft/deberta-v3-base"

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

        dataset = PropagationDataset(
            dataframe=dataframe,
            tokenizer=tokenizer,
            tree_dir=get_tree_dir(
                args.dataset
            ),
            max_length=args.max_length,
        )

        collate_fn = propagation_collate

    else:

        from src.data.dataset import FakeNewsDataset

        dataset = FakeNewsDataset(
            dataframe=dataframe,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        collate_fn = None

    # --------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------

    train_dataset, val_dataset, test_dataset = (
        create_splits(
            dataset,
            args.seed,
        )
    )

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print(
        f"Test: {len(test_dataset)}"
    )

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
    )

    model = model.to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = WeightedCrossEntropy()

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    if args.save_path is not None:

        save_path = args.save_path

    else:

        save_path = (
            f"results/checkpoints/"
            f"{args.dataset}_"
            f"{args.model}_"
            f"seed{args.seed}.pt"
        )

    print(
        f"Checkpoint: {save_path}"
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_path=save_path,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=args.epochs,
    )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    history_path = (
        save_path
        .replace(".pt", "_history.pt")
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
        "\nTraining complete."
    )


if __name__ == "__main__":
    main()