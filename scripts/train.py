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
from src.data.dataset import FakeNewsDataset
from src.models.tegfnd import TEGFND
from src.models.deberta_baseline import DeBERTaBaseline
from src.training.losses import WeightedCrossEntropy
from src.training.trainer import Trainer


TRANSFORMER_NAME = "microsoft/deberta-v3-base"


DATASETS = {
    "twitter15": {
        "path": "data/twitter15",
        "classes": [
            "non-rumor",
            "true",
            "false",
            "unverified",
        ],
    },
    "twitter16": {
        "path": "data/twitter16",
        "classes": [
            "non-rumor",
            "true",
            "false",
            "unverified",
        ],
    },
    "politifact": {
        "path": "data/politifact/politifact_factcheck_data.json",
        "classes": [
            "pants-fire",
            "false",
            "mostly-false",
            "half-true",
            "mostly-true",
            "true",
        ],
    },
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_splits(df, seed):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
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


def build_model(
    model_type,
    num_classes,
):

    if model_type == "deberta":

        return DeBERTaBaseline(
            num_classes=num_classes,
            model_name=TRANSFORMER_NAME,
        )

    return TEGFND(
        num_classes=num_classes,
        model_name=TRANSFORMER_NAME,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    parser.add_argument(
        "--model",
        default="tegfnd",
        choices=[
            "tegfnd",
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
        default=8,
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

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")

    config = DATASETS[
        args.dataset
    ]

    dataset_path = (
        PROJECT_ROOT
        / config["path"]
    )

    df = load_dataset(
        args.dataset,
        dataset_path,
    )

    print(
        f"Loaded {len(df)} samples."
    )

    train_df, val_df, test_df = build_splits(
        df,
        args.seed,
    )

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_NAME
    )

    train_dataset = FakeNewsDataset(
        train_df,
        tokenizer,
        max_length=args.max_length,
    )

    val_dataset = FakeNewsDataset(
        val_df,
        tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        args.model,
        len(config["classes"]),
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    criterion = WeightedCrossEntropy()

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "checkpoints"
        / f"{args.dataset}_{args.model}_best.pt"
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_path=checkpoint_path,
    )

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=args.epochs,
    )

    print("\nTraining complete.")

    print(
        f"Best validation F1: "
        f"{trainer.best_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()