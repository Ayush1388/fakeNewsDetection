import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.loaders import load_dataset
from src.data.dataset import FakeNewsDataset

from src.models.tegfnd import TEGFND

from src.training.seed import set_seed
from src.training.losses import WeightedCrossEntropy
from src.training.trainer import Trainer


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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    parser.add_argument(
        "--model",
        default="microsoft/deberta-v3-base",
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
        default=192,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    dataset_config = DATASETS[
        args.dataset
    ]

    df = load_dataset(
        args.dataset,
        dataset_config["path"],
    )

    print(
        f"Loaded {len(df)} samples."
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    train_dataset = FakeNewsDataset(
        train_df,
        tokenizer,
        args.max_length,
    )

    val_dataset = FakeNewsDataset(
        val_df,
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    classes = dataset_config["classes"]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=sorted(
            train_df["label"].unique()
        ),
        y=train_df["label"],
    )

    weights = torch.ones(
        len(classes),
        dtype=torch.float32,
    )

    for class_id, weight in zip(
        sorted(train_df["label"].unique()),
        class_weights,
    ):
        weights[class_id] = weight

    weights = weights.to(device)

    model = TEGFND(
        num_classes=len(classes),
        model_name=args.model,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    criterion = WeightedCrossEntropy(
        class_weights=weights
    )

    checkpoint = (
        f"results/checkpoints/"
        f"{args.dataset}_tegfnd.pt"
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_path=checkpoint,
    )

    trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=args.epochs,
    )

    print(
        "\nTraining completed."
    )

    print(
        f"Best validation Macro-F1: "
        f"{trainer.best_f1:.4f}"
    )

    print(
        f"Checkpoint: {checkpoint}"
    )


if __name__ == "__main__":
    main()