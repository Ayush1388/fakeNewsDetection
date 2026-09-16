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
from sklearn.metrics import f1_score

from src.data.loaders import load_dataset
from src.data.graph_dataset import (
    PropagationDataset,
    propagation_collate,
)
from src.models.rumor_model import (
    PropagationRumorModel,
)


TRANSFORMER_NAME = (
    "microsoft/deberta-v3-base"
)


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
}


def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_splits(
    dataframe,
    seed,
):

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


def move_graph_batch_to_device(
    batch,
    device,
):

    node_features = [
        x.to(device)
        for x in batch["node_features"]
    ]

    edge_index = [
        x.to(device)
        for x in batch["edge_index"]
    ]

    delays = [
        x.to(device)
        for x in batch["delays"]
    ]

    return (
        node_features,
        edge_index,
        delays,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    predictions = []
    labels = []

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        linguistic_features = batch[
            "linguistic_features"
        ].to(device)

        target = batch[
            "labels"
        ].to(device)

        (
            node_features,
            edge_index,
            delays,
        ) = move_graph_batch_to_device(
            batch,
            device,
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_features=linguistic_features,
                node_features=node_features,
                edge_index=edge_index,
                delays=delays,
            )

            loss = criterion(
                output["logits"],
                target,
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        total_loss += loss.item()

        prediction = (
            output["logits"]
            .argmax(dim=-1)
        )

        predictions.extend(
            prediction.detach()
            .cpu()
            .numpy()
        )

        labels.extend(
            target.detach()
            .cpu()
            .numpy()
        )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        total_loss
        / max(len(loader), 1),
        macro_f1,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
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

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = DATASETS[
        args.dataset
    ]

    print("=" * 60)
    print("PROPAGATION RUMOR MODEL")
    print("=" * 60)
    print(
        f"Device: {device}"
    )
    print(
        f"Dataset: {args.dataset}"
    )

    dataset_path = (
        PROJECT_ROOT
        / config["path"]
    )

    dataframe = load_dataset(
        args.dataset,
        dataset_path,
    )

    print(
        f"Total samples: {len(dataframe)}"
    )

    train_df, val_df, test_df = (
        create_splits(
            dataframe,
            args.seed,
        )
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

    tokenizer = (
        AutoTokenizer.from_pretrained(
            TRANSFORMER_NAME
        )
    )

    tree_root = (
        dataset_path / "tree"
    )

    train_dataset = PropagationDataset(
        train_df,
        tokenizer,
        tree_root,
        args.max_length,
    )

    val_dataset = PropagationDataset(
        val_df,
        tokenizer,
        tree_root,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=propagation_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=propagation_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = PropagationRumorModel(
        num_classes=len(
            config["classes"]
        ),
        model_name=TRANSFORMER_NAME,
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "checkpoints"
        / f"{args.dataset}_propagation_best.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_f1 = -1.0

    print("\nStarting training...\n")

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss, train_f1 = (
            run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                True,
            )
        )

        val_loss, val_f1 = (
            run_epoch(
                model,
                val_loader,
                optimizer,
                criterion,
                device,
                False,
            )
        )

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "val_f1":
                        val_f1,

                    "seed":
                        args.seed,
                },
                checkpoint_path,
            )

            print(
                f"Saved best model -> "
                f"{checkpoint_path}"
            )


if __name__ == "__main__":
    main()