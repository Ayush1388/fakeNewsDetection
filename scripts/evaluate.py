import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from src.data.loaders import load_dataset
from src.data.dataset import FakeNewsDataset

from src.models.tegfnd import TEGFND
from src.models.deberta_baseline import DeBERTaBaseline

from src.evaluation.evaluate import evaluate_model


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


def build_test_split(
    df,
    seed,
):

    _, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
    )

    _, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return test_df.reset_index(
        drop=True
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
        "--checkpoint",
        required=True,
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
        "--batch-size",
        type=int,
        default=8,
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Model: {args.model}"
    )

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

    test_df = build_test_split(
        df,
        args.seed,
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_NAME
    )

    test_dataset = FakeNewsDataset(
        test_df,
        tokenizer,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        args.model,
        len(config["classes"]),
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_absolute():

        checkpoint_path = (
            PROJECT_ROOT
            / checkpoint_path
        )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / f"{args.dataset}_{args.model}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        class_names=config["classes"],
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()