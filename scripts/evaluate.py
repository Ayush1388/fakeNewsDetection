import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from src.data.loaders import load_dataset
from src.data.dataset import FakeNewsDataset
from src.models.tegfnd import TEGFND
from src.evaluation.evaluate import evaluate_model


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
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="microsoft/deberta-v3-base",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = DATASETS[
        args.dataset
    ]

    df = load_dataset(
        args.dataset,
        config["path"],
    )

    _, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
        stratify=df["label"],
    )

    _, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    test_dataset = FakeNewsDataset(
        test_df,
        tokenizer,
        args.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TEGFND(
        num_classes=len(
            config["classes"]
        ),
        model_name=args.model,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        class_names=config["classes"],
        output_dir=(
            f"results/metrics/"
            f"{args.dataset}"
        ),
    )


if __name__ == "__main__":
    main()