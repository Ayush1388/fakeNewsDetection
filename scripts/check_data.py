import argparse
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.data.loaders import load_dataset


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "twitter15",
            "twitter16",
            "politifact",
        ],
    )

    parser.add_argument(
        "--path",
        required=True,
    )

    args = parser.parse_args()

    df = load_dataset(
        args.dataset,
        args.path,
    )

    print("\n===== DATASET CHECK =====")

    print(
        "Dataset:",
        args.dataset,
    )

    print(
        "Samples:",
        len(df),
    )

    print(
        "Missing text:",
        df["text"].isna().sum(),
    )

    print(
        "Duplicate IDs:",
        df["id"].duplicated().sum(),
    )

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nData check completed.")


if __name__ == "__main__":
    main()