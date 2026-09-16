from pathlib import Path
import json

import pandas as pd


TWITTER_LABELS = {
    "non-rumor": 0,
    "true": 1,
    "false": 2,
    "unverified": 3,
}


POLITIFACT_LABELS = {
    "pants-fire": 0,
    "false": 1,
    "mostly-false": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}


def load_twitter_dataset(data_dir):
    data_dir = Path(data_dir)

    label_file = data_dir / "label.txt"
    source_file = data_dir / "source_tweets.txt"

    if not label_file.exists():
        raise FileNotFoundError(f"Missing: {label_file}")

    if not source_file.exists():
        raise FileNotFoundError(f"Missing: {source_file}")

    labels = {}

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            label, tweet_id = line.split(":", 1)

            label = label.strip().lower()
            tweet_id = tweet_id.strip()

            if label not in TWITTER_LABELS:
                continue

            labels[tweet_id] = TWITTER_LABELS[label]

    tweets = {}

    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if "\t" not in line:
                continue

            tweet_id, text = line.split("\t", 1)

            tweets[tweet_id.strip()] = text.strip()

    rows = []

    for tweet_id, label in labels.items():

        if tweet_id not in tweets:
            continue

        rows.append(
            {
                "id": tweet_id,
                "text": tweets[tweet_id],
                "label": label,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No valid samples found in {data_dir}"
        )

    return df


def load_politifact(data_file):
    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(f"Missing: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        raw = json.loads(content)

        if isinstance(raw, dict):
            if "data" in raw:
                raw = raw["data"]
            else:
                raw = list(raw.values())

    except json.JSONDecodeError:
        raw = []

        for line in content.splitlines():
            line = line.strip()

            if line:
                raw.append(json.loads(line))

    rows = []

    for item in raw:

        text = (
            item.get("statement")
            or item.get("text")
            or item.get("claim")
        )

        label = (
            item.get("verdict")
            or item.get("label")
            or item.get("rating")
        )

        if text is None or label is None:
            continue

        label = str(label).strip().lower()

        if label not in POLITIFACT_LABELS:
            continue

        rows.append(
            {
                "id": str(item.get("id", len(rows))),
                "text": str(text),
                "label": POLITIFACT_LABELS[label],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No valid PolitiFact samples found in {data_file}"
        )

    return df


def load_dataset(name, path):
    name = name.lower()

    if name in {"twitter15", "twitter16"}:
        return load_twitter_dataset(path)

    if name == "politifact":
        return load_politifact(path)

    raise ValueError(
        f"Unknown dataset: {name}"
    )