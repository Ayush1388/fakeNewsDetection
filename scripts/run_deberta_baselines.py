"""
Neural baselines on EXACTLY the same test sets GE-Stack is scored on,
for a paired comparison in the paper.

For every (dataset, protocol, seed) this script:
  1. builds the test set with the same functions scripts/train_ensemble.py
     uses (random 70/15/15, story-disjoint, temporal),
  2. fine-tunes each neural model with scripts/train.py --test-ids
     (remaining tweets -> 85% train / 15% validation for early stopping),
  3. scores GE-Stack on the same test set (fit on all non-test tweets),
  4. appends the result to results/baselines/baseline_results.json.

Finished runs are skipped on restart, so if the Kaggle session dies
just run the same command again and it continues where it stopped.
Checkpoints are deleted after each run to keep disk usage low.

    python scripts/run_deberta_baselines.py                 # full set (GPU, ~2-3 h)
    python scripts/run_deberta_baselines.py --quick         # 1 seed per protocol
    python scripts/run_deberta_baselines.py --models deberta
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import json
import os
import subprocess
import time
import warnings

import numpy as np
import torch

from src.data.loaders import load_dataset
from src.models.graph_ensemble import GraphEnhancedStack, load_propagation_views
from train_ensemble import (
    create_grouped_splits,
    create_splits,
    create_temporal_split,
    metrics,
    story_groups,
    subset,
)

warnings.filterwarnings("ignore", category=UserWarning)

# (protocol name, split type, group threshold, seeds)
FULL_PROTOCOLS = [
    ("random", "random", None, [0, 1, 2]),
    ("grouped0.5", "grouped", 0.5, [0, 1, 2]),
    ("temporal", "temporal", None, [0]),
]
QUICK_PROTOCOLS = [
    ("random", "random", None, [0]),
    ("grouped0.5", "grouped", 0.5, [0]),
    ("temporal", "temporal", None, [0]),
]


def make_split(labels, tweet_ids, texts, split, threshold, seed, cache):
    """(fit_idx, test_idx) exactly as scripts/train_ensemble.py builds them."""
    if split == "random":
        train_idx, val_idx, test_idx = create_splits(labels, seed)
        fit_idx = np.concatenate([train_idx, val_idx])
    elif split == "grouped":
        key = ("groups", threshold)
        if key not in cache:
            cache[key] = story_groups(texts, threshold=threshold)
        fit_idx, test_idx = create_grouped_splits(labels, cache[key], seed)
    elif split == "temporal":
        fit_idx, test_idx = create_temporal_split(labels, tweet_ids)
    else:
        raise ValueError(split)
    return fit_idx, test_idx


def load_results(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_results(path, results):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def run_key(r):
    return (r["dataset"], r["protocol"], r["seed"], r["model"])


def run_neural(model, dataset, seed, test_ids_path, work_dir, args):
    save_path = work_dir / f"{dataset}_{model}_{test_ids_path.stem}.pt"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "train.py"),
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--test-ids", str(test_ids_path),
        "--save-path", str(save_path),
        "--epochs", str(args.epochs),
        "--max-length", str(args.max_length),
        "--model-name", args.model_name,
    ]
    print("  $", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    minutes = (time.time() - t0) / 60

    history_path = Path(str(save_path).replace(".pt", "_history.pt"))
    result = None

    if proc.returncode == 0 and history_path.exists():
        history = torch.load(history_path, map_location="cpu", weights_only=False)
        result = {
            "accuracy": float(history["test_accuracy"]),
            "macro_f1": float(history["test_f1"]),
            "best_val_f1": float(history["best_val_f1"]),
            "epochs_run": len(history["val_f1"]),
            "minutes": round(minutes, 1),
        }

    # Free disk: checkpoints are ~1-2 GB each.
    for f in work_dir.glob(f"{save_path.stem}*"):
        f.unlink(missing_ok=True)

    return result


def summarize(results):
    print("\n" + "=" * 78)
    print("SUMMARY  (mean test accuracy / macro-F1 over seeds, identical test sets)")
    print("=" * 78)
    rows = {}
    for r in results:
        if r.get("accuracy") is None:
            continue
        rows.setdefault((r["dataset"], r["protocol"], r["model"]), []).append(r)

    for (dataset, protocol, model), rs in sorted(rows.items()):
        acc = np.mean([r["accuracy"] for r in rs])
        f1 = np.mean([r["macro_f1"] for r in rs])
        sd = np.std([r["accuracy"] for r in rs])
        print(
            f"{dataset:10s} {protocol:11s} {model:10s} "
            f"acc={acc * 100:6.2f} +/- {sd * 100:4.2f}  macro-F1={f1 * 100:6.2f}  "
            f"(n={len(rs)})"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="twitter15,twitter16")
    parser.add_argument(
        "--models", default="deberta,tegfnd",
        help="Neural models from scripts/train.py to run (deberta, tegfnd, propagation).",
    )
    parser.add_argument("--quick", action="store_true", help="One seed per protocol.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--max-length", type=int, default=64,
        help="Tweets are <= 140 chars / 29 words, so 64 tokens loses nothing.",
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--out-dir", default="results/baselines")
    parser.add_argument("--no-gestack", action="store_true",
                        help="Skip re-scoring GE-Stack on the same splits.")
    args = parser.parse_args(argv)

    out_dir = PROJECT_ROOT / args.out_dir
    split_dir = out_dir / "splits"
    work_dir = out_dir / "work"
    for d in (out_dir, split_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "baseline_results.json"
    results = load_results(results_path)
    done = {run_key(r) for r in results if r.get("accuracy") is not None}

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.no_gestack:
        models = ["gestack"] + models
    protocols = QUICK_PROTOCOLS if args.quick else FULL_PROTOCOLS

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Models: {models}")
    print(f"Results file: {results_path}  ({len(done)} runs already done)")

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        df = load_dataset(dataset, str(PROJECT_ROOT / "data" / dataset))
        labels = df["label"].to_numpy()
        tweet_ids = df["id"].astype(str).to_numpy()
        texts = df["text"].to_numpy(dtype=object)
        cache = {}
        views = None

        for protocol, split, threshold, seeds in protocols:
            for seed in seeds:
                fit_idx, test_idx = make_split(labels, tweet_ids, texts, split, threshold, seed, cache)
                test_ids_path = split_dir / f"{dataset}_{protocol}_seed{seed}_test_ids.json"
                with open(test_ids_path, "w") as f:
                    json.dump(sorted(tweet_ids[test_idx].tolist()), f)

                for model in models:
                    key = (dataset, protocol, seed, model)
                    if key in done:
                        continue

                    print(f"\n### {dataset} | {protocol} | seed {seed} | {model} "
                          f"| test={len(test_idx)}", flush=True)

                    if model == "gestack":
                        if views is None:
                            spreaders, cascade = load_propagation_views(
                                PROJECT_ROOT / "data" / dataset / "tree", tweet_ids.tolist(),
                            )
                            views = {"text": texts, "spreaders": spreaders, "cascade": cascade}
                        t0 = time.time()
                        stack = GraphEnhancedStack(seed=seed).fit(subset(views, fit_idx), labels[fit_idx])
                        pred = stack.predict(subset(views, test_idx))
                        m = metrics(labels[test_idx], pred)
                        m["minutes"] = round((time.time() - t0) / 60, 1)
                    else:
                        m = run_neural(model, dataset, seed, test_ids_path, work_dir, args)
                        if m is None:
                            print(f"  !! {model} run FAILED -- skipped (see log above)")
                            continue

                    r = {"dataset": dataset, "protocol": protocol, "seed": seed,
                         "model": model, "n_test": int(len(test_idx)), **m}
                    results.append(r)
                    done.add(key)
                    save_results(results_path, results)
                    print(f"  -> acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}", flush=True)

    summarize(results)
    print(f"\nAll results: {results_path}")


if __name__ == "__main__":
    main()
