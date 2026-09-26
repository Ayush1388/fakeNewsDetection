"""
Train and evaluate GE-Stack (Graph-Enhanced Stacked Ensemble) on
Twitter15 / Twitter16.

    python scripts/train_ensemble.py --dataset twitter15
    python scripts/train_ensemble.py --dataset twitter16
    python scripts/train_ensemble.py --dataset both

Protocol (same split function and seeds as scripts/train.py):
    * stratified 70 / 15 / 15 train / val / test split per seed
    * the model is fit on train + val (its meta-learner is trained on
      5-fold out-of-fold predictions inside that data, so no separate
      validation set is needed for model selection)
    * the 15% test split is never touched until the final prediction
    * repeated over several seeds (default 5) and reported as
      mean +/- std, because one ~224-tweet test split is noisy
      (+/- 2-3 points from split luck alone)

Runs on CPU in about a minute per dataset; no GPU needed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.data.loaders import TWITTER_LABELS, load_dataset
from src.models.graph_ensemble import GraphEnhancedStack, load_propagation_views

warnings.filterwarnings("ignore", category=UserWarning)

CLASS_NAMES = [
    name for name, _ in sorted(TWITTER_LABELS.items(), key=lambda kv: kv[1])
]


def create_splits(labels, seed):
    """Identical to scripts/train.py create_splits, on row indices."""
    idx = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=seed, stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=labels[temp_idx],
    )
    return train_idx, val_idx, test_idx


def story_groups(texts, threshold=0.5):
    """
    Group near-duplicate tweets (word TF-IDF cosine >= threshold,
    transitively) into "story" clusters. Twitter15/16 contain many
    tweets about the same story with the same label; a random split
    puts siblings on both sides, which inflates every model's score
    (the literature's numbers included).
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer().fit_transform(texts)
    sim = (X @ X.T).tocsr()
    sim.data = (sim.data >= threshold).astype(np.int8)
    sim.eliminate_zeros()
    _, groups = connected_components(sim, directed=False)
    return groups


def create_grouped_splits(labels, groups, seed):
    """
    Story-disjoint split: no story cluster appears in both the fit
    data and the test data. One of 7 stratified group folds (~14%)
    is the test set.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    fit_idx, test_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    return fit_idx, test_idx


def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1),
        "macro_precision": float(p),
        "macro_recall": float(r),
    }


def subset(views, idx):
    return {k: v[idx] for k, v in views.items()}


def run_dataset(dataset, seeds, save_dir, split="random"):
    print("=" * 64)
    print(f"Dataset: {dataset}")

    df = load_dataset(dataset, f"data/{dataset}")
    labels = df["label"].to_numpy()

    t0 = time.time()
    spreaders, cascade = load_propagation_views(
        f"data/{dataset}/tree", df["id"].tolist(),
    )
    print(f"Loaded {len(df)} tweets + propagation trees in {time.time() - t0:.1f}s")

    views = {
        "text": df["text"].to_numpy(dtype=object),
        "spreaders": spreaders,
        "cascade": cascade,
    }

    per_seed = []
    ablation = {}

    if split == "grouped":
        groups = story_groups(views["text"])
        print(
            f"Split: GROUPED (story-disjoint) -- {len(df)} tweets in "
            f"{len(np.unique(groups))} story clusters"
        )
    else:
        print("Split: RANDOM stratified 70/15/15 (the protocol the base papers use)")

    for i, seed in enumerate(seeds):
        if split == "grouped":
            fit_idx, test_idx = create_grouped_splits(labels, groups, seed)
        else:
            train_idx, val_idx, test_idx = create_splits(labels, seed)
            fit_idx = np.concatenate([train_idx, val_idx])

        model = GraphEnhancedStack(seed=seed).fit(subset(views, fit_idx), labels[fit_idx])

        test_views = subset(views, test_idx)
        y_test = labels[test_idx]
        y_pred = model.predict(test_views)

        m = metrics(y_test, y_pred)
        m["seed"] = seed
        per_seed.append(m)

        # Per-view ablation: each base view on its own, same test split.
        for name, proba in model.view_probabilities(test_views).items():
            acc = accuracy_score(y_test, model.classes_[proba.argmax(axis=1)])
            ablation.setdefault(name, []).append(float(acc))

        print(
            f"seed {seed:>4}: test acc={m['accuracy']:.4f}  "
            f"macro-F1={m['macro_f1']:.4f}  "
            f"(train+val={len(fit_idx)}, test={len(test_idx)})"
        )

        if i == 0:
            first = {
                "seed": seed,
                "report": classification_report(
                    y_test, y_pred, labels=range(len(CLASS_NAMES)),
                    target_names=CLASS_NAMES, digits=4, zero_division=0,
                ),
                "confusion": confusion_matrix(
                    y_test, y_pred, labels=range(len(CLASS_NAMES)),
                ).tolist(),
            }
            save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, save_dir / f"{dataset}_gestack_{split}_seed{seed}.joblib")

    accs = np.array([m["accuracy"] for m in per_seed])
    f1s = np.array([m["macro_f1"] for m in per_seed])

    print("-" * 64)
    print(f"Per-class report (seed {first['seed']}):")
    print(first["report"])
    print(f"Confusion matrix (rows=true, cols=pred, order={CLASS_NAMES}):")
    for row in first["confusion"]:
        print("   ", row)

    print("-" * 64)
    print("Ablation - each view alone (mean test accuracy over seeds):")
    for name, vals in ablation.items():
        print(f"    {name:10s} {np.mean(vals):.4f}")
    print(f"    {'GE-Stack':10s} {accs.mean():.4f}   <- all views stacked")

    print("-" * 64)
    print(
        f"{dataset} GE-Stack ({split} split) over {len(seeds)} seeds: "
        f"TEST ACCURACY = {accs.mean():.4f} +/- {accs.std():.4f}   "
        f"MACRO-F1 = {f1s.mean():.4f} +/- {f1s.std():.4f}"
    )

    summary = {
        "dataset": dataset,
        "split": split,
        "seeds": seeds,
        "per_seed": per_seed,
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "mean_macro_f1": float(f1s.mean()),
        "std_macro_f1": float(f1s.std()),
        "ablation_mean_accuracy": {k: float(np.mean(v)) for k, v in ablation.items()},
        "first_seed_report": first["report"],
        "first_seed_confusion": first["confusion"],
    }

    with open(save_dir / f"{dataset}_gestack_{split}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="twitter15",
        choices=["twitter15", "twitter16", "both"],
    )
    parser.add_argument(
        "--seeds", default="42,1,2,3,4",
        help="Comma-separated split seeds. The first one's model is saved.",
    )
    parser.add_argument("--save-dir", default="results/ensemble")
    parser.add_argument(
        "--split", default="random", choices=["random", "grouped"],
        help=(
            "random = stratified 70/15/15, same protocol as the base "
            "papers (comparable to their numbers). grouped = story-"
            "disjoint split, near-duplicate tweets never cross train/"
            "test (harder, closer to performance on unseen stories)."
        ),
    )
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = ["twitter15", "twitter16"] if args.dataset == "both" else [args.dataset]
    save_dir = Path(args.save_dir)

    summaries = [run_dataset(d, seeds, save_dir, args.split) for d in datasets]

    print("=" * 64)
    for s in summaries:
        print(
            f"FINAL {s['dataset']} ({s['split']} split): "
            f"accuracy {s['mean_accuracy'] * 100:.2f}% "
            f"+/- {s['std_accuracy'] * 100:.2f}, "
            f"macro-F1 {s['mean_macro_f1'] * 100:.2f}%"
        )
    print(f"Results + saved model: {save_dir}/")
    return summaries


if __name__ == "__main__":
    main()
