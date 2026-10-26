"""
Experiment results for the UI, read from the repo's own result files
(docs/extended_results.json and docs/revision/neural_paired.json).
Nothing here is hard-coded: if the files change, the page changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
CLASSES = ["non-rumor", "true", "false", "unverified"]
PROTOCOLS = [
    ("random", "Random split (literature protocol)"),
    ("story0.5", "Unseen stories (cosine 0.5)"),
    ("story0.3", "Unseen stories, strict (cosine 0.3)"),
    ("temporal", "Predict the future from the past"),
]
DATASET_LABELS = {"twitter15": "Twitter15", "twitter16": "Twitter16"}


def _stat(d):
    return {"mean": round(d["mean"], 2), "sd": round(d["sd"], 2) if "sd" in d else None}


@lru_cache(maxsize=1)
def load() -> dict:
    ext = json.loads((DOCS / "extended_results.json").read_text(encoding="utf-8"))
    neural_path = DOCS / "revision" / "neural_paired.json"
    neural = json.loads(neural_path.read_text(encoding="utf-8")) if neural_path.exists() else {}

    datasets = {}
    for ds, label in DATASET_LABELS.items():
        if ds not in ext:
            continue
        block = ext[ds]
        protocols = []
        for key, title in PROTOCOLS:
            p = block.get(key)
            if not p:
                continue
            protocols.append({
                "key": key,
                "title": title,
                "accuracy": _stat(p["acc"]),
                "macro_f1": _stat(p["macro_f1"]),
                "n_runs": p["acc"].get("n", len(p["acc"].get("vals", []))),
                "views": {
                    "text": _stat(p["single_views"]["text"]),
                    "spreaders": _stat(p["single_views"]["spreaders"]),
                    "cascade": _stat(p["single_views"]["cascade"]),
                },
                "without": {
                    name.lstrip("-"): _stat(v) for name, v in p["leave_one_out"].items()
                },
                "per_class_f1": dict(zip(CLASSES, p["per_class_precision_recall_f1"][2])),
            })
        rnd = block["random"]
        datasets[ds] = {
            "label": label,
            "protocols": protocols,
            "confusion_random": rnd["confusion_sum"],
        }

    comparison = []
    for ds, protos in neural.items():
        for proto, models in protos.items():
            for model, r in models.items():
                comparison.append({
                    "dataset": ds, "protocol": proto, "model": model,
                    "gestack": r["g"], "neural": r["nn"], "gain": r["delta"],
                    "ci": [r["lo"], r["hi"]], "n_test": r["n"],
                })

    return {
        "classes": CLASSES,
        "model": {
            "name": "GE-Stack",
            "summary": "Graph-enhanced stacked ensemble: three logistic-regression views combined by a meta-learner trained on out-of-fold predictions.",
            "views": [
                {"key": "text", "name": "Text view", "detail": "Word 1-2-gram and character 2-5-gram TF-IDF of the tweet"},
                {"key": "spreaders", "name": "Spreader view", "detail": "Bag of the user ids that shared the tweet"},
                {"key": "cascade", "name": "Cascade view", "detail": "Size, unique users, delays and spread within 5 min, 1 h and 1 day"},
            ],
            "meta": "Logistic regression over the log-probabilities of the three views",
            "trains_on": "about 1,000 labelled tweets per dataset, on CPU in about a minute",
        },
        "datasets": datasets,
        "neural_comparison": comparison,
    }
