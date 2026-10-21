"""
Train / load GE-Stack models for the API.

Two models per dataset are kept:
    full       text + spreaders + cascade (used when a propagation tree is given)
    text_only  text view only (used otherwise; the README shows a stack that
               still trusts an empty spreader view does worse than text only)

Models are trained on ALL labelled tweets of the dataset and cached with
joblib under backend/artifacts/ (git-ignored).
"""

from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np

from src.data.loaders import TWITTER_LABELS, load_dataset
from src.models.graph_ensemble import (
    GraphEnhancedStack,
    load_propagation_views,
    propagation_views_from_nodes,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DATASETS = ("twitter15", "twitter16")
CLASS_NAMES = [n for n, _ in sorted(TWITTER_LABELS.items(), key=lambda kv: kv[1])]


class Bundle:
    def __init__(self, dataset: str, full, text_only, frame):
        self.dataset = dataset
        self.full = full
        self.text_only = text_only
        self.frame = frame  # id, text, label


_cache: dict[str, Bundle] = {}
_lock = threading.Lock()


def _train(dataset: str) -> Bundle:
    frame = load_dataset(dataset, DATA_DIR / dataset)
    y = frame["label"].to_numpy()
    spreaders, cascade = load_propagation_views(
        DATA_DIR / dataset / "tree", frame["id"]
    )
    views = {
        "text": frame["text"].to_numpy(dtype=object),
        "spreaders": spreaders,
        "cascade": cascade,
    }
    full = GraphEnhancedStack().fit(views, y)
    text_only = GraphEnhancedStack(use_views=("text",)).fit(views, y)
    return Bundle(dataset, full, text_only, frame)


def get_bundle(dataset: str) -> Bundle:
    if dataset not in DATASETS:
        raise KeyError(dataset)
    with _lock:
        if dataset in _cache:
            return _cache[dataset]
        path = ARTIFACT_DIR / f"{dataset}.joblib"
        bundle = None
        if path.exists():
            try:
                bundle = joblib.load(path)
            except Exception:
                bundle = None  # stale / incompatible artifact: retrain
        if bundle is None:
            bundle = _train(dataset)
            ARTIFACT_DIR.mkdir(exist_ok=True)
            joblib.dump(bundle, path)
        _cache[dataset] = bundle
        return bundle


def predict(dataset: str, text: str, nodes=None) -> dict:
    """nodes: optional list of (user_id, delay_minutes) for the cascade."""
    bundle = get_bundle(dataset)
    texts = np.asarray([text], dtype=object)

    if nodes:
        users = [str(u) for u, _ in nodes]
        delays = [float(d) for _, d in nodes]
        s, c = propagation_views_from_nodes(users, delays)
        views = {
            "text": texts,
            "spreaders": np.asarray([s], dtype=object),
            "cascade": c[None, :],
        }
        model = bundle.full
    else:
        views = {"text": texts}
        model = bundle.text_only

    proba = model.predict_proba(views)[0]
    per_view = {
        name: {CLASS_NAMES[int(k)]: float(p) for k, p in zip(model.classes_, pv[0])}
        for name, pv in model.view_probabilities(views).items()
    }
    probs = {CLASS_NAMES[int(k)]: float(p) for k, p in zip(model.classes_, proba)}
    label = max(probs, key=probs.get)
    return {
        "dataset": dataset,
        "label": label,
        "confidence": probs[label],
        "probabilities": probs,
        "views_used": list(model.use_views),
        "per_view": per_view,
    }


def sample_nodes(dataset: str, tweet_id: str):
    """(user, delay) list parsed from a dataset tree file, or None."""
    import re

    path = DATA_DIR / dataset / "tree" / f"{tweet_id}.txt"
    if not path.exists():
        return None
    node = re.compile(r"\[\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*\]")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "->" not in line:
            continue
        m = node.search(line.split("->", 1)[1])
        if m:
            try:
                out.append((m.group(1), float(m.group(3))))
            except ValueError:
                pass
    return out or None
