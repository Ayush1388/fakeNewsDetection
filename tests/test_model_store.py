import numpy as np
import pytest

from backend import model_store as store
from src.models.graph_ensemble import (
    GraphEnhancedStack,
    propagation_views_from_nodes,
    read_propagation_views,
)


def test_nodes_parser_matches_file_parser():
    ds = "twitter15"
    tid = next(iter(store.get_bundle(ds).frame["id"]))
    nodes = store.sample_nodes(ds, tid)
    s_file, c_file = read_propagation_views(store.DATA_DIR / ds / "tree" / f"{tid}.txt")
    s_mem, c_mem = propagation_views_from_nodes([u for u, _ in nodes], [d for _, d in nodes])
    assert s_file == s_mem and np.allclose(c_file, c_mem)


def test_spread_summary():
    s = store._spread_summary([("a", 0), ("b", 4), ("a", 90), ("c", 2000)])
    assert s["nodes"] == 4 and s["unique_users"] == 3
    assert s["within_5min"] == 0.5 and s["within_60min"] == 0.5 and s["within_1day"] == 0.75
    assert store._spread_summary(None) is None


def test_predict_probabilities_sum_to_one_and_label_is_argmax():
    r = store.predict("twitter16", "a viral claim spreads quickly")
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert r["label"] == max(r["probabilities"], key=r["probabilities"].get)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        store.get_bundle("nope")


def test_text_only_model_beats_chance_on_held_out_data():
    """Sanity check of the ML pipeline: fit on 80%, test on the unseen 20%."""
    frame = store.get_bundle("twitter15").frame
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(frame))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    texts = frame["text"].to_numpy(dtype=object)
    y = frame["label"].to_numpy()
    m = GraphEnhancedStack(use_views=("text",)).fit({"text": texts[tr]}, y[tr])
    acc = (m.predict({"text": texts[te]}) == y[te]).mean()
    assert acc > 0.6  # chance is about 0.25
