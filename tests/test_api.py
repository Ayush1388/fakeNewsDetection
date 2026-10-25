import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and "twitter15" in r.json()["datasets"]


def test_text_only():
    r = client.post("/api/predict", json={"text": "breaking: celebrity dead, sources say"})
    assert r.status_code == 200
    j = r.json()
    assert j["views_used"] == ["text"] and abs(sum(j["probabilities"].values()) - 1) < 1e-6


def test_with_propagation():
    nodes = [{"user": f"u{i}", "delay_minutes": i * 2.0} for i in range(20)]
    j = client.post("/api/predict", json={"text": "some claim", "propagation": nodes}).json()
    assert j["views_used"] == ["text", "spreaders", "cascade"] and j["propagation_nodes"] == 20


def test_sample_roundtrip():
    s = client.get("/api/samples?dataset=twitter16&n=2&seed=1").json()
    r = client.post("/api/predict", json={"dataset": "twitter16", "sample_id": s[0]["id"]})
    assert r.status_code == 200 and r.json()["in_training_data"]


@pytest.mark.parametrize("body,code", [
    ({}, 422), ({"text": "  "}, 422), ({"dataset": "x", "text": "a"}, 404),
    ({"text": "a", "propagation": [{"user": "u", "delay_minutes": -1}]}, 422),
])
def test_errors(body, code):
    assert client.post("/api/predict", json=body).status_code == code


def test_spread_summary_returned_with_propagation():
    nodes = [{"user": f"u{i}", "delay_minutes": float(i)} for i in range(10)]
    j = client.post("/api/predict", json={"text": "x claim", "propagation": nodes}).json()
    assert j["spread"]["nodes"] == 10 and j["spread"]["within_5min"] == 0.6
    assert client.post("/api/predict", json={"text": "x claim"}).json()["spread"] is None
