from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_results_shape_and_values():
    r = client.get("/api/results")
    assert r.status_code == 200
    j = r.json()
    assert j["model"]["name"] == "GE-Stack" and len(j["model"]["views"]) == 3
    assert set(j["datasets"]) == {"twitter15", "twitter16"}
    for ds in j["datasets"].values():
        keys = {p["key"] for p in ds["protocols"]}
        assert {"random", "story0.5", "temporal"} <= keys
        for p in ds["protocols"]:
            assert 0 < p["accuracy"]["mean"] <= 100
            assert set(p["without"]) == {"text", "spreaders", "cascade"}
            assert set(p["per_class_f1"]) == set(j["classes"])
        assert len(ds["confusion_random"]) == 4


def test_results_match_readme_headline():
    j = client.get("/api/results").json()
    rnd = {p["key"]: p for p in j["datasets"]["twitter15"]["protocols"]}["random"]
    assert abs(rnd["accuracy"]["mean"] - 91.79) < 0.01


def test_neural_comparison_gestack_wins():
    cmp = client.get("/api/results").json()["neural_comparison"]
    assert len(cmp) == 12
    assert all(r["gain"] > 10 for r in cmp)
