import re
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
STATIC = Path(__file__).resolve().parents[1] / "backend" / "static"


def test_index_served_and_assets_exist():
    r = client.get("/")
    assert r.status_code == 200 and "Rumour Desk" in r.text
    for path in re.findall(r'(?:href|src)="(/static/[^"]+)"', r.text):
        assert client.get(path).status_code == 200, path


def test_results_section_is_in_page_and_script_calls_api():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    assert 'id="results"' in html
    assert "/api/results" in js and "/api/predict" in js


def test_page_never_uses_innerhtml_with_model_or_user_text():
    assert "innerHTML" not in (STATIC / "app.js").read_text()
