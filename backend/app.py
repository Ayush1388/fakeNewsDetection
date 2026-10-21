"""
FastAPI backend for GE-Stack rumour detection.

    uvicorn backend.app:app --reload
    -> http://localhost:8000        (frontend)
    -> http://localhost:8000/docs   (API docs)
"""

from __future__ import annotations

import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import model_store as store

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_NODES = 100_000

app = FastAPI(title="GE-Stack Rumour Detection API", version="1.0.0")


class Node(BaseModel):
    user: str = Field(min_length=1, max_length=64)
    delay_minutes: float = Field(ge=0)


class PredictRequest(BaseModel):
    dataset: str = "twitter15"
    text: str | None = Field(default=None, max_length=2000)
    propagation: list[Node] | None = Field(default=None, max_length=MAX_NODES)
    sample_id: str | None = None


def _check_dataset(name: str):
    if name not in store.DATASETS:
        raise HTTPException(404, f"Unknown dataset '{name}'. Use one of {list(store.DATASETS)}.")


@app.get("/api/health")
def health():
    return {"status": "ok", "datasets": list(store.DATASETS), "classes": store.CLASS_NAMES}


@app.get("/api/samples")
def samples(
    dataset: str = "twitter15",
    n: int = Query(8, ge=1, le=50),
    seed: int | None = None,
):
    _check_dataset(dataset)
    frame = store.get_bundle(dataset).frame
    idx = random.Random(seed).sample(range(len(frame)), min(n, len(frame)))
    return [
        {
            "id": str(frame.iloc[i]["id"]),
            "text": frame.iloc[i]["text"],
            "true_label": store.CLASS_NAMES[int(frame.iloc[i]["label"])],
        }
        for i in idx
    ]


@app.post("/api/predict")
def predict(req: PredictRequest):
    _check_dataset(req.dataset)
    text, nodes, in_training = req.text, None, False

    if req.sample_id:
        frame = store.get_bundle(req.dataset).frame
        row = frame[frame["id"] == req.sample_id]
        if row.empty:
            raise HTTPException(404, "Unknown sample_id for this dataset.")
        text = row.iloc[0]["text"]
        nodes = store.sample_nodes(req.dataset, req.sample_id)
        in_training = True

    if req.propagation:
        nodes = [(n.user, n.delay_minutes) for n in req.propagation]

    if not text or not text.strip():
        raise HTTPException(422, "Provide 'text' or a valid 'sample_id'.")

    result = store.predict(req.dataset, text.strip(), nodes)
    result["propagation_nodes"] = len(nodes) if nodes else 0
    # Dataset samples were part of the training data, so their scores are optimistic.
    result["in_training_data"] = in_training
    return result


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
