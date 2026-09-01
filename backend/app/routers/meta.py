"""Metadata endpoints: health, models, explanation methods, examples, history, results."""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas import (
    ExampleNote,
    HealthResponse,
    HistoryEntry,
    MethodInfo,
    ModelInfo,
)
from backend.app.services.inference import InferenceService
from medxplain.explain import METHOD_DESCRIPTIONS
from medxplain.utils import read_json

router = APIRouter(tags=["metadata"])


class History:
    """In-memory ring buffer of recent predictions.

    Deliberately not persisted: notes pasted into the demo are potentially
    sensitive, and writing them to disk would turn a stateless demo into a
    system with a data-retention question attached. The frontend keeps its own
    session log for the same reason.
    """

    def __init__(self, maxlen: int = 50):
        self._items: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, entry: dict) -> None:
        with self._lock:
            self._items.appendleft(entry)

    def list(self) -> list[dict]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


@router.get("/health", response_model=HealthResponse, summary="Liveness and model status")
def health(request: Request) -> HealthResponse:
    app = request.app
    settings = app.state.settings
    registry = app.state.registry
    return HealthResponse(
        status="ok",
        version=settings.version,
        device=settings.device,
        models_loaded=registry.loaded_keys(),
        models_available=registry.available(),
        default_model=settings.default_model,
        default_method=settings.default_method,
    )


@router.get("/models", response_model=list[ModelInfo], summary="Trained models available to serve")
def models(request: Request) -> list[ModelInfo]:
    registry = request.app.state.registry
    available = registry.available()
    if not available:
        return [
            ModelInfo(
                key=request.app.state.settings.default_model,
                display_name="(not trained yet)",
                hf_id="",
                available=False,
                note="Run python scripts/02_train_transformers.py to produce a checkpoint.",
            )
        ]
    out = []
    for key in available:
        info = registry.describe(key)
        out.append(
            ModelInfo(
                key=key,
                display_name=info.get("display_name", key),
                hf_id=info.get("hf_id", ""),
                available=bool(info.get("available")),
                outcome=info.get("outcome"),
                test_auroc=info.get("test_auroc"),
                test_auprc=info.get("test_auprc"),
                threshold=info.get("threshold"),
                max_length=info.get("max_length"),
                seed=info.get("seed"),
            )
        )
    return out


@router.get("/methods", response_model=list[MethodInfo], summary="Explanation methods")
def methods() -> list[MethodInfo]:
    typical = {
        "attention_rollout": 400,
        "attention_raw": 350,
        "attention_grad": 800,
        "integrated_gradients": 6000,
        "occlusion": 20000,
        "shap": 25000,
    }
    return [
        MethodInfo(**InferenceService.method_metadata(key), typical_ms=typical.get(key))
        for key in METHOD_DESCRIPTIONS
    ]


@router.get("/examples", response_model=list[ExampleNote], summary="Case-study notes for the demo")
def examples(request: Request) -> list[ExampleNote]:
    path: Path = request.app.state.settings.examples_path
    if not path.exists():
        return []
    try:
        return [ExampleNote(**row) for row in read_json(path)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not read examples: {exc}") from exc


@router.get("/history", response_model=list[HistoryEntry], summary="Recent predictions (in-memory)")
def history(request: Request) -> list[HistoryEntry]:
    return [HistoryEntry(**e) for e in request.app.state.history.list()]


@router.delete("/history", summary="Clear the session log")
def clear_history(request: Request) -> dict:
    request.app.state.history.clear()
    return {"cleared": True}


@router.get("/results", summary="Published evaluation results for the About/Results page")
def results(request: Request) -> dict:
    """Serves the metrics files so the frontend can show real numbers, not claims."""
    root: Path = request.app.state.settings.results_root / "metrics"
    payload: dict = {}
    for name in (
        "dataset_report",
        "baselines",
        "transformers",
        "evaluation",
        "faithfulness",
        "stability",
        "plausibility",
    ):
        path = root / f"{name}.json"
        if path.exists():
            try:
                payload[name] = read_json(path)
            except Exception:
                payload[name] = {"error": "could not parse results file"}
    payload["available"] = sorted(payload)
    return payload


@router.get("/cache", summary="Prediction cache statistics")
def cache_stats(request: Request) -> dict:
    return request.app.state.inference.cache.stats()


@router.delete("/cache", summary="Clear the prediction cache")
def clear_cache(request: Request) -> dict:
    request.app.state.inference.cache.clear()
    return {"cleared": True}
