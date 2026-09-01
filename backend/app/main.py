"""MedXplain FastAPI application.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000

Interactive API docs are at http://localhost:8000/docs.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.routers import meta, predict
from backend.app.routers.meta import History
from backend.app.services.inference import InferenceService
from backend.app.services.registry import ModelRegistry
from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.api")

DESCRIPTION = """
Explainable transformer-based clinical risk prediction.

`POST /predict` takes the free text of a clinical note and returns a risk
probability together with a word-level explanation of that probability.

Two things about the response are easy to misread, so they are stated in the
payload itself:

* **`decision_threshold`** is chosen on the validation split, never on test.
  `predicted_label` is `probability >= decision_threshold`, which is generally
  not 0.5 for a rare outcome.
* **`signed`** on each explanation says whether negative scores mean "lowers the
  risk". Attention-based maps are unsigned - a high score means the model used
  that word, not that the word raises risk.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.registry = ModelRegistry(settings.checkpoint_root, device=settings.device)
    app.state.inference = InferenceService(app.state.registry, settings)
    app.state.history = History(settings.history_size)

    available = app.state.registry.available()
    LOGGER.info("checkpoint root: %s", settings.checkpoint_root)
    LOGGER.info("models available: %s", available or "(none yet)")

    if settings.preload_default_model and available:
        key = settings.default_model if settings.default_model in available else available[0]
        # Warm the model at startup so the first real request is not the one that
        # pays the several-second load cost.
        app.state.registry.preload(key)
    elif not available:
        LOGGER.warning(
            "no trained checkpoints found - /predict will return 503 until "
            "scripts/02_train_transformers.py has run"
        )

    yield
    LOGGER.info("shutting down")


app = FastAPI(
    title="MedXplain API",
    description=DESCRIPTION,
    version=get_settings().version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(FileNotFoundError)
async def missing_model_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(predict.router, prefix="/api")
app.include_router(meta.router, prefix="/api")


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": "MedXplain API",
        "version": get_settings().version,
        "docs": "/docs",
        "health": "/api/health",
    }
