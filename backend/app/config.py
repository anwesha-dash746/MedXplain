"""Backend settings, read from the environment with sensible local defaults."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    """Runtime configuration for the API."""

    app_name: str = "MedXplain API"
    version: str = "1.0.0"

    #: where scripts/02_train_transformers.py wrote its checkpoints
    checkpoint_root: Path = REPO_ROOT / "results" / "checkpoints"
    results_root: Path = REPO_ROOT / "results"
    examples_path: Path = REPO_ROOT / "data" / "processed" / "examples.json"
    experiment_config: Path = REPO_ROOT / "configs" / "default.yaml"

    default_model: str = "clinicalbert"
    default_method: str = "attention_rollout"
    device: str = "cpu"

    #: Guard rails. Transformer inference is not free, and an unbounded note
    #: would let a single request occupy the worker for minutes.
    max_note_chars: int = 40_000
    min_note_chars: int = 20
    max_length: int = 128

    #: explanation cost controls, overridable per request within these ceilings
    shap_max_evals: int = 200
    shap_max_evals_ceiling: int = 800
    ig_steps: int = 16
    occlusion_max_words: int = 200

    prediction_cache_size: int = 128
    history_size: int = 50

    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    #: load models lazily on first use rather than at startup (faster boot,
    #: slower first request). Startup preload is the default so the first
    #: clinician-facing request is not the one that pays for it.
    preload_default_model: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        def path_env(key: str, default: Path) -> Path:
            raw = os.environ.get(key)
            return Path(raw) if raw else default

        base = cls()
        return cls(
            checkpoint_root=path_env("MEDXPLAIN_CHECKPOINT_ROOT", base.checkpoint_root),
            results_root=path_env("MEDXPLAIN_RESULTS_ROOT", base.results_root),
            examples_path=path_env("MEDXPLAIN_EXAMPLES", base.examples_path),
            default_model=os.environ.get("MEDXPLAIN_DEFAULT_MODEL", base.default_model),
            default_method=os.environ.get("MEDXPLAIN_DEFAULT_METHOD", base.default_method),
            device=os.environ.get("MEDXPLAIN_DEVICE", base.device),
            max_length=int(os.environ.get("MEDXPLAIN_MAX_LENGTH", base.max_length)),
            preload_default_model=os.environ.get("MEDXPLAIN_PRELOAD", "1") not in ("0", "false"),
            cors_origins=(
                os.environ["MEDXPLAIN_CORS_ORIGINS"].split(",")
                if os.environ.get("MEDXPLAIN_CORS_ORIGINS")
                else base.cors_origins
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
