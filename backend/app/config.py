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
    #: historical case library for the clinical workspace (the processed corpus)
    case_library_path: Path = REPO_ROOT / "data" / "processed" / "corpus.parquet"
    #: external Peer Consult URL; unused since Peer Consult is built in (kept for compatibility)
    peer_consult_url: str = ""

    #: Peer Consult storage, encryption and policy
    peer_db_path: Path = REPO_ROOT / "data" / "peer_consult" / "peer_consult.db"
    peer_key: str | None = None  # MEDXPLAIN_PEER_KEY (Fernet key); falls back to a dev key file
    peer_key_path: Path = REPO_ROOT / "data" / "peer_consult" / ".dev_encryption_key"
    peer_verifier: str = "none"
    #: honour development verification overrides; never enable in a shared deployment
    peer_allow_dev_verification: bool = False
    peer_session_hours: float = 12.0
    peer_retention_days: int = 365
    peer_password_iterations: int = 310_000

    #: preferred model; falls back to clinicalbert / the first available at startup
    default_model: str = "clinicalbert_512"
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
    #: words occlusion masks one at a time; 350 covers the longest note in the corpus (344 words)
    occlusion_max_words: int = 350

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
            case_library_path=path_env("MEDXPLAIN_CASE_LIBRARY", base.case_library_path),
            peer_consult_url=os.environ.get("MEDXPLAIN_PEER_CONSULT_URL", base.peer_consult_url),
            peer_db_path=path_env("MEDXPLAIN_PEER_DB", base.peer_db_path),
            peer_key=os.environ.get("MEDXPLAIN_PEER_KEY") or None,
            peer_key_path=path_env("MEDXPLAIN_PEER_KEY_PATH", base.peer_key_path),
            peer_verifier=os.environ.get("MEDXPLAIN_PEER_VERIFIER", base.peer_verifier),
            peer_allow_dev_verification=os.environ.get("MEDXPLAIN_PEER_ALLOW_DEV_VERIFICATION", "0")
            in ("1", "true", "yes"),
            peer_retention_days=int(os.environ.get("MEDXPLAIN_PEER_RETENTION_DAYS", base.peer_retention_days)),
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
