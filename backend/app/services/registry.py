"""Model registry: discovers, loads and caches trained checkpoints.

Loading a BERT checkpoint takes seconds and roughly 500 MB of RSS, so models are
loaded once and held for the process lifetime. Discovery is filesystem-driven —
whatever ``scripts/02_train_transformers.py`` checkpointed is what the API can
serve — which keeps the frontend's model list honest instead of advertising a
model that was never trained.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from medxplain.models.transformer import MODEL_DISPLAY, MODEL_REGISTRY, ClinicalRiskModel
from medxplain.utils import get_logger, read_json

LOGGER = get_logger("medxplain.api.registry")


@dataclass
class LoadedModel:
    key: str
    model: ClinicalRiskModel
    tokenizer: object
    info: dict

    @property
    def display_name(self) -> str:
        return self.info.get("display_name") or MODEL_DISPLAY.get(self.key, self.key)

    @property
    def threshold(self) -> float:
        return float(self.info.get("threshold", 0.5))

    @property
    def max_length(self) -> int:
        return int(self.info.get("max_length", 128))

    @property
    def outcome(self) -> str:
        return self.info.get("outcome", "mortality")


class ModelRegistry:
    """Thread-safe lazy loader over the checkpoint directory."""

    def __init__(self, checkpoint_root: Path, device: str = "cpu"):
        self.checkpoint_root = Path(checkpoint_root)
        self.device = device
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    # -- discovery ---------------------------------------------------------------------

    def available(self) -> list[str]:
        """Checkpoint keys present on disk, ordered as in the model registry."""
        if not self.checkpoint_root.exists():
            return []
        found = {
            d.name
            for d in self.checkpoint_root.iterdir()
            if d.is_dir() and (d / "medxplain_config.json").exists()
        }
        ordered = [k for k in MODEL_REGISTRY if k in found]
        return ordered + sorted(found - set(ordered))

    def describe(self, key: str) -> dict:
        """Metadata for one checkpoint without loading its weights."""
        directory = self.checkpoint_root / key
        info: dict = {
            "key": key,
            "display_name": MODEL_DISPLAY.get(key, key),
            "hf_id": MODEL_REGISTRY.get(key, key),
            "available": (directory / "medxplain_config.json").exists(),
        }
        run_info = directory / "run_info.json"
        if run_info.exists():
            try:
                info.update(read_json(run_info))
            except Exception as exc:
                LOGGER.warning("could not read %s: %s", run_info, exc)
        return info

    def loaded_keys(self) -> list[str]:
        return sorted(self._loaded)

    # -- loading -----------------------------------------------------------------------

    def get(self, key: str | None = None, default: str = "clinicalbert") -> LoadedModel:
        key = key or default
        if key in self._loaded:
            return self._loaded[key]

        with self._lock:
            if key in self._loaded:  # another thread won the race
                return self._loaded[key]

            directory = self.checkpoint_root / key
            if not (directory / "medxplain_config.json").exists():
                available = self.available()
                raise FileNotFoundError(
                    f"no trained checkpoint for {key!r} at {directory}. "
                    + (
                        f"Available: {', '.join(available)}."
                        if available
                        else "Train one with: python scripts/02_train_transformers.py"
                    )
                )

            LOGGER.info("loading checkpoint %s from %s", key, directory)
            model, tokenizer = ClinicalRiskModel.load(directory, device=self.device)
            model.eval()
            torch.set_grad_enabled(False)  # re-enabled per-request by gradient explainers

            entry = LoadedModel(key=key, model=model, tokenizer=tokenizer, info=self.describe(key))
            self._loaded[key] = entry
            LOGGER.info("loaded %s (%s)", key, entry.display_name)
            return entry

    def preload(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except Exception as exc:
            LOGGER.warning("could not preload %s: %s", key, exc)
            return False

    def unload(self, key: str) -> None:
        with self._lock:
            self._loaded.pop(key, None)
