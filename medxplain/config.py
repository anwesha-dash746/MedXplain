"""Configuration loading.

A run is defined entirely by a YAML file plus command-line overrides. The
resolved config is written next to the results, so every reported number is
traceable to the exact settings that produced it — which is the difference
between a reproducible artefact and a directory of numbers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class Config(dict):
    """A dict with dotted access, so ``cfg.get_path('train.epochs')`` works."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict:
        return self.get(name, {}) or {}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config and apply ``dotted.key=value`` overrides."""
    path = Path(path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(copy.deepcopy(raw))
    cfg["_config_path"] = str(path)
    for key, value in (overrides or {}).items():
        cfg.set_path(key, value)
    return cfg


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["train.epochs=2", "data.n_notes=500"]`` into a dict.

    Values are parsed as YAML so numbers, booleans, ``null`` and inline lists all
    work without extra flags.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def save_config(cfg: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
