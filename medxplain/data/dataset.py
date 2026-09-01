"""Torch dataset + collator for clinical-note classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "EncodedBatch":
        return EncodedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=None if self.labels is None else self.labels.to(device),
        )

    def as_model_inputs(self) -> dict:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


class ClinicalNoteDataset(Dataset):
    """Tokenises on demand and caches, so a tokenizer swap costs nothing on disk.

    Notes longer than ``max_length`` are handled by ``head_tail`` truncation by
    default: keeping the first *and* last tokens preserves both the presenting
    history and the discharge disposition, which plain right-truncation throws
    away. Sliding-window chunking is available for inference where the full
    document must be covered.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None,
        tokenizer,
        max_length: int = 512,
        truncation_strategy: str = "head_tail",
        head_fraction: float = 0.35,
    ) -> None:
        if truncation_strategy not in {"head", "tail", "head_tail"}:
            raise ValueError(f"unknown truncation strategy: {truncation_strategy}")
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.head_fraction = head_fraction
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, text: str) -> dict:
        tok = self.tokenizer
        body = self.max_length - 2
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]

        if len(ids) <= body:
            kept = ids
        elif self.truncation_strategy == "head":
            kept = ids[:body]
        elif self.truncation_strategy == "tail":
            kept = ids[-body:]
        else:
            n_head = int(body * self.head_fraction)
            n_tail = body - n_head
            kept = ids[:n_head] + ids[-n_tail:]

        input_ids = [tok.cls_token_id] + kept + [tok.sep_token_id]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def __getitem__(self, index: int) -> dict:
        if index not in self._cache:
            self._cache[index] = self._encode(self.texts[index])
        item = dict(self._cache[index])
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item


@dataclass
class Collator:
    """Pads a list of encodings to the longest item in the batch."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> EncodedBatch:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = np.full((len(features), max_len), self.pad_token_id, dtype=np.int64)
        attention = np.zeros((len(features), max_len), dtype=np.int64)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = f["input_ids"]
            attention[i, :n] = f["attention_mask"]
        labels = None
        if "label" in features[0]:
            labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return EncodedBatch(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention),
            labels=labels,
        )


def class_weights(labels: Sequence[int], scheme: str = "balanced") -> torch.Tensor:
    """Loss weights for the rare-event setting.

    ``balanced`` follows sklearn's n/(k*n_c); ``sqrt_balanced`` softens it, which
    in practice calibrates better while keeping most of the recall benefit.
    """
    y = np.asarray(labels)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    if scheme == "balanced":
        w = len(y) / (2.0 * counts)
    elif scheme == "sqrt_balanced":
        w = np.sqrt(len(y) / (2.0 * counts))
    elif scheme == "none":
        w = np.ones(2)
    else:
        raise ValueError(f"unknown class weight scheme: {scheme}")
    return torch.tensor(w, dtype=torch.float32)
