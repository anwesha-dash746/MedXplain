"""The explanation payload: one shape shared by every explainer and the API.

Every method in this package — attention rollout, SHAP, integrated gradients,
occlusion, and the TF-IDF baseline — returns an :class:`Explanation`. That
uniformity is what makes the paper's central comparison possible: the
faithfulness metrics, the frontend renderer, and the API schema all consume the
same structure, so swapping explainers changes nothing downstream.

Two levels of granularity are carried at once:

* **token level** — the model's own subword units, which is what faithfulness
  metrics must operate on (they delete exactly what the model consumes);
* **word level** — subwords merged back into whitespace words with their
  character offsets in the *original* note, which is what a clinician reads.

Keeping both avoids the common bug where an explanation is scored on wordpieces
but displayed on words, so the numbers in the paper describe something subtly
different from the picture in the figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

Normalisation = Literal["none", "l1", "max_abs", "zscore"]

#: Subword tokens that carry no clinical content and are excluded from the
#: displayed explanation (they are still present for the model).
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<unk>"}

_WORD_RE = re.compile(r"\S+")


@dataclass
class TokenAttribution:
    """Importance of one model token."""

    index: int
    token: str
    score: float
    start: int = -1  # character offset in the original note, -1 if unmapped
    end: int = -1

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "token": self.token,
            "score": round(float(self.score), 6),
            "start": self.start,
            "end": self.end,
        }


@dataclass
class WordAttribution:
    """Importance of one whitespace word, aggregated from its subwords."""

    text: str
    score: float
    start: int
    end: int
    token_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(float(self.score), 6),
            "start": self.start,
            "end": self.end,
            "token_indices": self.token_indices,
        }


@dataclass
class Explanation:
    """A complete, renderable explanation of one prediction."""

    method: str
    tokens: list[TokenAttribution]
    words: list[WordAttribution]
    probability: float
    predicted_label: int
    #: free-form provenance: number of model evaluations, layer used, etc.
    meta: dict = field(default_factory=dict)

    # -- views -------------------------------------------------------------------------

    def token_scores(self) -> np.ndarray:
        return np.asarray([t.score for t in self.tokens], dtype=np.float64)

    def top_words(self, k: int = 10, signed: bool = True) -> list[WordAttribution]:
        """The k words a clinician should look at first."""
        key = (lambda w: w.score) if signed else (lambda w: abs(w.score))
        return sorted(self.words, key=key, reverse=True)[:k]

    def top_negative_words(self, k: int = 10) -> list[WordAttribution]:
        return sorted(self.words, key=lambda w: w.score)[:k]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "probability": round(float(self.probability), 6),
            "predicted_label": int(self.predicted_label),
            "tokens": [t.to_dict() for t in self.tokens],
            "words": [w.to_dict() for w in self.words],
            "meta": self.meta,
        }


# --------------------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------------------


def normalise(scores: np.ndarray, method: Normalisation = "max_abs") -> np.ndarray:
    """Put scores on a comparable scale across explainers.

    SHAP values are in log-odds units, attention rollout is a probability
    distribution summing to one, and gradient norms have no natural unit. The
    faithfulness metrics only use the *ranking*, so they are unaffected, but the
    UI colour scale and any cross-method plot need a shared range. ``max_abs``
    is the default because it preserves sign and zero, which matters when a
    phrase is protective rather than merely unimportant.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if method == "none" or scores.size == 0:
        return scores
    if method == "l1":
        total = np.abs(scores).sum()
        return scores / total if total > 0 else scores
    if method == "max_abs":
        peak = np.abs(scores).max()
        return scores / peak if peak > 0 else scores
    if method == "zscore":
        std = scores.std()
        return (scores - scores.mean()) / std if std > 0 else scores - scores.mean()
    raise ValueError(f"unknown normalisation: {method}")


def aggregate_to_words(
    text: str,
    tokens: Sequence[TokenAttribution],
    reduction: str = "sum",
) -> list[WordAttribution]:
    """Merge subword attributions into whitespace words using character offsets.

    ``sum`` is the default: an explanation's total credit for "norepinephrine"
    should not shrink because the tokenizer split it into five pieces. ``max``
    and ``mean`` are available for comparison.
    """
    words: list[WordAttribution] = []
    for match in _WORD_RE.finditer(text):
        start, end = match.start(), match.end()
        members = [
            t for t in tokens
            if t.start >= 0 and t.end > t.start and t.start < end and t.end > start
        ]
        if members:
            values = [t.score for t in members]
            if reduction == "sum":
                score = float(np.sum(values))
            elif reduction == "mean":
                score = float(np.mean(values))
            elif reduction == "max":
                score = float(max(values, key=abs))
            else:
                raise ValueError(f"unknown reduction: {reduction}")
            indices = [t.index for t in members]
        else:
            score, indices = 0.0, []
        words.append(
            WordAttribution(text=match.group(0), score=score, start=start, end=end, token_indices=indices)
        )
    return words


def build_token_attributions(
    tokens: Sequence[str],
    scores: Sequence[float],
    offsets: Sequence[tuple[int, int]] | None = None,
    drop_special: bool = True,
) -> list[TokenAttribution]:
    """Zip tokens/scores/offsets into attributions, dropping special tokens."""
    out: list[TokenAttribution] = []
    for i, (tok, score) in enumerate(zip(tokens, scores)):
        if drop_special and tok in SPECIAL_TOKENS:
            continue
        start, end = (offsets[i] if offsets is not None else (-1, -1))
        out.append(
            TokenAttribution(index=i, token=tok, score=float(score), start=int(start), end=int(end))
        )
    return out


def head_fraction_of(model) -> float | None:
    """The head share for head+tail truncation the model was trained with, else None (head only)."""
    cfg = getattr(model, "cfg", None)
    if cfg is not None and getattr(cfg, "truncation_strategy", None) == "head_tail":
        return float(cfg.head_fraction)
    return None


def _window(n_tokens: int, max_length: int, head_fraction: float | None) -> list[int]:
    """Indices of the body tokens the model reads, matching ClinicalNoteDataset._encode."""
    body = max_length - 2
    if n_tokens <= body:
        return list(range(n_tokens))
    if head_fraction is None:
        return list(range(body))
    n_head = int(body * head_fraction)
    n_tail = body - n_head
    return list(range(n_head)) + list(range(n_tokens - n_tail, n_tokens))


def encode_for_model(tokenizer, texts, max_length: int, head_fraction: float | None = None) -> dict:
    """Batch-encode texts exactly as the model was trained to read them, padded to a tensor.

    With ``head_fraction`` set this reproduces the training-time head+tail
    truncation; without it, plain head truncation.
    """
    if isinstance(texts, str):
        texts = [texts]
    rows = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        kept = [ids[i] for i in _window(len(ids), max_length, head_fraction)]
        rows.append([tokenizer.cls_token_id] + kept + [tokenizer.sep_token_id])
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        attention[i, : len(r)] = 1
    return {"input_ids": input_ids, "attention_mask": attention}


def encode_with_offsets(tokenizer, text: str, max_length: int, head_fraction: float | None = None) -> dict:
    """Tokenise once, keeping the character offsets the UI needs.

    The tokens kept are exactly those the model reads: head+tail when the model
    was trained that way (``head_fraction`` set), otherwise the head. When the
    middle of a long note is skipped, ``unread_span`` gives its character range
    so the UI can mark it as not read rather than leaving an unexplained gap.
    """
    full = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids, offsets_all = full["input_ids"], [tuple(o) for o in full["offset_mapping"]]
    keep = _window(len(ids), max_length, head_fraction)

    input_ids = [tokenizer.cls_token_id] + [ids[i] for i in keep] + [tokenizer.sep_token_id]
    offsets = [(0, 0)] + [offsets_all[i] for i in keep] + [(0, 0)]
    truncated = len(ids) > max_length - 2

    unread_span = None
    if truncated:
        gap = [i for i in range(1, len(keep)) if keep[i] != keep[i - 1] + 1]
        if gap:  # head+tail: the middle was skipped
            last_head, first_tail = keep[gap[0] - 1], keep[gap[0]]
            unread_span = (int(offsets_all[last_head][1]), int(offsets_all[first_tail][0]))
        else:  # head only: everything after the window was skipped
            unread_span = (int(offsets_all[keep[-1]][1]), len(text))

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        "offsets": offsets,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "truncated": truncated,
        "unread_span": unread_span,
    }


def merge_explanations(explanations: Iterable[Explanation], method: str = "ensemble") -> Explanation:
    """Average several normalised explanations of the same input.

    Used for the rank-ensemble condition (attention + SHAP), which the
    evaluation reports alongside the individual methods.
    """
    explanations = list(explanations)
    if not explanations:
        raise ValueError("nothing to merge")
    base = explanations[0]
    stacked = np.vstack([normalise(e.token_scores(), "max_abs") for e in explanations])
    mean_scores = stacked.mean(axis=0)
    tokens = [
        TokenAttribution(index=t.index, token=t.token, score=float(s), start=t.start, end=t.end)
        for t, s in zip(base.tokens, mean_scores)
    ]
    return Explanation(
        method=method,
        tokens=tokens,
        words=[],  # recomputed by the caller, which has the source text
        probability=base.probability,
        predicted_label=base.predicted_label,
        meta={"merged_from": [e.method for e in explanations]},
    )
