"""SHAP explanations for the clinical risk classifier.

SHAP's ``PartitionExplainer`` with a ``Text`` masker is the standard choice for
transformers: it exploits the hierarchical structure of text so that the number
of model evaluations grows with ``O(n log n)`` in the number of tokens instead of
the ``2^n`` a naive Shapley computation would need. Even so it is by far the most
expensive explainer here — hundreds of forward passes per note — which is itself
a finding worth reporting, because a method that takes 30 seconds per note is not
deployable at the bedside no matter how faithful it is.

Two practical decisions are worth flagging for the paper:

*   **Output space.** SHAP is computed on the *log-odds* of the positive class,
    not the probability. Shapley values are additive, and additivity only holds
    on a scale where the model is (locally) linear; on the probability scale the
    values do not sum to the prediction once the sigmoid saturates, which is
    exactly the high-risk regime we care about.
*   **The baseline matters.** The masker replaces tokens with the tokenizer's
    mask token by default. Deleting them instead changes positional structure
    and produces systematically different attributions, so the choice is a
    config value rather than an implementation detail.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from medxplain.explain.payload import (
    Explanation,
    aggregate_to_words,
    build_token_attributions,
    encode_with_offsets,
    normalise,
)
from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.explain.shap")


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class ShapExplainer:
    """Partition-SHAP over the note, returned in the shared payload format."""

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 512,
        max_evals: int = 300,
        batch_size: int = 16,
        device: str = "cpu",
        output_space: str = "logit",  # logit | probability
        mask_token: str | None = None,
        word_reduction: str = "sum",
        silent: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.device = device
        self.output_space = output_space
        self.mask_token = mask_token or tokenizer.mask_token or "[MASK]"
        self.word_reduction = word_reduction
        self.silent = silent
        self._explainer = None

    # -- model wrapper -----------------------------------------------------------------

    def _predict(self, texts: Sequence[str]) -> np.ndarray:
        """Batched scalar output for SHAP: positive-class score for each string."""
        self.model.eval()
        scores: list[float] = []
        texts = [str(t) for t in texts]
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                logits = self.model(
                    enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                )["logits"]
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                scores.extend(probs.tolist())
        arr = np.asarray(scores)
        return _logit(arr) if self.output_space == "logit" else arr

    def _build(self):
        import shap

        masker = shap.maskers.Text(self.tokenizer, mask_token=self.mask_token, collapse_mask_token=True)
        self._explainer = shap.Explainer(self._predict, masker, silent=self.silent)
        return self._explainer

    # -- public API --------------------------------------------------------------------

    def explain(self, text: str, target: int | None = None) -> Explanation:
        explainer = self._explainer or self._build()

        # SHAP tokenises the raw string itself; we truncate first so the values it
        # returns describe exactly the span the model actually scored.
        enc = encode_with_offsets(self.tokenizer, text, self.max_length)
        if enc["truncated"]:
            keep_to = max((e for _, e in enc["offsets"] if e > 0), default=len(text))
            scored_text = text[:keep_to]
        else:
            scored_text = text

        values = explainer([scored_text], max_evals=self.max_evals, silent=self.silent)
        shap_tokens = list(values.data[0])
        shap_scores = np.asarray(values.values[0], dtype=np.float64)

        with torch.no_grad():
            probs = self.model.predict_proba(
                enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            )[0]
        probability, predicted = float(probs[1]), int(probs.argmax())

        # SHAP's Text masker emits whitespace-preserving surface strings; map each
        # back to its character span by walking the source text left to right.
        spans, cursor = [], 0
        for tok in shap_tokens:
            stripped = tok.strip()
            if not stripped:
                spans.append((-1, -1))
                continue
            found = scored_text.find(stripped, cursor)
            if found < 0:
                spans.append((-1, -1))
                continue
            spans.append((found, found + len(stripped)))
            cursor = found + len(stripped)

        tokens = build_token_attributions(
            [t.strip() or "_" for t in shap_tokens],
            normalise(shap_scores, "max_abs"),
            spans,
            drop_special=False,
        )
        words = aggregate_to_words(text, tokens, reduction=self.word_reduction)

        return Explanation(
            method="shap",
            tokens=tokens,
            words=words,
            probability=probability,
            predicted_label=predicted,
            meta={
                "max_evals": self.max_evals,
                "output_space": self.output_space,
                "mask_token": self.mask_token,
                "base_value": float(np.asarray(values.base_values).ravel()[0]),
                "sum_of_values": float(shap_scores.sum()),
                "truncated": enc["truncated"],
                "n_model_evaluations": int(self.max_evals),
                "signed": True,
            },
        )


class OcclusionExplainer:
    """Leave-one-word-out occlusion: the honest, cheap upper bound on cost.

    Each word is replaced by the mask token and the drop in positive-class
    log-odds is recorded. It needs exactly one forward pass per word, which is
    far fewer than SHAP for a typical note, and it is exactly the quantity the
    comprehensiveness metric measures at k=1. Including it keeps the comparison
    honest: if occlusion matches SHAP on faithfulness at a fraction of the cost,
    that is the deployable choice and the paper should say so.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 512,
        batch_size: int = 16,
        device: str = "cpu",
        max_words: int = 400,
        word_reduction: str = "sum",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.max_words = max_words
        self.word_reduction = word_reduction

    def _score(self, texts: Sequence[str]) -> np.ndarray:
        self.model.eval()
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    list(texts[i : i + self.batch_size]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                logits = self.model(
                    enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                )["logits"]
                out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
        return _logit(np.asarray(out))

    def explain(self, text: str, target: int | None = None) -> Explanation:
        import re

        enc = encode_with_offsets(self.tokenizer, text, self.max_length)
        mask = self.tokenizer.mask_token or "[MASK]"

        spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)][: self.max_words]
        variants = [text[:s] + mask + text[e:] for _, s, e in spans]

        base = float(self._score([text])[0])
        perturbed = self._score(variants) if variants else np.array([])
        drops = base - perturbed  # positive = removing the word lowered the risk

        tokens = build_token_attributions(
            [w for w, _, _ in spans],
            normalise(drops, "max_abs"),
            [(s, e) for _, s, e in spans],
            drop_special=False,
        )
        words = aggregate_to_words(text, tokens, reduction="max")

        with torch.no_grad():
            probs = self.model.predict_proba(
                enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            )[0]

        return Explanation(
            method="occlusion",
            tokens=tokens,
            words=words,
            probability=float(probs[1]),
            predicted_label=int(probs.argmax()),
            meta={
                "n_model_evaluations": len(variants) + 1,
                "base_logit": base,
                "truncated": enc["truncated"],
                "signed": True,
            },
        )
