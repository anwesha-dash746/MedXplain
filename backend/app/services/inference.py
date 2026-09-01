"""Prediction + explanation service.

Everything a request needs, in one place: cleaning, scoring, explanation, the
risk-band language shown to the user, and the caching that keeps a re-run of the
same note instant. The route handlers stay thin so this logic is testable
without an HTTP client.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from backend.app.config import Settings
from backend.app.services.registry import LoadedModel, ModelRegistry
from medxplain.data.preprocess import PreprocessConfig, clean_note
from medxplain.explain import METHOD_DESCRIPTIONS, build_explainer
from medxplain.explain.payload import Explanation
from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.api.inference")

#: Explanation cost class, surfaced to the UI so a user can predict the wait.
METHOD_COST: dict[str, str] = {
    "attention_rollout": "fast",
    "attention_raw": "fast",
    "attention_grad": "fast",
    "integrated_gradients": "moderate",
    "occlusion": "slow",
    "shap": "slow",
}

METHOD_DISPLAY: dict[str, str] = {
    "attention_rollout": "Attention rollout",
    "attention_raw": "Raw attention (last layer)",
    "attention_grad": "Gradient-weighted attention",
    "shap": "SHAP",
    "occlusion": "Word occlusion",
    "integrated_gradients": "Integrated gradients",
}

#: Methods whose scores carry a direction (negative = lowers risk).
SIGNED_METHODS = {"shap", "occlusion", "integrated_gradients", "attention_grad"}


@dataclass
class PredictionOutcome:
    note_id: str
    probability: float
    predicted_label: int
    threshold: float
    explanations: list[tuple[Explanation, float]]  # (explanation, compute_ms)
    n_chars: int
    n_tokens: int
    truncated: bool
    inference_ms: float
    warnings: list[str]
    model: LoadedModel


class LruCache:
    """Small thread-safe LRU over (note, model, method) -> result."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._data: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        return {"size": len(self._data), "maxsize": self.maxsize, "hits": self.hits, "misses": self.misses}


def risk_band(probability: float, threshold: float) -> str:
    """Coarse label shown next to the number.

    Bands are anchored on the model's own validation-selected threshold rather
    than fixed cut-points, so "high" always means "above the operating point
    this model was tuned to", which is the only reading that stays true when the
    model or the outcome changes.
    """
    if probability >= max(0.75, threshold * 1.5):
        return "high"
    if probability >= threshold:
        return "elevated"
    if probability >= threshold * 0.5:
        return "borderline"
    return "low"


def _hash(*parts: str) -> str:
    digest = hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


class InferenceService:
    """Scores notes and produces explanations, with caching and guard rails."""

    def __init__(self, registry: ModelRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings
        self.cache = LruCache(settings.prediction_cache_size)
        self._preprocess = PreprocessConfig()
        # Explainers hold no per-request state, so one per (model, method) is safe
        # and avoids rebuilding SHAP's masker on every call.
        self._explainers: dict[tuple[str, str], object] = {}
        self._lock = threading.Lock()

    # -- validation --------------------------------------------------------------------

    def validate(self, text: str) -> tuple[str, list[str]]:
        """Clean the note and return any warnings worth showing the user."""
        warnings: list[str] = []
        stripped = text.strip()
        if len(stripped) < self.settings.min_note_chars:
            raise ValueError(
                f"note is too short to score ({len(stripped)} characters; "
                f"at least {self.settings.min_note_chars} are needed)"
            )
        if len(stripped) > self.settings.max_note_chars:
            raise ValueError(
                f"note is too long ({len(stripped)} characters; the limit is "
                f"{self.settings.max_note_chars}). Split it into separate notes."
            )

        cleaned = clean_note(stripped, self._preprocess)
        if not cleaned.strip():
            raise ValueError("note contained no usable text after cleaning")

        # A note with almost no clinical vocabulary is out of distribution; the model
        # will still return a number, and the user deserves to know it means little.
        if not _looks_clinical(cleaned):
            warnings.append(
                "This text does not look like a clinical note. The model was fine-tuned on "
                "discharge summaries, so the risk score and explanation may be meaningless."
            )
        return cleaned, warnings

    # -- explainers --------------------------------------------------------------------

    def _explainer(self, entry: LoadedModel, method: str, max_evals: int | None = None):
        key = (entry.key, method if method != "shap" else f"shap:{max_evals}")
        with self._lock:
            if key in self._explainers:
                return self._explainers[key]

            kwargs: dict = {"max_length": entry.max_length}
            if method == "shap":
                kwargs["max_evals"] = min(
                    max_evals or self.settings.shap_max_evals,
                    self.settings.shap_max_evals_ceiling,
                )
            elif method == "integrated_gradients":
                kwargs["n_steps"] = self.settings.ig_steps
            elif method == "occlusion":
                kwargs["max_words"] = self.settings.occlusion_max_words

            explainer = build_explainer(
                method, entry.model, entry.tokenizer, device=self.settings.device, **kwargs
            )
            self._explainers[key] = explainer
            return explainer

    # -- main entry point --------------------------------------------------------------

    def predict(
        self,
        text: str,
        model_key: str | None = None,
        methods: list[str] | None = None,
        explain: bool = True,
        max_evals: int | None = None,
    ) -> PredictionOutcome:
        started = time.perf_counter()
        cleaned, warnings = self.validate(text)

        entry = self.registry.get(model_key, default=self.settings.default_model)
        methods = methods if (explain and methods) else ([] if not explain else [self.settings.default_method])

        note_id = _hash(cleaned, entry.key)
        cache_key = _hash(cleaned, entry.key, ",".join(methods), str(max_evals))
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        enc = entry.tokenizer(cleaned, add_special_tokens=False, truncation=False)
        n_tokens = len(enc["input_ids"])
        truncated = n_tokens > entry.max_length - 2
        if truncated:
            warnings.append(
                f"The note is {n_tokens} tokens; this model reads {entry.max_length}. "
                "The score and highlights cover the truncated window only."
            )

        # score once, then explain — so every explanation describes the same number
        with torch.no_grad():
            model_inputs = entry.tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=entry.max_length,
            )
            probs = entry.model.predict_proba(
                model_inputs["input_ids"].to(self.settings.device),
                model_inputs["attention_mask"].to(self.settings.device),
            )[0]
        probability = float(probs[1])
        threshold = entry.threshold
        predicted = int(probability >= threshold)

        explanations: list[tuple[Explanation, float]] = []
        for method in methods:
            explain_started = time.perf_counter()
            try:
                explainer = self._explainer(entry, method, max_evals)
                # gradient methods need autograd, which the registry disabled globally
                with torch.set_grad_enabled(method in ("integrated_gradients", "attention_grad")):
                    explanation = explainer.explain(cleaned)
                explanations.append((explanation, (time.perf_counter() - explain_started) * 1000))
            except Exception as exc:  # one failing method must not sink the request
                LOGGER.exception("explanation method %s failed", method)
                warnings.append(f"The {method} explanation could not be computed: {exc}")

        outcome = PredictionOutcome(
            note_id=note_id,
            probability=probability,
            predicted_label=predicted,
            threshold=threshold,
            explanations=explanations,
            n_chars=len(cleaned),
            n_tokens=n_tokens,
            truncated=truncated,
            inference_ms=(time.perf_counter() - started) * 1000,
            warnings=warnings,
            model=entry,
        )
        self.cache.put(cache_key, outcome)
        return outcome

    # -- presentation helpers ----------------------------------------------------------

    @staticmethod
    def method_metadata(method: str) -> dict:
        return {
            "key": method,
            "display_name": METHOD_DISPLAY.get(method, method),
            "description": METHOD_DESCRIPTIONS.get(method, ""),
            "signed": method in SIGNED_METHODS,
            "cost": METHOD_COST.get(method, "moderate"),
        }


#: A short list of high-frequency clinical tokens. Deliberately crude: it exists
#: to catch "hello world" and pasted lorem ipsum, not to gate real notes.
_CLINICAL_MARKERS = (
    "patient", "pt ", "admission", "discharge", "hospital", "diagnosis", "history",
    "mg ", "dose", "exam", "blood", "hr ", "bp ", "wbc", "lab", "course", "chronic",
    "acute", "treatment", "medication", "clinical", "symptoms", "presented", "follow up",
)


def _looks_clinical(text: str, min_markers: int = 2) -> bool:
    lowered = text.lower()
    return sum(1 for marker in _CLINICAL_MARKERS if marker in lowered) >= min_markers
