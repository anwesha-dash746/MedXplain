"""Classical baselines: TF-IDF with linear and tree ensembles.

A transformer paper is only as convincing as the baselines it beats. TF-IDF +
logistic regression is a genuinely strong baseline on clinical notes — often
within a few AUROC points of a fine-tuned BERT — so it is tuned here rather
than deliberately handicapped: n-gram range, minimum document frequency and
regularisation strength are all selected on the validation split.

Each baseline also exposes ``explain_tokens``, which returns per-token
contributions in the same payload shape as the transformer explainers. That
makes the "does domain pretraining buy better *explanations*, not just better
AUROC" comparison possible, and it gives the frontend a fast fallback explainer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.models.baselines")

_TOKEN_RE = re.compile(r"\b\w[\w\-/]*\b")


@dataclass
class BaselineConfig:
    kind: str = "logreg"  # logreg | random_forest | xgboost
    max_features: int = 60_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 3
    sublinear_tf: bool = True
    class_weight: str | None = "balanced"
    seed: int = 42
    #: grid searched on the validation split; empty means "use the defaults above"
    grid: dict = field(default_factory=dict)


def _vectorizer(cfg: BaselineConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"(?u)\b\w[\w\-/]+\b",
    )


def _estimator(cfg: BaselineConfig):
    if cfg.kind == "logreg":
        return LogisticRegression(
            max_iter=2000,
            C=cfg.grid.get("C", 1.0),
            class_weight=cfg.class_weight,
            solver="liblinear",
            random_state=cfg.seed,
        )
    if cfg.kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.grid.get("n_estimators", 500),
            max_depth=cfg.grid.get("max_depth", None),
            min_samples_leaf=cfg.grid.get("min_samples_leaf", 2),
            class_weight=cfg.class_weight,
            n_jobs=-1,
            random_state=cfg.seed,
        )
    if cfg.kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=cfg.grid.get("n_estimators", 400),
            max_depth=cfg.grid.get("max_depth", 6),
            learning_rate=cfg.grid.get("learning_rate", 0.1),
            subsample=0.9,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=cfg.seed,
        )
    raise ValueError(f"unknown baseline kind: {cfg.kind}")


class TfidfBaseline:
    """TF-IDF + a scikit-learn classifier, with token-level explanations."""

    def __init__(self, cfg: BaselineConfig | None = None):
        self.cfg = cfg or BaselineConfig()
        self.pipeline: Pipeline | None = None
        self.calibrated = False

    # -- training ---------------------------------------------------------------------

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfBaseline":
        self.pipeline = Pipeline(
            [("tfidf", _vectorizer(self.cfg)), ("clf", _estimator(self.cfg))]
        )
        sample_weight = None
        if self.cfg.kind == "xgboost" and self.cfg.class_weight == "balanced":
            # XGBClassifier has no class_weight; the equivalent is scale_pos_weight
            y = np.asarray(labels)
            pos = max(int(y.sum()), 1)
            self.pipeline.named_steps["clf"].set_params(
                scale_pos_weight=float((len(y) - pos) / pos)
            )
        self.pipeline.fit(list(texts), np.asarray(labels), **({} if sample_weight is None else {}))
        return self

    def tune(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Sequence[str],
        val_labels: Sequence[int],
        grid: list[dict] | None = None,
    ) -> dict:
        """Select hyperparameters on the validation split by AUROC.

        Explicit rather than ``GridSearchCV`` so the exact same validation split
        used for the transformer's early stopping is used here too — otherwise
        the comparison between model families is confounded by model selection.
        """
        grid = grid or default_grid(self.cfg.kind)
        best = {"auroc": -np.inf, "params": {}}
        for params in grid:
            trial_cfg = BaselineConfig(**{**self.cfg.__dict__, "grid": params})
            trial_cfg.ngram_range = params.get("ngram_range", self.cfg.ngram_range)
            trial_cfg.min_df = params.get("min_df", self.cfg.min_df)
            model = TfidfBaseline(trial_cfg).fit(train_texts, train_labels)
            auroc = roc_auc_score(val_labels, model.predict_proba(val_texts))
            LOGGER.info("  %s %s -> val AUROC %.4f", self.cfg.kind, params, auroc)
            if auroc > best["auroc"]:
                best = {"auroc": float(auroc), "params": params, "model": model}
        self.pipeline = best.pop("model").pipeline
        self.cfg.grid = best["params"]
        self.cfg.ngram_range = best["params"].get("ngram_range", self.cfg.ngram_range)
        self.cfg.min_df = best["params"].get("min_df", self.cfg.min_df)
        return best

    def calibrate(self, val_texts: Sequence[str], val_labels: Sequence[int]) -> "TfidfBaseline":
        """Post-hoc Platt/isotonic calibration on the validation split.

        Reported alongside Brier score: an uncalibrated risk score is unusable
        at the bedside even when its ranking (AUROC) is good.
        """
        assert self.pipeline is not None, "fit() before calibrate()"
        # scikit-learn removed cv="prefit" in 1.6 in favour of wrapping the fitted
        # estimator; support both so the repo runs on either version.
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(FrozenEstimator(self.pipeline), method="sigmoid")
        except ImportError:
            calibrated = CalibratedClassifierCV(self.pipeline, method="sigmoid", cv="prefit")
        calibrated.fit(list(val_texts), np.asarray(val_labels))
        self.pipeline = calibrated
        self.calibrated = True
        return self

    # -- inference --------------------------------------------------------------------

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        assert self.pipeline is not None, "model is not fitted"
        return self.pipeline.predict_proba(list(texts))[:, 1]

    # -- explanation ------------------------------------------------------------------

    def _inner(self):
        """Unwrap the calibration wrapper to reach the fitted vectorizer/estimator."""
        pipe = self.pipeline
        if isinstance(pipe, CalibratedClassifierCV):
            pipe = pipe.calibrated_classifiers_[0].estimator
        return pipe

    def explain_tokens(self, text: str, top_k: int | None = None) -> list[dict]:
        """Per-token contribution to the positive class.

        For logistic regression this is exact: contribution = tfidf_value *
        coefficient, which sums (plus the intercept) to the decision function.
        For tree ensembles we fall back to impurity/gain importance multiplied by
        the token's TF-IDF value, which is a heuristic and is labelled as such in
        the payload.
        """
        pipe = self._inner()
        vec: TfidfVectorizer = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]

        row = vec.transform([text])
        vocab = vec.get_feature_names_out()

        if hasattr(clf, "coef_"):
            weights = clf.coef_.ravel()
            exact = True
        elif hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
            exact = False
        else:
            return []

        contributions: dict[str, float] = {}
        for idx, value in zip(row.indices, row.data):
            feature = vocab[idx]
            contributions[feature] = float(value * weights[idx])

        # project n-gram contributions back onto the note's word occurrences
        spans = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
        scores = np.zeros(len(spans), dtype=np.float64)
        lowered = [s[0].lower() for s in spans]
        for feature, contribution in contributions.items():
            parts = feature.split()
            n = len(parts)
            share = contribution / n
            for i in range(len(lowered) - n + 1):
                if lowered[i : i + n] == parts:
                    scores[i : i + n] += share

        payload = [
            {
                "token": tok,
                "start": start,
                "end": end,
                "score": float(score),
                "exact": exact,
            }
            for (tok, start, end), score in zip(spans, scores)
        ]
        if top_k is not None:
            payload.sort(key=lambda d: abs(d["score"]), reverse=True)
            payload = payload[:top_k]
        return payload

    def top_features(self, k: int = 25) -> dict[str, list[tuple[str, float]]]:
        """Globally most predictive n-grams, for the paper's qualitative table."""
        pipe = self._inner()
        vec = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        vocab = vec.get_feature_names_out()
        if hasattr(clf, "coef_"):
            w = clf.coef_.ravel()
            order = np.argsort(w)
            return {
                "positive": [(vocab[i], float(w[i])) for i in order[::-1][:k]],
                "negative": [(vocab[i], float(w[i])) for i in order[:k]],
            }
        w = clf.feature_importances_
        order = np.argsort(w)[::-1][:k]
        return {"positive": [(vocab[i], float(w[i])) for i in order], "negative": []}

    # -- persistence ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "cfg": self.cfg, "calibrated": self.calibrated}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TfidfBaseline":
        import joblib

        blob = joblib.load(path)
        model = cls(blob["cfg"])
        model.pipeline = blob["pipeline"]
        model.calibrated = blob.get("calibrated", False)
        return model


def default_grid(kind: str) -> list[dict]:
    """Validation-set search space for each baseline family."""
    if kind == "logreg":
        return [
            {"C": c, "ngram_range": ng, "min_df": 3}
            for c in (0.25, 1.0, 4.0)
            for ng in ((1, 1), (1, 2))
        ]
    if kind == "random_forest":
        return [
            {"n_estimators": 400, "max_depth": d, "min_samples_leaf": leaf}
            for d in (None, 24)
            for leaf in (1, 3)
        ]
    if kind == "xgboost":
        return [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in (300,)
            for d in (4, 6)
            for lr in (0.1, 0.2)
        ]
    return [{}]
