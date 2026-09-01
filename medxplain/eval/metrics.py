"""Predictive-performance metrics.

AUPRC is treated as the headline number rather than AUROC. With a ~15% positive
rate, AUROC is inflated by the ease of ranking the abundant negatives, and two
models with the same AUROC can differ substantially in the precision a clinician
would actually see at a usable alert threshold. Both are reported; AUPRC is the
one the discussion should lead with, and its no-skill floor (the prevalence) is
carried in the output so a reader can judge lift rather than the raw number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict:
    """Full metric set for one model on one split.

    ``threshold`` defaults to the F1-optimal operating point *chosen on this
    split*, which would be optimistic if reported as-is; the calling scripts
    select the threshold on validation and pass it in explicitly for the test
    set. The chosen value is always echoed back in the result.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if threshold is None:
        threshold = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    prevalence = float(y_true.mean())
    out = {
        "n": int(y_true.size),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auprc_baseline": prevalence,
        "auprc_lift": float(average_precision_score(y_true, y_prob) / prevalence) if prevalence else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "brier_skill_score": _brier_skill(y_true, y_prob),
        "specificity": _specificity(y_true, y_pred),
        "balanced_accuracy": float(
            (recall_score(y_true, y_pred, zero_division=0) + _specificity(y_true, y_pred)) / 2
        ),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # clinically meaningful operating points: what recall costs in false alarms
    out["operating_points"] = operating_points(y_true, y_prob)

    if n_bootstrap:
        out["ci"] = bootstrap_ci(y_true, y_prob, n_bootstrap=n_bootstrap, seed=seed)
    return out


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def _brier_skill(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score relative to always predicting the base rate. 0 = no better."""
    base = float(y_true.mean())
    reference = brier_score_loss(y_true, np.full_like(y_prob, base))
    if reference == 0:
        return float("nan")
    return float(1.0 - brier_score_loss(y_true, y_prob) / reference)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising F1, found exactly from the PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # precision_recall_curve returns one more point than thresholds
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def threshold_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.8) -> float:
    """Lowest threshold that still achieves ``target_recall``."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    viable = np.where(recall[:-1] >= target_recall)[0]
    if viable.size == 0:
        return 0.0
    return float(thresholds[viable[-1]])


def operating_points(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    """Precision/alert-rate at recall levels a clinical service would consider."""
    points = []
    for target in (0.50, 0.70, 0.80, 0.90):
        thr = threshold_at_recall(y_true, y_prob, target)
        pred = (y_prob >= thr).astype(int)
        points.append(
            {
                "target_recall": target,
                "threshold": float(thr),
                "achieved_recall": float(recall_score(y_true, pred, zero_division=0)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "alert_rate": float(pred.mean()),
            }
        )
    return points


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Binned |confidence - accuracy|, weighted by bin size."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Points for the reliability diagram (Phase 5 figure)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            continue
        rows.append(
            {
                "bin": b,
                "mean_predicted": float(y_prob[in_bin].mean()),
                "observed_rate": float(y_true[in_bin].mean()),
                "count": int(in_bin.sum()),
            }
        )
    return {"bins": rows, "ece": expected_calibration_error(y_true, y_prob, n_bins)}


def curve_points(y_true: np.ndarray, y_prob: np.ndarray, max_points: int = 300) -> dict:
    """Downsampled ROC and PR curves for plotting without shipping full arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    def thin(a: np.ndarray, b: np.ndarray) -> tuple[list, list]:
        if len(a) <= max_points:
            return a.tolist(), b.tolist()
        idx = np.linspace(0, len(a) - 1, max_points).astype(int)
        return a[idx].tolist(), b[idx].tolist()

    x_roc, y_roc = thin(fpr, tpr)
    x_pr, y_pr = thin(recall, precision)
    return {
        "roc": {"fpr": x_roc, "tpr": y_roc, "auc": float(roc_auc_score(y_true, y_prob))},
        "pr": {
            "recall": x_pr,
            "precision": y_pr,
            "ap": float(average_precision_score(y_true, y_prob)),
            "baseline": float(y_true.mean()),
        },
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: tuple[str, ...] = ("auroc", "auprc", "f1"),
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap confidence intervals over test-set resamples.

    Stratified by class so every resample contains positives — with a rare
    outcome an unstratified bootstrap produces degenerate folds where AUROC is
    undefined, which silently biases the interval.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    samples: dict[str, list[float]] = {m: [] for m in metrics}
    threshold = best_f1_threshold(y_true, y_prob)

    for _ in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        yt, yp = y_true[idx], y_prob[idx]
        for m in metrics:
            if m == "auroc":
                samples[m].append(roc_auc_score(yt, yp))
            elif m == "auprc":
                samples[m].append(average_precision_score(yt, yp))
            elif m == "f1":
                samples[m].append(f1_score(yt, (yp >= threshold).astype(int), zero_division=0))

    out = {}
    for m, vals in samples.items():
        arr = np.asarray(vals)
        out[m] = {
            "mean": float(arr.mean()),
            "lo": float(np.percentile(arr, 100 * alpha / 2)),
            "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n_bootstrap": n_bootstrap,
        }
    return out


def aggregate_seeds(runs: list[dict], keys: tuple[str, ...] = ("auroc", "auprc", "f1", "brier")) -> dict:
    """Mean +/- std across random seeds, the format the main table requires."""
    out: dict = {}
    for key in keys:
        values = [r[key] for r in runs if key in r and r[key] == r[key]]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
            "values": [float(v) for v in arr],
        }
    return out
