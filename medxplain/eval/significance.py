"""Statistical comparison between models.

"Model A got 0.842 and model B got 0.831" is not a result. Three tests are
provided, each appropriate to a different comparison the paper makes:

* :func:`delong_test` — the standard parametric test for two correlated AUROCs
  measured on the *same* test set (DeLong et al., 1988). Correlated is the
  operative word: an unpaired test ignores that both models saw identical
  patients and is badly over-conservative.
* :func:`paired_bootstrap_test` — a distribution-free paired test that works for
  any metric, including AUPRC where no closed-form variance exists. This is the
  workhorse for the main table.
* :func:`permutation_test` — an exact-in-the-limit alternative that swaps model
  predictions between the two systems under the null of exchangeability.

For the multi-seed setting there is also :func:`seed_level_test`, which treats
each seed as an observation. It answers a different question (is the *training
procedure* better, accounting for initialisation variance?) and is the honest
test when both models are retrained per seed.

Because the paper compares one proposed model against several baselines,
:func:`holm_bonferroni` is provided to control the family-wise error rate; the
raw and adjusted p-values are both reported.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

_METRIC_FNS = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "f1": lambda y, p: f1_score(y, (np.asarray(p) >= 0.5).astype(int), zero_division=0),
}


# --------------------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, the core of the DeLong covariance estimate."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(predictions: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong: returns per-model AUC and the covariance matrix."""
    pos = predictions[:, :n_pos]
    neg = predictions[:, n_pos:]
    k, m = predictions.shape[0], pos.shape[1]
    n = neg.shape[1]

    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(predictions[r]) for r in range(k)])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx, sy = sx.reshape(1, 1), sy.reshape(1, 1)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Two-sided DeLong test for the difference between two correlated AUROCs."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)  # positives first, as fast-DeLong expects
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    predictions = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])

    aucs, cov = _fast_delong(predictions, n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])

    if var <= 0:
        return {
            "auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": diff,
            "z": float("nan"), "p_value": 1.0, "test": "delong",
            "note": "degenerate variance (identical or near-identical predictions)",
        }

    z = diff / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    se = float(np.sqrt(var))
    return {
        "test": "delong",
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": diff,
        "std_error": se,
        "ci95": [diff - 1.96 * se, diff + 1.96 * se],
        "z": float(z),
        "p_value": p,
    }


# --------------------------------------------------------------------------------------
# Resampling tests
# --------------------------------------------------------------------------------------


def paired_bootstrap_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric: str = "auprc",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired stratified bootstrap of the metric difference.

    Both models are evaluated on the *same* resampled patients each iteration,
    which removes the between-patient variance that dominates an unpaired
    comparison. The p-value is the two-sided proportion of resamples where the
    difference crosses zero.
    """
    fn = _METRIC_FNS[metric]
    y_true = np.asarray(y_true).astype(int)
    prob_a, prob_b = np.asarray(prob_a, float), np.asarray(prob_b, float)

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    observed = float(fn(y_true, prob_a) - fn(y_true, prob_b))
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, pos_idx.size, replace=True),
                rng.choice(neg_idx, neg_idx.size, replace=True),
            ]
        )
        yt = y_true[idx]
        diffs[i] = fn(yt, prob_a[idx]) - fn(yt, prob_b[idx])

    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "test": "paired_bootstrap",
        "metric": metric,
        "metric_a": float(fn(y_true, prob_a)),
        "metric_b": float(fn(y_true, prob_b)),
        "difference": observed,
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_value": min(1.0, p),
        "n_bootstrap": n_bootstrap,
    }


def permutation_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric: str = "auprc",
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired permutation test: randomly swap the two models' scores per patient."""
    fn = _METRIC_FNS[metric]
    y_true = np.asarray(y_true).astype(int)
    a, b = np.asarray(prob_a, float), np.asarray(prob_b, float)
    observed = abs(float(fn(y_true, a) - fn(y_true, b)))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        swap = rng.random(a.size) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        if abs(fn(y_true, pa) - fn(y_true, pb)) >= observed:
            count += 1
    return {
        "test": "paired_permutation",
        "metric": metric,
        "difference": float(fn(y_true, a) - fn(y_true, b)),
        "p_value": float((count + 1) / (n_permutations + 1)),
        "n_permutations": n_permutations,
    }


def seed_level_test(values_a: list[float], values_b: list[float], paired: bool = True) -> dict:
    """Compare two systems across random seeds.

    Uses a paired t-test when the seeds correspond (seed i used for both
    systems), plus the non-parametric Wilcoxon signed-rank as a robustness check
    since 3-5 seeds is far too few to trust normality. Cohen's d is reported
    because with so few seeds a p-value alone is nearly uninformative.
    """
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    out: dict = {
        "test": "paired_t" if paired else "welch_t",
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "std_b": float(b.std(ddof=1)) if b.size > 1 else 0.0,
        "n_seeds": int(a.size),
        "difference": float(a.mean() - b.mean()),
    }
    if a.size < 2 or b.size < 2:
        out["p_value"] = float("nan")
        out["note"] = "at least two seeds per system are needed for a test"
        return out

    if paired and a.size == b.size:
        t = stats.ttest_rel(a, b)
        diff = a - b
        pooled = diff.std(ddof=1)
        out["cohens_d"] = float(diff.mean() / pooled) if pooled > 0 else float("inf")
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")  # all differences zero
    else:
        t = stats.ttest_ind(a, b, equal_var=False)
        out["test"] = "welch_t"
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out["cohens_d"] = float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("inf")

    out["t_statistic"] = float(t.statistic)
    out["p_value"] = float(t.pvalue)
    return out


# --------------------------------------------------------------------------------------
# Multiple-comparison control
# --------------------------------------------------------------------------------------


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down adjustment over a family of comparisons.

    Preferred over plain Bonferroni: uniformly more powerful at the same
    family-wise error rate, with no additional assumptions.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, dict] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotonicity
        adjusted[name] = {
            "p_raw": float(p),
            "p_adjusted": float(running),
            "significant": bool(running < alpha),
            "rank": i + 1,
        }
    return {"alpha": alpha, "n_comparisons": m, "results": adjusted}


def compare_against_baselines(
    y_true: np.ndarray,
    proposed: np.ndarray,
    baselines: dict[str, np.ndarray],
    metric: str = "auprc",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Run the full comparison of one proposed model against every baseline."""
    tests: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for name, probs in baselines.items():
        boot = paired_bootstrap_test(y_true, proposed, probs, metric=metric,
                                     n_bootstrap=n_bootstrap, seed=seed)
        entry = {"bootstrap": boot}
        if metric == "auroc":
            entry["delong"] = delong_test(y_true, proposed, probs)
        tests[name] = entry
        raw_p[name] = boot["p_value"]
    return {"metric": metric, "tests": tests, "holm": holm_bonferroni(raw_p)}
