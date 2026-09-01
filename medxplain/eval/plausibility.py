"""Explanation plausibility: does the highlighted text make clinical sense?

Faithfulness and plausibility are different properties and can diverge — an
explanation can be a perfectly faithful description of a model that learned a
shortcut, and therefore be clinically nonsensical. Both are measured.

Two routes are implemented:

**Automatic (prototype corpus only).** The synthetic generator records the exact
character span and true log-odds weight of every risk factor it injected, so
gold rationales exist for free. Explanations are scored against them with the
ERASER rationale metrics — token-level F1 and AUPRC — plus two measures specific
to this setup: *weight correlation* (do the highest-scoring phrases correspond
to the largest true weights?) and *sign agreement* (does the explainer mark
protective findings as protective, or merely as "important"?). Sign agreement is
the interesting one, because attention cannot express direction at all.

**Human (any corpus).** :func:`build_review_packet` renders a blinded review
form: the note, the model's risk score, and the top highlighted phrases from
each method with method names replaced by neutral labels and the order shuffled
per note. Raters score clinical sensibility on a 1-5 Likert scale.
:func:`score_review` computes per-method means and inter-rater agreement
(Krippendorff's alpha for ordinal data, plus pairwise weighted kappa), so a
2-3 rater study can be reported defensibly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from medxplain.explain.payload import Explanation


# --------------------------------------------------------------------------------------
# Automatic scoring against gold rationale spans
# --------------------------------------------------------------------------------------


def gold_word_mask(text: str, gold_spans: Sequence[dict], words: Sequence) -> np.ndarray:
    """Boolean mask over words: is this word inside any gold rationale span?"""
    mask = np.zeros(len(words), dtype=bool)
    for i, word in enumerate(words):
        for span in gold_spans:
            if word.start < span["end"] and word.end > span["start"]:
                mask[i] = True
                break
    return mask


def gold_word_weights(text: str, gold_spans: Sequence[dict], words: Sequence, key: str) -> np.ndarray:
    """Per-word true log-odds weight (0 outside every gold span)."""
    weights = np.zeros(len(words), dtype=float)
    for i, word in enumerate(words):
        for span in gold_spans:
            if word.start < span["end"] and word.end > span["start"]:
                weights[i] = span[key]
                break
    return weights


def rationale_metrics(
    explanation: Explanation,
    text: str,
    gold_spans: Sequence[dict],
    weight_key: str = "mortality_weight",
    top_k_fraction: float = 0.20,
) -> dict:
    """Score one explanation against the gold rationales of one note."""
    words = explanation.words
    if not words or not gold_spans:
        return {"error": "no words or no gold spans"}

    gold = gold_word_mask(text, gold_spans, words)
    weights = gold_word_weights(text, gold_spans, words, weight_key)
    scores = np.asarray([w.score for w in words], dtype=float)

    if gold.sum() == 0:
        return {"error": "gold spans did not align to any word"}

    # hard rationale: take the top fraction of words by absolute importance
    k = max(1, int(round(top_k_fraction * len(words))))
    top_idx = np.argsort(-np.abs(scores))[:k]
    predicted = np.zeros(len(words), dtype=bool)
    predicted[top_idx] = True

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, average="binary", zero_division=0
    )

    out: dict = {
        "method": explanation.method,
        "n_words": len(words),
        "n_gold_words": int(gold.sum()),
        "top_k_fraction": top_k_fraction,
        "token_precision": float(precision),
        "token_recall": float(recall),
        "token_f1": float(f1),
        # soft rationale: rank all words by importance, score against the gold mask
        "rationale_auprc": float(average_precision_score(gold, np.abs(scores)))
        if gold.sum() < len(words) else float("nan"),
        "rationale_auprc_baseline": float(gold.mean()),
    }

    # do the biggest attributions land on the biggest true weights?
    gold_only = gold
    if gold_only.sum() >= 3:
        gw, sw = weights[gold_only], scores[gold_only]
        if gw.std() > 1e-12 and sw.std() > 1e-12:
            from scipy.stats import spearmanr

            out["weight_spearman"] = float(spearmanr(gw, sw).statistic)
        else:
            out["weight_spearman"] = float("nan")

        # sign agreement: only meaningful for signed explainers
        if explanation.meta.get("signed", False):
            nonzero = np.abs(gw) > 1e-9
            if nonzero.any():
                agree = np.sign(gw[nonzero]) == np.sign(sw[nonzero])
                out["sign_agreement"] = float(agree.mean())
        else:
            out["sign_agreement"] = float("nan")
    else:
        out["weight_spearman"] = float("nan")
        out["sign_agreement"] = float("nan")

    # how much importance mass is spent outside any true risk factor
    total = np.abs(scores).sum()
    out["off_rationale_mass"] = float(np.abs(scores[~gold]).sum() / total) if total > 0 else float("nan")
    return out


def summarise_rationale_metrics(results: list[dict]) -> dict:
    """Mean +/- std over notes."""
    usable = [r for r in results if "error" not in r]
    keys = (
        "token_precision", "token_recall", "token_f1", "rationale_auprc",
        "rationale_auprc_baseline", "weight_spearman", "sign_agreement", "off_rationale_mass",
    )
    out: dict = {"n_notes": len(usable), "n_skipped": len(results) - len(usable)}
    for key in keys:
        vals = np.asarray(
            [r[key] for r in usable if key in r and r[key] == r[key]], dtype=float
        )
        if vals.size:
            out[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "n": int(vals.size),
            }
    return out


# --------------------------------------------------------------------------------------
# Human review harness
# --------------------------------------------------------------------------------------

LIKERT_RUBRIC: dict[int, str] = {
    5: "All highlighted phrases are clinically relevant to the predicted risk; "
       "nothing important is missed.",
    4: "Most highlighted phrases are relevant; minor irrelevant highlights or "
       "one missed factor.",
    3: "Mixed - some genuinely relevant phrases alongside clearly irrelevant ones.",
    2: "Mostly irrelevant highlights; at best incidental overlap with the real drivers.",
    1: "Highlights are clinically meaningless or actively misleading.",
}

REVIEW_INSTRUCTIONS = """\
For each note you will see the model's predicted risk and several sets of
highlighted phrases, labelled A, B, C ... The labels are shuffled per note and do
not correspond to the same method across notes.

For each labelled set, answer:
  Q1 (1-5) Clinical sensibility: do the highlighted phrases correspond to the
           factors a clinician would actually weigh for this outcome?
  Q2 (1-5) Completeness: are the important drivers present in the note covered?
  Q3 (yes/no) Would you be willing to show this explanation to a colleague
           alongside the risk score?

Score the explanation, not the prediction: an explanation can be good even if you
disagree with the risk score, and vice versa.
"""


@dataclass
class ReviewItem:
    note_id: str
    text: str
    probability: float
    #: label -> list of highlighted phrases (blinded)
    candidates: dict[str, list[str]]
    #: label -> real method name, withheld from the rater
    key: dict[str, str]


def build_review_packet(
    notes: Sequence[dict],
    explanations_by_method: dict[str, list[Explanation]],
    top_k: int = 8,
    seed: int = 0,
) -> dict:
    """Blinded, shuffled review packet for a small clinician study."""
    rng = random.Random(seed)
    methods = sorted(explanations_by_method)
    items: list[ReviewItem] = []

    for i, note in enumerate(notes):
        labels = [chr(ord("A") + j) for j in range(len(methods))]
        shuffled = methods[:]
        rng.shuffle(shuffled)
        candidates, key = {}, {}
        for label, method in zip(labels, shuffled):
            explanation = explanations_by_method[method][i]
            phrases = [
                w.text for w in sorted(explanation.words, key=lambda w: abs(w.score), reverse=True)[:top_k]
            ]
            candidates[label] = phrases
            key[label] = method
        items.append(
            ReviewItem(
                note_id=note.get("note_id", f"note_{i}"),
                text=note["text"],
                probability=float(explanations_by_method[methods[0]][i].probability),
                candidates=candidates,
                key=key,
            )
        )

    return {
        "instructions": REVIEW_INSTRUCTIONS,
        "rubric": LIKERT_RUBRIC,
        "items": [
            {
                "note_id": it.note_id,
                "text": it.text,
                "probability": it.probability,
                "candidates": it.candidates,
            }
            for it in items
        ],
        "answer_key": [{"note_id": it.note_id, "key": it.key} for it in items],
        "n_items": len(items),
        "methods": methods,
    }


def krippendorff_alpha_ordinal(ratings: np.ndarray) -> float:
    """Krippendorff's alpha for ordinal data.

    ``ratings`` is (n_raters, n_items) with NaN for missing. Chosen over Fleiss'
    kappa because it handles missing ratings and respects the ordering of a
    Likert scale — treating 1-vs-5 and 4-vs-5 disagreements as equally severe
    (as kappa does) understates agreement on ordinal rubrics.
    """
    ratings = np.asarray(ratings, dtype=float)
    values = ratings[~np.isnan(ratings)]
    if values.size == 0:
        return float("nan")
    levels = np.unique(values)
    if levels.size < 2:
        return 1.0

    # ordinal difference function over the observed levels
    counts_by_level = {v: float((values == v).sum()) for v in levels}
    n_total = values.size
    index = {v: i for i, v in enumerate(levels)}

    def delta(a: float, b: float) -> float:
        i, j = sorted((index[a], index[b]))
        interior = sum(counts_by_level[levels[g]] for g in range(i, j + 1))
        correction = (counts_by_level[a] + counts_by_level[b]) / 2.0
        return (interior - correction) ** 2

    observed_num, observed_den = 0.0, 0.0
    for item in range(ratings.shape[1]):
        column = ratings[:, item]
        column = column[~np.isnan(column)]
        m = column.size
        if m < 2:
            continue
        for a, b in combinations(column, 2):
            observed_num += 2 * delta(a, b)  # both orderings
        observed_den += m - 1
    if observed_den == 0:
        return float("nan")
    observed_disagreement = observed_num / (observed_den * 2)

    expected = 0.0
    for a in levels:
        for b in levels:
            expected += counts_by_level[a] * counts_by_level[b] * delta(a, b)
    expected /= n_total * (n_total - 1)

    if expected == 0:
        return 1.0
    return float(1.0 - observed_disagreement / expected)


def score_review(
    responses: list[dict],
    answer_key: list[dict],
    question: str = "q1_sensibility",
) -> dict:
    """Aggregate a completed review.

    ``responses`` items look like
    ``{"rater": "R1", "note_id": ..., "label": "A", "q1_sensibility": 4, ...}``.
    Labels are un-blinded via ``answer_key`` before aggregation.
    """
    key_lookup = {entry["note_id"]: entry["key"] for entry in answer_key}
    by_method: dict[str, list[float]] = {}
    by_rater_method: dict[tuple[str, str], dict[str, float]] = {}

    for response in responses:
        note_id, label = response["note_id"], response["label"]
        method = key_lookup.get(note_id, {}).get(label)
        if method is None or question not in response:
            continue
        value = float(response[question])
        by_method.setdefault(method, []).append(value)
        by_rater_method.setdefault((response["rater"], method), {})[note_id] = value

    summary: dict = {"question": question, "methods": {}}
    for method, values in sorted(by_method.items()):
        arr = np.asarray(values, dtype=float)
        summary["methods"][method] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "n_ratings": int(arr.size),
        }

    raters = sorted({r["rater"] for r in responses})
    note_ids = sorted({r["note_id"] for r in responses})
    alphas: dict[str, float] = {}
    for method in sorted(by_method):
        matrix = np.full((len(raters), len(note_ids)), np.nan)
        for ri, rater in enumerate(raters):
            cell = by_rater_method.get((rater, method), {})
            for ni, note_id in enumerate(note_ids):
                if note_id in cell:
                    matrix[ri, ni] = cell[note_id]
        alphas[method] = krippendorff_alpha_ordinal(matrix)

    summary["inter_rater"] = {
        "n_raters": len(raters),
        "raters": raters,
        "krippendorff_alpha_by_method": alphas,
    }

    # paired comparison between methods on the notes both were rated on
    if len(by_method) >= 2:
        from scipy import stats as _stats

        pairs: dict[str, dict] = {}
        for a, b in combinations(sorted(by_method), 2):
            va, vb = np.asarray(by_method[a]), np.asarray(by_method[b])
            n = min(va.size, vb.size)
            if n >= 3:
                try:
                    p = float(_stats.wilcoxon(va[:n], vb[:n]).pvalue)
                except ValueError:
                    p = float("nan")
                pairs[f"{a}_vs_{b}"] = {
                    "mean_difference": float(va[:n].mean() - vb[:n].mean()),
                    "wilcoxon_p": p,
                    "n": int(n),
                }
        summary["pairwise"] = pairs
    return summary
