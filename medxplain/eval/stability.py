"""Explanation stability under meaning-preserving edits.

A clinician who pastes the same note twice, once with "SOB" and once with
"shortness of breath", should not get a different set of highlighted phrases. An
explainer that fails this is unusable regardless of how it scores on
faithfulness, and instability is rarely reported in clinical XAI papers.

Three perturbation families are used, in increasing severity:

* ``near_duplicate`` - whitespace and casing only. The prediction should be
  essentially identical; any explanation change here is pure noise.
* ``paraphrase``     - abbreviation and synonym swaps that preserve clinical
  meaning. Word-level scores are compared on the *shared* vocabulary, since the
  surface forms differ by construction.
* ``distractor``     - a clinically irrelevant sentence is appended. Importance
  assigned to the inserted sentence is wasted attribution, and we report it
  directly as ``distractor_mass``.

Agreement is measured three ways because they answer different questions:
rank correlation (does the ordering survive?), top-k Jaccard (does the
clinician see the same highlights?), and prediction shift (did the model itself
move, which would make explanation change legitimate?).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from medxplain.explain.payload import Explanation

DISTRACTOR_SENTENCES: tuple[str, ...] = (
    "The patient's daughter brought in a home medication list which was reconciled by pharmacy.",
    "A dietary consultation was completed and educational materials were provided.",
    "The room was changed on the second hospital day for bed management reasons.",
    "Parking validation was provided to the family at the front desk.",
)


@dataclass
class StabilityConfig:
    top_k: int = 10
    perturbations: tuple[str, ...] = ("near_duplicate", "paraphrase", "distractor")
    seed: int = 0


def _word_score_map(explanation: Explanation, lowercase: bool = True) -> dict[str, float]:
    """Sum scores per surface word, so two notes can be compared by vocabulary."""
    scores: dict[str, float] = {}
    for word in explanation.words:
        key = word.text.lower().strip(".,;:()[]") if lowercase else word.text
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + word.score
    return scores


def _top_k_words(explanation: Explanation, k: int) -> set[str]:
    ranked = sorted(explanation.words, key=lambda w: abs(w.score), reverse=True)[:k]
    return {w.text.lower().strip(".,;:()[]") for w in ranked}


def compare_explanations(a: Explanation, b: Explanation, top_k: int = 10) -> dict:
    """Agreement between two explanations of (nearly) the same note."""
    map_a, map_b = _word_score_map(a), _word_score_map(b)
    shared = sorted(set(map_a) & set(map_b))

    result: dict = {
        "n_shared_words": len(shared),
        "vocabulary_overlap": (
            len(shared) / max(1, len(set(map_a) | set(map_b)))
        ),
        "probability_shift": abs(a.probability - b.probability),
    }

    if len(shared) >= 3:
        va = np.asarray([map_a[w] for w in shared])
        vb = np.asarray([map_b[w] for w in shared])
        # constant vectors make the correlation undefined rather than zero
        if va.std() > 1e-12 and vb.std() > 1e-12:
            result["spearman"] = float(spearmanr(va, vb).statistic)
            result["kendall_tau"] = float(kendalltau(va, vb).statistic)
            result["pearson"] = float(np.corrcoef(va, vb)[0, 1])
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            result["cosine"] = float(va @ vb / denom) if denom > 0 else float("nan")
        else:
            result["spearman"] = float("nan")
            result["kendall_tau"] = float("nan")
            result["pearson"] = float("nan")
            result["cosine"] = float("nan")
    else:
        result.update({"spearman": float("nan"), "kendall_tau": float("nan"),
                       "pearson": float("nan"), "cosine": float("nan")})

    ta, tb = _top_k_words(a, top_k), _top_k_words(b, top_k)
    union = ta | tb
    result["top_k"] = top_k
    result["top_k_jaccard"] = len(ta & tb) / len(union) if union else float("nan")
    result["top_k_overlap"] = len(ta & tb) / top_k
    return result


def distractor_mass(explanation: Explanation, inserted: str) -> float:
    """Fraction of total absolute importance landing on the inserted sentence.

    Zero is ideal. This is the cleanest single number for "is the explainer
    distracted by irrelevant but fluent clinical boilerplate".
    """
    inserted_words = {w.lower().strip(".,;:()[]") for w in inserted.split()}
    total = sum(abs(w.score) for w in explanation.words)
    if total == 0:
        return 0.0
    on_inserted = sum(
        abs(w.score) for w in explanation.words
        if w.text.lower().strip(".,;:()[]") in inserted_words
    )
    return float(on_inserted / total)


class StabilityEvaluator:
    """Runs the perturbation suite for one explainer."""

    def __init__(self, explain_fn: Callable[[str], Explanation], cfg: StabilityConfig | None = None):
        self.explain_fn = explain_fn
        self.cfg = cfg or StabilityConfig()

    def evaluate_note(self, text: str, index: int = 0) -> dict:
        from medxplain.data.synthetic import near_duplicate_note, paraphrase_note

        base = self.explain_fn(text)
        out: dict = {"method": base.method, "base_probability": base.probability}

        for kind in self.cfg.perturbations:
            if kind == "near_duplicate":
                variant = near_duplicate_note(text, seed=self.cfg.seed + index)
                inserted = None
            elif kind == "paraphrase":
                variant = paraphrase_note(text, seed=self.cfg.seed + index)
                inserted = None
            elif kind == "distractor":
                inserted = DISTRACTOR_SENTENCES[index % len(DISTRACTOR_SENTENCES)]
                variant = text.rstrip() + "\n" + inserted + "\n"
            else:
                raise ValueError(f"unknown perturbation: {kind}")

            if variant == text:
                # nothing changed (e.g. no paraphrasable phrase present); skip rather
                # than record a trivially perfect agreement that inflates the mean
                out[kind] = {"skipped": True, "reason": "perturbation was a no-op"}
                continue

            perturbed = self.explain_fn(variant)
            entry = compare_explanations(base, perturbed, top_k=self.cfg.top_k)
            if inserted is not None:
                entry["distractor_mass"] = distractor_mass(perturbed, inserted)
            out[kind] = entry
        return out

    def evaluate(self, texts: Sequence[str]) -> list[dict]:
        return [self.evaluate_note(t, i) for i, t in enumerate(texts)]


def summarise_stability(results: list[dict], perturbations: Sequence[str]) -> dict:
    """Mean +/- std of every agreement metric, per perturbation type."""
    summary: dict = {"n_notes": len(results)}
    for kind in perturbations:
        entries = [r[kind] for r in results if kind in r and not r[kind].get("skipped")]
        if not entries:
            summary[kind] = {"n": 0, "skipped": len(results)}
            continue
        block: dict = {"n": len(entries)}
        for metric in (
            "spearman", "kendall_tau", "cosine", "top_k_jaccard", "top_k_overlap",
            "probability_shift", "distractor_mass", "vocabulary_overlap",
        ):
            vals = np.asarray(
                [e[metric] for e in entries if metric in e and e[metric] == e[metric]], dtype=float
            )
            if vals.size:
                block[metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                }
        summary[kind] = block
    return summary
