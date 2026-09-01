"""Tests for the evaluation metrics, faithfulness machinery and significance tests.

Where a metric has a known closed form or a known limiting behaviour, the test
asserts against that rather than against a recorded output, so a refactor that
silently changes the definition of a reported number will fail here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from medxplain.eval.faithfulness import FaithfulnessConfig, FaithfulnessEvaluator, summarise_faithfulness
from medxplain.eval.metrics import (
    aggregate_seeds,
    best_f1_threshold,
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    reliability_curve,
    threshold_at_recall,
)
from medxplain.eval.plausibility import krippendorff_alpha_ordinal, rationale_metrics, score_review
from medxplain.eval.significance import (
    delong_test,
    holm_bonferroni,
    paired_bootstrap_test,
    seed_level_test,
)
from medxplain.eval.stability import compare_explanations, distractor_mass
from medxplain.explain import build_explainer
from medxplain.explain.payload import Explanation, TokenAttribution, WordAttribution


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.15).astype(int)
    # a decent but imperfect scorer
    p = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.22, y.size), 0.001, 0.999)
    return y, p


class TestMetrics:
    def test_perfect_separation_gives_auroc_one(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        assert classification_metrics(y, p)["auroc"] == pytest.approx(1.0)

    def test_random_scores_give_auroc_near_half(self):
        rng = np.random.default_rng(2)
        y = (rng.random(4000) < 0.3).astype(int)
        p = rng.random(4000)
        assert abs(classification_metrics(y, p)["auroc"] - 0.5) < 0.05

    def test_auprc_baseline_equals_prevalence(self, imbalanced):
        y, p = imbalanced
        out = classification_metrics(y, p)
        assert out["auprc_baseline"] == pytest.approx(y.mean())
        assert out["auprc"] > out["auprc_baseline"], "scorer should beat the no-skill floor"

    def test_confusion_counts_sum_to_n(self, imbalanced):
        y, p = imbalanced
        out = classification_metrics(y, p)
        c = out["confusion"]
        assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == y.size

    def test_best_f1_threshold_is_at_least_as_good_as_half(self, imbalanced):
        from sklearn.metrics import f1_score

        y, p = imbalanced
        thr = best_f1_threshold(y, p)
        assert f1_score(y, (p >= thr).astype(int)) >= f1_score(y, (p >= 0.5).astype(int)) - 1e-9

    def test_threshold_at_recall_achieves_the_target(self, imbalanced):
        from sklearn.metrics import recall_score

        y, p = imbalanced
        for target in (0.5, 0.8, 0.9):
            thr = threshold_at_recall(y, p, target)
            assert recall_score(y, (p >= thr).astype(int)) >= target - 1e-9

    def test_perfectly_calibrated_scores_have_near_zero_ece(self):
        rng = np.random.default_rng(3)
        p = rng.random(20000)
        y = (rng.random(20000) < p).astype(int)
        assert expected_calibration_error(y, p) < 0.02

    def test_brier_skill_score_is_zero_for_the_base_rate(self, imbalanced):
        y, _ = imbalanced
        constant = np.full(y.size, y.mean())
        assert classification_metrics(y, constant)["brier_skill_score"] == pytest.approx(0.0, abs=1e-9)

    def test_reliability_curve_bins_cover_all_points(self, imbalanced):
        y, p = imbalanced
        curve = reliability_curve(y, p, n_bins=10)
        assert sum(b["count"] for b in curve["bins"]) == y.size

    def test_bootstrap_ci_brackets_the_point_estimate(self, imbalanced):
        y, p = imbalanced
        ci = bootstrap_ci(y, p, n_bootstrap=200, seed=1)
        point = classification_metrics(y, p)["auroc"]
        assert ci["auroc"]["lo"] <= point <= ci["auroc"]["hi"]

    def test_aggregate_seeds_computes_mean_and_std(self):
        runs = [{"auroc": 0.80}, {"auroc": 0.84}, {"auroc": 0.82}]
        agg = aggregate_seeds(runs, keys=("auroc",))
        assert agg["auroc"]["mean"] == pytest.approx(0.82)
        assert agg["auroc"]["std"] == pytest.approx(np.std([0.80, 0.84, 0.82], ddof=1))
        assert agg["auroc"]["n_seeds"] == 3


class TestSignificance:
    def test_delong_auc_matches_sklearn(self, imbalanced):
        from sklearn.metrics import roc_auc_score

        y, p = imbalanced
        other = np.clip(p + np.random.default_rng(4).normal(0, 0.05, p.size), 0, 1)
        out = delong_test(y, p, other)
        assert out["auc_a"] == pytest.approx(roc_auc_score(y, p), abs=1e-6)
        assert out["auc_b"] == pytest.approx(roc_auc_score(y, other), abs=1e-6)

    def test_identical_predictions_are_not_significant(self, imbalanced):
        y, p = imbalanced
        assert delong_test(y, p, p.copy())["p_value"] > 0.99

    def test_clearly_better_model_is_significant(self, imbalanced):
        y, p = imbalanced
        rng = np.random.default_rng(5)
        worse = np.clip(0.3 * p + 0.7 * rng.random(p.size), 0, 1)
        assert delong_test(y, p, worse)["p_value"] < 0.05
        assert paired_bootstrap_test(y, p, worse, n_bootstrap=300)["p_value"] < 0.05

    def test_paired_bootstrap_ci_contains_the_difference(self, imbalanced):
        y, p = imbalanced
        rng = np.random.default_rng(6)
        other = np.clip(p + rng.normal(0, 0.1, p.size), 0, 1)
        out = paired_bootstrap_test(y, p, other, n_bootstrap=300)
        assert out["ci95"][0] <= out["difference"] <= out["ci95"][1]

    def test_seed_level_test_reports_effect_size(self):
        out = seed_level_test([0.84, 0.85, 0.86], [0.80, 0.81, 0.82])
        assert out["difference"] == pytest.approx(0.04, abs=1e-6)
        assert out["p_value"] < 0.05
        assert out["cohens_d"] > 1

    def test_seed_level_test_handles_a_single_seed(self):
        out = seed_level_test([0.8], [0.7])
        assert np.isnan(out["p_value"])

    def test_holm_is_monotone_and_conservative(self):
        adjusted = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04})["results"]
        assert adjusted["a"]["p_adjusted"] <= adjusted["b"]["p_adjusted"] <= adjusted["c"]["p_adjusted"]
        for name, block in adjusted.items():
            assert block["p_adjusted"] >= block["p_raw"]


class TestFaithfulness:
    def test_metrics_have_the_expected_shape_and_ranges(self, tiny_model, sample_note):
        model, tokenizer = tiny_model
        explainer = build_explainer("attention_rollout", model, tokenizer, device="cpu", max_length=64)
        explanation = explainer.explain(sample_note)

        evaluator = FaithfulnessEvaluator(
            model, tokenizer, FaithfulnessConfig(max_length=64, n_random_controls=2), device="cpu"
        )
        out = evaluator.evaluate(sample_note, explanation)

        assert 0.0 <= out["p_base"] <= 1.0
        assert -1.0 <= out["comprehensiveness"] <= 1.0
        assert -1.0 <= out["sufficiency"] <= 1.0
        assert len(out["per_k"]) == len(FaithfulnessConfig().k_fractions)
        assert "comprehensiveness_random" in out

    def test_deletion_and_insertion_curves_are_complementary_at_the_ends(self, tiny_model, sample_note):
        model, tokenizer = tiny_model
        explainer = build_explainer("attention_rollout", model, tokenizer, device="cpu", max_length=64)
        explanation = explainer.explain(sample_note)
        evaluator = FaithfulnessEvaluator(model, tokenizer, FaithfulnessConfig(max_length=64), device="cpu")

        curves = evaluator.deletion_insertion_curves(sample_note, explanation)
        # at fraction 0 nothing is deleted, so deletion starts at the base prediction;
        # at fraction 1 everything is retained, so insertion ends there
        assert curves["deletion"][0] == pytest.approx(curves["insertion"][-1], abs=1e-5)
        assert len(curves["deletion"]) == len(curves["fractions"])
        assert 0.0 <= curves["deletion_auc"] <= 1.0

    def test_an_oracle_explanation_beats_a_reversed_one(self, tiny_model, sample_note):
        """Sanity check on the metric's direction, not on any method's quality."""
        model, tokenizer = tiny_model
        explainer = build_explainer("occlusion", model, tokenizer, device="cpu", max_length=64, max_words=30)
        good = explainer.explain(sample_note)

        reversed_tokens = [
            TokenAttribution(index=t.index, token=t.token, score=-t.score, start=t.start, end=t.end)
            for t in good.tokens
        ]
        bad = Explanation(
            method="reversed", tokens=reversed_tokens, words=good.words,
            probability=good.probability, predicted_label=good.predicted_label,
            meta={"signed": True},
        )

        evaluator = FaithfulnessEvaluator(
            model, tokenizer, FaithfulnessConfig(max_length=64, n_random_controls=1), device="cpu"
        )
        assert (
            evaluator.evaluate(sample_note, good)["comprehensiveness"]
            >= evaluator.evaluate(sample_note, bad)["comprehensiveness"]
        )

    def test_summary_reports_mean_and_std(self):
        rows = [
            {"comprehensiveness": 0.2, "sufficiency": 0.1, "comprehensiveness_random": 0.05,
             "sufficiency_random": 0.3, "comprehensiveness_over_random": 0.15,
             "sufficiency_over_random": 0.2},
            {"comprehensiveness": 0.3, "sufficiency": 0.05, "comprehensiveness_random": 0.06,
             "sufficiency_random": 0.32, "comprehensiveness_over_random": 0.24,
             "sufficiency_over_random": 0.27},
        ]
        summary = summarise_faithfulness(rows)
        assert summary["n_notes"] == 2
        assert summary["comprehensiveness"]["mean"] == pytest.approx(0.25)


class TestStability:
    def _explanation(self, pairs, method="test", signed=True):
        words = [
            WordAttribution(text=t, score=s, start=i * 10, end=i * 10 + len(t))
            for i, (t, s) in enumerate(pairs)
        ]
        return Explanation(method=method, tokens=[], words=words, probability=0.5,
                           predicted_label=1, meta={"signed": signed})

    def test_identical_explanations_agree_perfectly(self):
        a = self._explanation([("shock", 0.9), ("stable", -0.4), ("patient", 0.1)])
        out = compare_explanations(a, a, top_k=3)
        assert out["spearman"] == pytest.approx(1.0)
        assert out["top_k_jaccard"] == pytest.approx(1.0)

    def test_reversed_ordering_gives_negative_correlation(self):
        a = self._explanation([("a", 0.9), ("b", 0.5), ("c", 0.1)])
        b = self._explanation([("a", 0.1), ("b", 0.5), ("c", 0.9)])
        assert compare_explanations(a, b, top_k=3)["spearman"] < 0

    def test_disjoint_top_k_gives_zero_jaccard(self):
        a = self._explanation([("x", 1.0), ("y", 0.0), ("z", 0.0)])
        b = self._explanation([("x", 0.0), ("y", 0.0), ("z", 1.0)])
        assert compare_explanations(a, b, top_k=1)["top_k_jaccard"] == pytest.approx(0.0)

    def test_distractor_mass_is_a_fraction(self):
        explanation = self._explanation([("shock", 0.8), ("parking", 0.2)])
        mass = distractor_mass(explanation, "parking validation")
        assert mass == pytest.approx(0.2)

    def test_distractor_mass_is_zero_when_untouched(self):
        explanation = self._explanation([("shock", 1.0)])
        assert distractor_mass(explanation, "parking validation") == 0.0


class TestPlausibility:
    def _explanation(self, words, signed=True):
        attributions = []
        cursor = 0
        for text, score in words:
            attributions.append(
                WordAttribution(text=text, score=score, start=cursor, end=cursor + len(text))
            )
            cursor += len(text) + 1
        return Explanation(method="test", tokens=[], words=attributions, probability=0.8,
                           predicted_label=1, meta={"signed": signed})

    def test_perfect_rationale_scores_high_f1(self):
        text = "patient in septic shock today"
        gold = [{"start": 11, "end": 23, "text": "septic shock", "mortality_weight": 1.6,
                 "readmission_weight": 0.3}]
        explanation = self._explanation(
            [("patient", 0.0), ("in", 0.0), ("septic", 0.9), ("shock", 0.95), ("today", 0.0)]
        )
        out = rationale_metrics(explanation, text, gold, top_k_fraction=0.4)
        assert out["token_f1"] == pytest.approx(1.0)
        assert out["rationale_auprc"] > 0.9

    def test_anti_correlated_rationale_scores_low(self):
        text = "patient in septic shock today"
        gold = [{"start": 11, "end": 23, "text": "septic shock", "mortality_weight": 1.6,
                 "readmission_weight": 0.3}]
        explanation = self._explanation(
            [("patient", 0.9), ("in", 0.85), ("septic", 0.0), ("shock", 0.0), ("today", 0.8)]
        )
        assert rationale_metrics(explanation, text, gold, top_k_fraction=0.4)["token_f1"] < 0.4

    def test_sign_agreement_is_nan_for_unsigned_methods(self):
        text = "patient in septic shock today"
        gold = [
            {"start": 0, "end": 7, "text": "patient", "mortality_weight": -0.8, "readmission_weight": 0.0},
            {"start": 11, "end": 23, "text": "septic shock", "mortality_weight": 1.6, "readmission_weight": 0.3},
        ]
        explanation = self._explanation(
            [("patient", 0.5), ("in", 0.1), ("septic", 0.9), ("shock", 0.9), ("today", 0.1)], signed=False
        )
        assert np.isnan(rationale_metrics(explanation, text, gold)["sign_agreement"])


class TestInterRaterAgreement:
    def test_full_agreement_gives_alpha_one(self):
        ratings = np.array([[5, 4, 3, 2], [5, 4, 3, 2]], dtype=float)
        assert krippendorff_alpha_ordinal(ratings) == pytest.approx(1.0, abs=1e-9)

    def test_missing_ratings_are_tolerated(self):
        ratings = np.array([[5, 4, np.nan, 2], [5, np.nan, 3, 2]], dtype=float)
        assert not np.isnan(krippendorff_alpha_ordinal(ratings))

    def test_score_review_unblinds_and_aggregates(self):
        answer_key = [{"note_id": "n1", "key": {"A": "shap", "B": "attention_rollout"}}]
        responses = [
            {"rater": "R1", "note_id": "n1", "label": "A", "q1_sensibility": 5},
            {"rater": "R1", "note_id": "n1", "label": "B", "q1_sensibility": 2},
            {"rater": "R2", "note_id": "n1", "label": "A", "q1_sensibility": 4},
            {"rater": "R2", "note_id": "n1", "label": "B", "q1_sensibility": 3},
        ]
        out = score_review(responses, answer_key)
        assert out["methods"]["shap"]["mean"] == pytest.approx(4.5)
        assert out["methods"]["attention_rollout"]["mean"] == pytest.approx(2.5)
        assert out["inter_rater"]["n_raters"] == 2
