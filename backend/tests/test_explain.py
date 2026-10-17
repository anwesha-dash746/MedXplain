"""Tests for the explanation methods and the shared payload contract.

These tests do not check that explanations are *good* — that is what the
faithfulness evaluation is for. They check that every method obeys the contract
the UI and the metrics depend on: correct character offsets, no attribution
leaking onto padding, signed methods actually signed, and IG's completeness
property holding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from medxplain.explain import build_explainer
from medxplain.explain.attention import attention_rollout
from medxplain.explain.payload import (
    Explanation,
    TokenAttribution,
    aggregate_to_words,
    encode_with_offsets,
    normalise,
)

ALL_METHODS = [
    "attention_rollout",
    "attention_raw",
    "attention_grad",
    "occlusion",
    "integrated_gradients",
]


def make_explainer(method: str, tiny_model):
    model, tokenizer = tiny_model
    kwargs: dict = {"max_length": 64}
    if method == "integrated_gradients":
        kwargs["n_steps"] = 4
    if method == "occlusion":
        kwargs["max_words"] = 40
    if method == "shap":
        kwargs["max_evals"] = 64
    return build_explainer(method, model, tokenizer, device="cpu", **kwargs)


def run(method: str, tiny_model, text: str) -> Explanation:
    explainer = make_explainer(method, tiny_model)
    with torch.set_grad_enabled(method in ("integrated_gradients", "attention_grad")):
        return explainer.explain(text)


class TestPayloadContract:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_returns_a_wellformed_explanation(self, method, tiny_model, sample_note):
        explanation = run(method, tiny_model, sample_note)
        assert explanation.method == method
        assert 0.0 <= explanation.probability <= 1.0
        assert explanation.predicted_label in (0, 1)
        assert explanation.words, "no word-level attributions returned"
        assert all(np.isfinite(w.score) for w in explanation.words)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_word_offsets_index_the_original_text(self, method, tiny_model, sample_note):
        """The UI slices the note with these offsets; drift shows as mis-highlighting."""
        explanation = run(method, tiny_model, sample_note)
        for word in explanation.words:
            assert 0 <= word.start < word.end <= len(sample_note)
            assert sample_note[word.start : word.end] == word.text

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_scores_are_normalised_to_unit_range(self, method, tiny_model, sample_note):
        explanation = run(method, tiny_model, sample_note)
        peak = max(abs(t.score) for t in explanation.tokens)
        assert peak <= 1.0 + 1e-6

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_special_tokens_get_no_visible_attribution(self, method, tiny_model, sample_note):
        explanation = run(method, tiny_model, sample_note)
        assert not any(t.token in ("[CLS]", "[SEP]", "[PAD]") for t in explanation.tokens)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_serialises_to_json_friendly_types(self, method, tiny_model, sample_note):
        import json

        payload = run(method, tiny_model, sample_note).to_dict()
        json.dumps(payload)  # must not raise
        assert {"method", "words", "tokens", "probability"} <= set(payload)

    @pytest.mark.parametrize("method", ["occlusion", "integrated_gradients", "attention_grad"])
    def test_signed_methods_declare_themselves_signed(self, method, tiny_model, sample_note):
        assert run(method, tiny_model, sample_note).meta.get("signed") is True

    def test_attention_rollout_is_declared_unsigned(self, tiny_model, sample_note):
        assert run("attention_rollout", tiny_model, sample_note).meta.get("signed") is False


class TestAttentionRollout:
    def test_rollout_row_is_a_distribution(self):
        torch.manual_seed(0)
        layers = tuple(torch.softmax(torch.randn(1, 2, 6, 6), dim=-1) for _ in range(3))
        row = attention_rollout(layers)
        assert row.shape == (6,)
        assert np.all(row >= 0)
        assert row.sum() == pytest.approx(1.0, abs=1e-6)

    def test_identity_attention_yields_identity_rollout(self):
        eye = torch.eye(5).unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
        row = attention_rollout((eye, eye))
        expected = np.zeros(5)
        expected[0] = 1.0
        assert np.allclose(row, expected, atol=1e-6)

    def test_more_layers_do_not_break_normalisation(self):
        torch.manual_seed(1)
        layers = tuple(torch.softmax(torch.randn(1, 2, 8, 8), dim=-1) for _ in range(12))
        assert attention_rollout(layers).sum() == pytest.approx(1.0, abs=1e-6)


class TestIntegratedGradients:
    def test_completeness_holds_approximately(self, tiny_model, sample_note):
        """IG's defining property: attributions sum to f(x) - f(baseline)."""
        model, tokenizer = tiny_model
        explainer = build_explainer(
            "integrated_gradients", model, tokenizer, device="cpu", max_length=64, n_steps=64
        )
        with torch.set_grad_enabled(True):
            explanation = explainer.explain(sample_note)
        assert explanation.meta["relative_convergence_error"] < 0.15

    def test_more_steps_reduce_convergence_error(self, tiny_model, sample_note):
        model, tokenizer = tiny_model
        errors = []
        for steps in (4, 64):
            explainer = build_explainer(
                "integrated_gradients", model, tokenizer, device="cpu", max_length=64, n_steps=steps
            )
            with torch.set_grad_enabled(True):
                errors.append(explainer.explain(sample_note).meta["relative_convergence_error"])
        assert errors[1] <= errors[0] + 1e-6


class TestOcclusion:
    def test_scores_reflect_actual_confidence_drops(self, tiny_model, sample_note):
        """Occlusion is definitionally the drop in log-odds when a word is masked."""
        model, tokenizer = tiny_model
        explainer = build_explainer(
            "occlusion", model, tokenizer, device="cpu", max_length=64, max_words=40
        )
        explanation = explainer.explain(sample_note)

        top = max(explanation.words, key=lambda w: w.score)
        masked = sample_note[: top.start] + tokenizer.mask_token + sample_note[top.end :]

        def positive_prob(text: str) -> float:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                return float(model.predict_proba(enc["input_ids"], enc["attention_mask"])[0, 1])

        assert positive_prob(masked) < positive_prob(sample_note) + 1e-9


class TestAggregation:
    def test_subword_scores_sum_into_their_word(self):
        text = "septic shock"
        tokens = [
            TokenAttribution(index=0, token="sep", score=0.3, start=0, end=3),
            TokenAttribution(index=1, token="##tic", score=0.2, start=3, end=6),
            TokenAttribution(index=2, token="shock", score=0.5, start=7, end=12),
        ]
        words = aggregate_to_words(text, tokens, reduction="sum")
        assert [w.text for w in words] == ["septic", "shock"]
        assert words[0].score == pytest.approx(0.5)
        assert words[1].score == pytest.approx(0.5)

    def test_max_reduction_keeps_the_largest_magnitude_with_its_sign(self):
        text = "septic"
        tokens = [
            TokenAttribution(index=0, token="sep", score=-0.7, start=0, end=3),
            TokenAttribution(index=1, token="##tic", score=0.2, start=3, end=6),
        ]
        words = aggregate_to_words(text, tokens, reduction="max")
        assert words[0].score == pytest.approx(-0.7)

    def test_words_with_no_tokens_score_zero(self):
        words = aggregate_to_words("alpha beta", [], reduction="sum")
        assert [w.score for w in words] == [0.0, 0.0]

    def test_normalise_max_abs_preserves_sign_and_zero(self):
        out = normalise(np.array([-4.0, 0.0, 2.0]), "max_abs")
        assert out.tolist() == [-1.0, 0.0, 0.5]

    def test_normalise_handles_all_zeros(self):
        assert normalise(np.zeros(4), "max_abs").tolist() == [0.0] * 4


class TestEncoding:
    def test_truncation_flag_is_accurate(self, tiny_model):
        _, tokenizer = tiny_model
        short = encode_with_offsets(tokenizer, "the patient was stable .", 64)
        assert short["truncated"] is False

        long_text = "the patient was in septic shock . " * 50
        assert encode_with_offsets(tokenizer, long_text, 32)["truncated"] is True

    def test_truncated_note_still_produces_valid_offsets(self, tiny_model):
        model, tokenizer = tiny_model
        long_text = "the patient was in septic shock and required vasopressor support . " * 30
        explainer = build_explainer("attention_rollout", model, tokenizer, device="cpu", max_length=32)
        explanation = explainer.explain(long_text)
        for word in explanation.words:
            if word.token_indices:
                assert long_text[word.start : word.end] == word.text


class TestTrainingMatchedTruncation:
    """Serving must read a long note exactly as training did (head + tail)."""

    LONG = ("the patient was in septic shock and required vasopressor support . " * 6
            + "hospital course stable . " * 6
            + "discharged home with follow up , no complications . " * 6)

    def test_served_ids_equal_training_ids(self, tiny_model):
        from medxplain.data.dataset import ClinicalNoteDataset
        from medxplain.explain.payload import encode_for_model, encode_with_offsets

        _, tokenizer = tiny_model
        trained = ClinicalNoteDataset([self.LONG], None, tokenizer, max_length=32,
                                      truncation_strategy="head_tail", head_fraction=0.35)[0]["input_ids"]
        served = encode_for_model(tokenizer, self.LONG, 32, head_fraction=0.35)["input_ids"][0].tolist()
        explained = encode_with_offsets(tokenizer, self.LONG, 32, head_fraction=0.35)["input_ids"][0].tolist()
        assert served == trained == explained

    def test_unread_span_marks_the_skipped_middle(self, tiny_model):
        from medxplain.explain.payload import encode_with_offsets

        _, tokenizer = tiny_model
        enc = encode_with_offsets(tokenizer, self.LONG, 32, head_fraction=0.35)
        start, end = enc["unread_span"]
        assert 0 < start < end < len(self.LONG)
        # every kept token lies outside the unread span, and its offsets index the text
        for (s, e), tok in zip(enc["offsets"][1:-1], enc["tokens"][1:-1]):
            assert e <= start or s >= end
            assert self.LONG[s:e].lower().replace(" ", "") == tok.replace("##", "")

    def test_short_note_has_no_unread_span(self, tiny_model):
        from medxplain.explain.payload import encode_with_offsets

        _, tokenizer = tiny_model
        assert encode_with_offsets(tokenizer, "the patient was stable .", 32, head_fraction=0.35)["unread_span"] is None

    def test_explainers_report_the_unread_span(self, tiny_model):
        model, tokenizer = tiny_model
        for method in ("attention_rollout", "occlusion"):
            explainer = build_explainer(method, model, tokenizer, device="cpu", max_length=32)
            explanation = explainer.explain(self.LONG)
            start, end = explanation.meta["unread_span"]
            # nothing inside the unread middle is credited
            assert all(abs(w.score) < 1e-9 for w in explanation.words if w.start >= start and w.end <= end)
