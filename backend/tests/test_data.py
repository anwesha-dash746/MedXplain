"""Tests for corpus generation, preprocessing and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medxplain.data.preprocess import (
    PreprocessConfig,
    chunk_text,
    clean_note,
    normalise_abbreviations,
    offset_preserving_clean,
)
from medxplain.data.splits import SplitConfig, check_leakage, split_report, stratified_split
from medxplain.data.synthetic import (
    SyntheticCorpusConfig,
    generate_corpus,
    near_duplicate_note,
    paraphrase_note,
)


class TestSyntheticCorpus:
    def test_generates_requested_number_of_notes(self):
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=50, seed=1))
        assert len(rows) == 50
        assert all(row["text"].strip() for row in rows)

    def test_is_deterministic_for_a_seed(self):
        a = generate_corpus(SyntheticCorpusConfig(n_notes=20, seed=7))
        b = generate_corpus(SyntheticCorpusConfig(n_notes=20, seed=7))
        assert [r["text"] for r in a] == [r["text"] for r in b]
        assert [r["mortality"] for r in a] == [r["mortality"] for r in b]

    def test_different_seeds_give_different_corpora(self):
        a = generate_corpus(SyntheticCorpusConfig(n_notes=20, seed=1))
        b = generate_corpus(SyntheticCorpusConfig(n_notes=20, seed=2))
        assert [r["text"] for r in a] != [r["text"] for r in b]

    def test_positive_rate_lands_near_the_target(self):
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=1500, seed=3, target_positive_rate=0.16))
        rate = np.mean([r["mortality"] for r in rows])
        assert 0.12 < rate < 0.21

    def test_gold_spans_index_the_note_exactly(self):
        """The whole plausibility evaluation rests on these offsets being right."""
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=40, seed=11))
        checked = 0
        for row in rows:
            for span in row["gold_spans"]:
                assert row["text"][span["start"] : span["end"]] == span["text"]
                checked += 1
        assert checked > 0

    def test_risk_factors_shift_the_label_in_the_right_direction(self):
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=1200, seed=5))
        with_shock = [r["mortality"] for r in rows if "septic_shock" in r["factors"]]
        without = [r["mortality"] for r in rows if "septic_shock" not in r["factors"]]
        assert np.mean(with_shock) > np.mean(without)

        # protective factors must move it the other way
        with_stable = [r["mortality"] for r in rows if "stable_course" in r["factors"]]
        assert np.mean(with_stable) < np.mean(without)

    def test_labels_are_sampled_not_thresholded(self):
        """A deterministic label would make the benchmark trivially saturable."""
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=800, seed=9))
        high = [r for r in rows if r["mortality_logit"] > 1.0]
        assert high, "expected some high-logit notes"
        outcomes = {r["mortality"] for r in high}
        assert outcomes == {0, 1}, "high-risk notes should not all share one outcome"


class TestParaphrasing:
    def test_paraphrase_changes_text_but_keeps_length_similar(self):
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=5, seed=2))
        text = rows[0]["text"]
        out = paraphrase_note(text, seed=1)
        assert out != text
        assert abs(len(out) - len(text)) < len(text) * 0.35

    def test_near_duplicate_preserves_all_words(self):
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=3, seed=4))
        text = rows[0]["text"]
        out = near_duplicate_note(text, seed=1)
        assert out.split() != [] and len(out.split()) == len(text.split())


class TestPreprocess:
    def test_deid_placeholders_become_typed_tokens(self):
        text = "Admitted [**2145-3-6**] to [**Hospital1 18**] under [**Known lastname 55**]."
        out = clean_note(text, PreprocessConfig(strip_header=False))
        assert "[**" not in out
        assert "<DATE>" in out and "<PLACE>" in out and "<NAME>" in out

    def test_header_lines_are_removed(self):
        text = "Admission Date: [**2145-3-6**]\nService: MEDICINE\nHistory: chest pain\n"
        out = clean_note(text, PreprocessConfig())
        assert "Service:" not in out
        assert "chest pain" in out

    def test_header_stripping_is_optional(self):
        text = "Service: MEDICINE\nHistory: chest pain\n"
        out = clean_note(text, PreprocessConfig(strip_header=False))
        assert "Service:" in out

    def test_abbreviation_expansion_respects_word_boundaries(self):
        assert "shortness of breath" in normalise_abbreviations("Patient with SOB today")
        # 'cp' inside another word must not be touched
        assert "acpx" in normalise_abbreviations("acpx unchanged")

    def test_offset_preserving_clean_keeps_length(self):
        text = "Seen on [**2145-3-6**] at [**Hospital1 18**]."
        assert len(offset_preserving_clean(text)) == len(text)

    def test_empty_input_does_not_raise(self):
        assert clean_note("") == ""

    def test_cleaning_is_idempotent(self):
        text = "Service: X\n\n\n\nPatient  seen [**2145-3-6**].\n"
        once = clean_note(text)
        assert clean_note(once) == once


class TestChunking:
    def test_chunks_cover_a_long_note_with_overlap(self, tiny_model):
        _, tokenizer = tiny_model
        text = "the patient was in septic shock and required vasopressor support . " * 40
        chunks = chunk_text(text, tokenizer, max_length=32, stride=8)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["input_ids"]) <= 32
            assert len(chunk["input_ids"]) == len(chunk["attention_mask"])
            assert chunk["input_ids"][0] == tokenizer.cls_token_id
            assert chunk["input_ids"][-1] == tokenizer.sep_token_id

    def test_short_note_yields_one_chunk(self, tiny_model):
        _, tokenizer = tiny_model
        chunks = chunk_text("the patient was stable .", tokenizer, max_length=64)
        assert len(chunks) == 1

    def test_empty_text_still_returns_a_chunk(self, tiny_model):
        _, tokenizer = tiny_model
        assert len(chunk_text("", tokenizer, max_length=32)) == 1


class TestSplits:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=400, seed=6))
        df = pd.DataFrame(rows)
        df["subject_id"] = range(len(df))
        return df

    def test_split_sizes_and_stratification(self, frame):
        out = stratified_split(frame, SplitConfig(test_size=0.2, val_size=0.2, group_col=None))
        counts = out["split"].value_counts()
        assert set(counts.index) == {"train", "val", "test"}
        assert abs(counts["test"] / len(out) - 0.2) < 0.03

        overall = out["mortality"].mean()
        for split in ("train", "val", "test"):
            rate = out[out["split"] == split]["mortality"].mean()
            assert abs(rate - overall) < 0.06, f"{split} rate {rate} drifted from {overall}"

    def test_grouped_split_keeps_patients_disjoint(self):
        rows = generate_corpus(SyntheticCorpusConfig(n_notes=400, seed=8))
        df = pd.DataFrame(rows)
        # two admissions per patient
        df["subject_id"] = [i // 2 for i in range(len(df))]
        out = stratified_split(df, SplitConfig(group_col="subject_id"))
        per_patient = out.groupby("subject_id")["split"].nunique()
        assert (per_patient == 1).all(), "a patient appeared in more than one split"

    def test_leakage_check_flags_duplicate_text(self, frame):
        out = stratified_split(frame, SplitConfig(group_col=None))
        assert check_leakage(out)["ok"]

        rigged = out.copy()
        train_text = rigged[rigged["split"] == "train"].iloc[0]["text"]
        test_index = rigged[rigged["split"] == "test"].index[0]
        rigged.loc[test_index, "text"] = train_text
        assert not check_leakage(rigged)["ok"]

    def test_split_report_shape(self, frame):
        out = stratified_split(frame, SplitConfig(group_col=None))
        report = split_report(out, ("mortality",))
        assert set(report) == {"train", "val", "test"}
        assert all("mortality_rate" in block for block in report.values())
