"""Unit tests for the clinical workspace: extraction, documentation review and
comparable-case retrieval.

The retrieval tests run against a tiny hand-built library so every expected
count can be checked by eye, which is the point: the UI shows these numbers as
facts about historical records.
"""

from __future__ import annotations

import json

import pytest

from medxplain.clinical.extraction import analyse_note
from medxplain.clinical.similar_cases import CaseLibrary
from medxplain.clinical.text import parse_note

FULL_NOTE = """Chief Complaint: shortness of breath

History of Present Illness:
Mr./Ms. <NAME> is a 72 year old female with known CHF with recurrent volume overload admissions who presented with dyspnea.

Physical Examination:
Vitals on admission: T 37.1 C, HR 104, BP 88/50. Hypotensive to 78/40 on arrival to the floor.

Pertinent Results:
Serum lactate 6.2 with a persistent anion gap acidosis.

Brief Hospital Course:
Patient developed septic shock requiring norepinephrine. Nutrition evaluated the patient and recommended a heart healthy diet.

Discharge Diagnosis:
1. Septic shock
2. Acute on chronic systolic heart failure

Discharge Medications:
1. Furosemide 40 mg PO daily (new)
2. Metoprolol 25 mg PO BID

Discharge Disposition and Condition:
Discharged home in stable condition with outpatient follow up in 2 weeks with her PCP. Blood cultures pending, PCP to follow up on the final results. The patient was advised to return for any new or worsening symptoms. Discharge instructions were reviewed with the patient and family.
"""

SPARSE_NOTE = """Brief Hospital Course:
The patient was seen daily by the primary team and treated for pneumonia with antibiotics.
"""

DEATH_NOTE = """Brief Hospital Course:
Course complicated by septic shock with escalating pressor requirement. Family meeting held and goals of care transitioned to comfort measures only.

Discharge Disposition:
The patient expired on hospital day 6 with family at the bedside.
"""


def _items(text: str) -> dict[str, dict]:
    return {i["key"]: i for i in analyse_note(text)["documentation_review"]["items"]}


def _fields(text: str) -> dict[str, dict]:
    return {f["key"]: f for f in analyse_note(text)["case_summary"]["fields"]}


class TestParsing:
    def test_every_sentence_offset_indexes_the_original_text(self):
        note = parse_note(FULL_NOTE)
        for sentence in note.sentences():
            assert FULL_NOTE[sentence.start:sentence.end] == sentence.text

    def test_known_headers_become_sections(self):
        keys = [s.key for s in parse_note(FULL_NOTE).sections]
        for key in ("chief_complaint", "hpi", "exam", "results", "course", "diagnosis", "meds", "disposition"):
            assert key in keys

    def test_honorific_does_not_split_a_sentence(self):
        hpi = parse_note(FULL_NOTE).sentences("hpi")
        assert hpi[0].text.startswith("Mr./Ms. <NAME> is a 72 year old")

    def test_numbered_list_lines_stay_whole(self):
        meds = parse_note(FULL_NOTE).sentences("meds")
        assert [m.text for m in meds] == ["1. Furosemide 40 mg PO daily (new)", "2. Metoprolol 25 mg PO BID"]


class TestCaseSummary:
    def test_documented_fields_quote_the_note(self):
        fields = _fields(FULL_NOTE)
        assert fields["presenting_complaint"]["value"] == "shortness of breath"
        assert fields["documented_diagnosis"]["status"] == "documented"
        assert fields["disposition"]["interpretation"] == "Home"
        assert fields["disposition"]["interpretation_basis"]

    def test_missing_fields_say_not_documented(self):
        fields = _fields(SPARSE_NOTE)
        for key in ("presenting_complaint", "documented_diagnosis", "disposition", "follow_up_plan"):
            assert fields[key]["status"] == "not_documented"
            assert fields[key]["display"] == "Not documented in the provided note"
            assert fields[key]["evidence"] == []

    def test_recognised_conditions_are_an_interpretation_not_a_diagnosis(self):
        note = SPARSE_NOTE + "History of present illness: known CHF and COPD.\n"
        dx = _fields(note)["documented_diagnosis"]
        assert dx["status"] == "not_documented"
        assert "Heart failure" in dx["interpretation"]
        assert "not a diagnosis" in dx["interpretation_basis"]


class TestDocumentationReview:
    def test_complete_note_documents_the_key_items(self):
        items = _items(FULL_NOTE)
        assert items["final_diagnosis"]["status"] == "documented"
        assert items["hospital_course"]["status"] == "documented"
        assert items["medications"]["status"] == "documented"
        assert items["follow_up"]["status"] == "documented"
        assert items["instructions"]["status"] == "documented"
        assert items["pending"]["status"] == "documented"
        assert items["disposition"]["status"] == "documented"

    def test_sparse_note_reports_missing_documentation(self):
        items = _items(SPARSE_NOTE)
        assert items["final_diagnosis"]["status"] == "not_found"
        assert items["follow_up"]["status"] == "not_found"
        assert items["disposition"]["status"] == "not_found"
        assert "does not mean none are pending" in items["pending"]["rationale"]

    def test_partial_follow_up_needs_verification(self):
        note = SPARSE_NOTE + "\nDischarge Disposition:\nDischarged home, follow up arranged with PCP.\n"
        assert _items(note)["follow_up"]["status"] == "needs_verification"

    def test_pending_without_owner_needs_verification(self):
        note = SPARSE_NOTE + "\nDischarge Disposition:\nDischarged home. Blood cultures pending at discharge.\n"
        assert _items(note)["pending"]["status"] == "needs_verification"

    def test_death_makes_outpatient_items_not_applicable(self):
        items = _items(DEATH_NOTE)
        for key in ("medications", "follow_up", "instructions"):
            assert items[key]["status"] == "not_applicable"
        assert items["disposition"]["status"] == "documented"

    def test_history_of_leaving_ama_is_not_this_admissions_disposition(self):
        note = (
            "History of Present Illness:\nActive polysubstance use, left AMA during a prior admission.\n\n"
            + SPARSE_NOTE
        )
        assert _items(note)["disposition"]["status"] == "not_found"

    def test_evidence_offsets_match_the_note(self):
        for item in analyse_note(FULL_NOTE)["documentation_review"]["items"]:
            for ev in item["evidence"]:
                assert FULL_NOTE[ev["start"]:ev["end"]] == ev["text"]


class TestFollowUp:
    def test_extracts_timeframe_service_and_return_precautions(self):
        fu = analyse_note(FULL_NOTE)["follow_up"]
        assert any("2 weeks" in t["value"] for t in fu["timeframes"])
        assert any(s["value"] == "PCP" for s in fu["responsible_services"])
        assert fu["return_precautions"]
        assert fu["pending_investigations"]

    def test_flags_missing_items_without_inventing_them(self):
        fu = analyse_note(SPARSE_NOTE)["follow_up"]
        assert fu["timeframes"] == [] and fu["responsible_services"] == []
        assert "No follow-up arrangement is documented." in fu["flags"]


@pytest.fixture()
def tiny_library(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    shock = (
        "Brief Hospital Course:\nPatient developed septic shock requiring norepinephrine. "
        "Required vasopressor support for persistent hypotension. {extra}\n\n"
        "Discharge Disposition and Condition:\n{dispo}\n"
    )
    rows = [
        # four shock cases: two died, one hospice, follow-up documented in two
        ("A1", shock.format(extra="Code status was addressed.", dispo="The patient expired."),
         1, 0, 70, "F", ["septic_shock", "vasopressors"]),
        ("A2", shock.format(extra="Wound care was consulted.", dispo="Discharged to inpatient hospice."),
         1, 0, 74, "M", ["septic_shock", "vasopressors", "hospice"]),
        ("A3", shock.format(extra="", dispo="Discharged home, follow up arranged with PCP."),
         0, 1, 68, "F", ["septic_shock", "vasopressors"]),
        ("A4", shock.format(extra="", dispo="Discharged home in stable condition with outpatient follow up in 2 weeks."),
         0, 0, 66, "M", ["septic_shock", "vasopressors"]),
        # unrelated cases that must not match
        ("B1", "Brief Hospital Course:\nUncomplicated elective knee replacement.\n", 0, 0, 55, "F", ["stable_course"]),
        ("B2", "Brief Hospital Course:\nCOPD exacerbation treated with steroids.\n", 0, 1, 60, "M", ["copd"]),
    ]
    frame = pd.DataFrame(
        [
            {"note_id": r[0], "text": r[1], "mortality": r[2], "readmission_30d": r[3],
             "age": r[4], "sex": r[5], "factors": json.dumps(r[6]), "split": "train"}
            for r in rows
        ]
    )
    path = tmp_path / "corpus.parquet"
    frame.to_parquet(path)
    report = tmp_path / "dataset_report.json"
    report.write_text(json.dumps({"source": "synthetic"}))
    return CaseLibrary(path, dataset_report_path=report, min_similarity=0.0)


QUERY = (
    "History of Present Illness:\nA 70 year old woman presented with fever.\n\n"
    "Brief Hospital Course:\nCourse complicated by septic shock and vasopressor support.\n"
)


class TestSimilarCases:
    def test_matches_only_cases_sharing_the_detected_conditions(self, tiny_library):
        result = tiny_library.query(QUERY)
        assert result["available"] is True
        assert result["n_matched"] == 4
        assert {e["note_id"] for e in result["examples"]} == {"A1", "A2", "A3", "A4"}

    def test_outcome_counts_are_computed_from_the_records(self, tiny_library):
        outcomes = {o["key"]: o for o in tiny_library.query(QUERY)["outcomes"]}
        assert outcomes["mortality"]["count"] == 2 and outcomes["mortality"]["n"] == 4
        # readmission is counted among the two survivors only
        assert outcomes["readmission_30d"]["count"] == 1 and outcomes["readmission_30d"]["n"] == 2

    def test_recommendation_frequencies_are_case_counts(self, tiny_library):
        recs = {r["category"]: r for r in tiny_library.query(QUERY)["recommendations"]}
        assert recs["Outpatient follow-up arranged"]["n_cases"] == 2
        assert recs["Hospice referral or enrolment"]["n_cases"] == 1
        for rec in recs.values():
            for quote in rec["examples"]:
                assert quote["note_id"] in {"A1", "A2", "A3", "A4"}

    def test_synthetic_source_is_labelled(self, tiny_library):
        source = tiny_library.query(QUERY)["source"]
        assert source["synthetic"] is True
        assert "Synthetic" in source["label"]

    def test_submitting_a_library_note_excludes_that_note(self, tiny_library):
        own = tiny_library._ensure_loaded().texts[0]
        result = tiny_library.query(own)
        assert result["excluded_duplicates"] >= 1
        assert "A1" not in {e["note_id"] for e in result["examples"]}

    def test_missing_library_returns_an_empty_state(self, tmp_path):
        result = CaseLibrary(tmp_path / "nope.parquet").query(QUERY)
        assert result["available"] is False
        assert result["data_needed"]


class TestSuggestions:
    def test_gaps_in_the_note_become_documentation_prompts(self):
        from medxplain.clinical.suggestions import generate_suggestions

        review = analyse_note(SPARSE_NOTE)["documentation_review"]
        out = generate_suggestions(SPARSE_NOTE, review, {"available": False})
        texts = " ".join(s["text"] for s in out)
        assert "follow the patient up" in texts
        assert all(s["kind"] == "documentation" and s["basis"] for s in out)

    def test_complete_items_produce_no_prompt(self):
        from medxplain.clinical.suggestions import generate_suggestions

        review = analyse_note(FULL_NOTE)["documentation_review"]
        out = generate_suggestions(FULL_NOTE, review, {"available": False})
        assert not any("follow the patient up" in s["text"] for s in out)

    def test_suggestions_do_not_come_from_historical_cases(self, tiny_library):
        from medxplain.clinical.suggestions import generate_suggestions

        similar = tiny_library.query(QUERY)
        review = analyse_note(QUERY)["documentation_review"]
        out = generate_suggestions(QUERY, review, similar)
        assert {s["kind"] for s in out} <= {"guideline", "documentation"}
        assert not any("previous cases" in s["basis"] or "matched" in s["text"].lower() for s in out)

    def test_recognised_conditions_get_external_guidance(self):
        from medxplain.clinical.suggestions import generate_suggestions

        out = generate_suggestions(FULL_NOTE, {"items": []})
        guides = [s for s in out if s["kind"] == "guideline"]
        urls = {src["url"] for s in guides for src in s["sources"]}
        assert "https://www.nice.org.uk/guidance/ng51" in urls      # septic shock -> sepsis guidance
        assert "https://www.nice.org.uk/guidance/ng106" in urls     # CHF -> heart failure guidance
        for s in guides:
            assert s["sources"] and all(src["url"].startswith("https://") for src in s["sources"])
            assert "has not read it" in s["basis"]

    def test_no_conditions_means_no_guideline_suggestions(self):
        from medxplain.clinical.suggestions import generate_suggestions

        assert [s for s in generate_suggestions("Patient feels well today.", {"items": []}) if s["kind"] == "guideline"] == []

    def test_each_guideline_is_suggested_once(self):
        from medxplain.clinical.suggestions import generate_suggestions

        note = "Brief Hospital Course:\nComfort measures only. Discharged to inpatient hospice.\n"
        guides = [s for s in generate_suggestions(note, {"items": []}) if s["kind"] == "guideline"]
        urls = [src["url"] for s in guides for src in s["sources"]]
        assert urls.count("https://www.nice.org.uk/guidance/ng31") == 1

    def test_the_list_stays_short(self):
        from medxplain.clinical.suggestions import (
            MAX_DOCUMENTATION_SUGGESTIONS,
            generate_suggestions,
        )

        review = analyse_note(SPARSE_NOTE)["documentation_review"]
        out = generate_suggestions(SPARSE_NOTE, review, {"available": False})
        assert len(out) <= MAX_DOCUMENTATION_SUGGESTIONS
