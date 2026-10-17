"""Rule-based extraction of a case summary, a discharge-documentation review and
the follow-up plan from a single discharge note.

Three rules govern every output of this module:

1. **Extracted facts are verbatim.** A fact is a sentence (or a substring of
   one) copied from the note, with its character offsets, so the UI can show
   exactly where it came from.
2. **Interpretations are labelled.** Anything this module *infers* — a
   disposition category, a list of recognised conditions — is returned in a
   separate ``interpretation`` field together with the basis for it.
3. **Absence is reported as absence of documentation**, never as absence of
   care. "Not found in the provided note" says nothing about what happened to
   the patient.

The patterns are transparent regular expressions rather than a learned model:
easy to audit, and their failure mode (missing an unusual phrasing) shows up
as "not found" rather than as a confident wrong answer.
"""

from __future__ import annotations

import re
from typing import Iterable

from medxplain.clinical.concepts import CONCEPTS_BY_KEY, detect_concepts
from medxplain.clinical.text import ParsedNote, Span, parse_note

NOT_DOCUMENTED = "Not documented in the provided note"

# -- statuses -----------------------------------------------------------------------
DOCUMENTED = "documented"
NOT_FOUND = "not_found"
NEEDS_VERIFICATION = "needs_verification"
NOT_APPLICABLE = "not_applicable"

STATUS_LABELS = {
    DOCUMENTED: "Documented",
    NOT_FOUND: "Not found in the provided note",
    NEEDS_VERIFICATION: "Needs clinician verification",
    NOT_APPLICABLE: "Not applicable",
}


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# -- patterns -----------------------------------------------------------------------
PRESENTED_WITH = _rx(r"\bpresent(?:ed|s|ing)\b[^.]*?\bwith ([^.;]+)")
DIAGNOSIS_SENTENCE = _rx(r"\b(final|discharge|primary|principal) diagnos[ie]s\b|\bdiagnosed with\b|\bdiagnosis of\b")
DEATH = _rx(r"\b(expired|passed away|pronounced dead|time of death|patient died|deceased)\b")
DISPOSITION = _rx(
    r"\bdischarged\b|\bdischarge to\b|\btransferred to\b|\bhospice\b|\bskilled nursing\b|\bSNF\b"
    r"|\b(acute )?rehab(ilitation)?( facility)?\b|\bleft against medical advice\b|\bAMA\b"
    r"|\bhome with (services|VNA|family)\b|\bcleared to go home\b"
)
CONDITION = _rx(
    r"\b(stable|good|fair|poor|guarded|critical) condition\b|\bcondition (at|on) discharge\b"
    r"|\bambulat\w*|\btolerating\b|\bmental status\b|\bhemodynamically stable\b|\bcomfort\b"
)
FOLLOW_UP = _rx(
    r"\bfollow[- ]?up\b|\bf/u\b|\bPCP\b|\bprimary care\b|\bappointment\b|\bclinic\b"
    r"|\bsee (Dr|your)\b|\boutpatient\b"
)
PROVIDER = _rx(
    r"\b(PCP|primary care( physician| provider| doctor)?|primary team|cardiolog\w*|nephrolog\w*|oncolog\w*"
    r"|pulmonolog\w*|neurolog\w*|hepatolog\w*|gastroenterolog\w*|endocrinolog\w*|surgeon|surgical clinic"
    r"|infectious disease|palliative care|hospice team|wound clinic|[A-Z][a-z]+ clinic|Dr\.? ?[A-Z][a-z]+)\b"
)
TIMEFRAME = _rx(
    r"\b(in|within|after)\s+(\d+|one|two|three|four|five|six|a|an)(\s*(-|to)\s*\d+)?\s*(day|week|month)s?\b"
    r"|\b(tomorrow|next week|next month)\b|\bon \[\*\*[^\]]*\*\*\]|\b\d{1,2}/\d{1,2}(/\d{2,4})?\b"
)
INSTRUCTIONS = _rx(r"\binstructions?\b|\binstructed\b|\beducat\w+|\bcounsel(l)?ed\b|\breviewed with the patient\b")
WARNING_SIGNS = _rx(
    r"\breturn (to|for)\b|\bwarning signs\b|\breturn precautions\b|\bseek (immediate )?medical\b"
    r"|\bcall (your|the) (doctor|PCP|physician|clinic|team)\b|\bif (you|he|she|they) (experience|develop|have)\b"
    r"|\bnew or worsening\b|\bcome back\b|\bcall 911\b"
)
PENDING = _rx(
    r"\bpending\b|\bawaiting\b|\boutstanding (results?|tests?|studies)\b|\bresults? (are|is) not yet\b"
    r"|\bfinal (read|report)\b|\bsend-?out\b"
)
RESPONSIBILITY = _rx(
    r"\b(PCP|primary care|primary team|clinic|team|Dr\.? ?[A-Z][a-z]+|cardiology|oncology|nephrology)\b[^.]{0,40}"
    r"\b(to|will)\s+(follow|review|check|f/u|track)\b|\b(follow(ed)?|review(ed)?)\b[^.]{0,20}\bby\b"
)
MED_CHANGE = _rx(
    r"\b(new|started|start|stop(ped)?|discontinu\w+|chang\w+|increas\w+|decreas\w+|hold|held|resume\w*"
    r"|continue\w*|reconcil\w+)\b"
)
MED_DOSE = _rx(r"\b\d+(\.\d+)?\s*(mg|mcg|g|units?|mL|meq)\b|\b(PO|IV|SC|IM|PR|SL)\b|\b(daily|BID|TID|QID|q\d+h|qHS|PRN)\b")
UNRESOLVED = _rx(
    r"\bunresolved\b|\boutstanding issues?\b|\btransitional issues\b|\bissues? (to be|for) (addressed|follow)"
    r"|\bongoing (issues?|problems?)\b|\bto be addressed\b|\bremains? to be\b|\bwill need\b"
)
SOCIAL_CONCEPTS = ("no_support", "poor_adherence", "substance_use", "prior_admissions")

_DISPOSITION_CATEGORIES: tuple[tuple[str, re.Pattern], ...] = (
    ("Died in hospital", DEATH),
    ("Hospice", _rx(r"\bhospice\b")),
    ("Left against medical advice", _rx(r"\bleft against medical advice\b|\bleft AMA\b(?! during)")),
    ("Skilled nursing / rehabilitation facility", _rx(r"\bskilled nursing\b|\bSNF\b|\brehab(ilitation)?\b|\bextended care\b")),
    ("Transferred to another facility", _rx(r"\btransferred to\b")),
    ("Home with services", _rx(r"\bhome with (services|VNA|home health)\b")),
    ("Home", _rx(r"\b(discharged|discharge) (to )?home\b|\bcleared to go home\b|\bhome with family\b")),
)


def _evidence(spans: Iterable[Span], limit: int | None = None) -> list[dict]:
    seen: set[tuple[int, int]] = set()
    out = []
    for span in spans:
        if (span.start, span.end) in seen:
            continue
        seen.add((span.start, span.end))
        out.append(span.as_dict())
        if limit and len(out) >= limit:
            break
    return out


def _matching(sentences: Iterable[Span], pattern: re.Pattern) -> list[Span]:
    return [s for s in sentences if pattern.search(s.text)]


def _without_history(note: ParsedNote) -> list[Span]:
    """Sentences outside the history sections, where disposition statements live.

    History sections describe the past ("left AMA during a prior admission") and
    must not be read as this admission's disposition.
    """
    return [s for s in note.sentences() if s.section not in ("preamble", "hpi", "pmh", "chief_complaint")]


def _discharge_sentences(note: ParsedNote) -> list[Span]:
    return note.sentences("disposition", "condition", "instructions", "followup")


def _field(key: str, label: str, spans: list[Span], *, value: str | None = None,
           interpretation: str | None = None, basis: str | None = None, limit: int | None = 6) -> dict:
    documented = bool(spans)
    return {
        "key": key,
        "label": label,
        "status": "documented" if documented else "not_documented",
        "value": value if documented else None,
        "display": None if documented else NOT_DOCUMENTED,
        "evidence": _evidence(spans, limit),
        "n_evidence": len(spans),
        "interpretation": interpretation,
        "interpretation_basis": basis if interpretation else None,
    }


# -----------------------------------------------------------------------------------
# Case summary
# -----------------------------------------------------------------------------------

def death_documented(note: ParsedNote) -> list[Span]:
    return _matching(_without_history(note), DEATH)


def disposition_spans(note: ParsedNote) -> list[Span]:
    preferred = _matching(_discharge_sentences(note), DISPOSITION)
    if preferred:
        return preferred
    return _matching(_without_history(note), DISPOSITION) + death_documented(note)


def disposition_category(spans: list[Span]) -> str | None:
    joined = " ".join(s.text for s in spans)
    for label, pattern in _DISPOSITION_CATEGORIES:
        if pattern.search(joined):
            return label
    return None


def follow_up_spans(note: ParsedNote) -> list[Span]:
    return _matching(_without_history(note), FOLLOW_UP)


def case_summary(note: ParsedNote, concepts: dict[str, list[Span]] | None = None) -> dict:
    concepts = concepts if concepts is not None else detect_concepts(note)

    # presenting complaint: the Chief Complaint field, else "presented ... with X"
    cc = note.sentences("chief_complaint")
    complaint_value = cc[0].text if cc else None
    if not cc:
        presented = _matching(note.sentences("hpi", "preamble"), PRESENTED_WITH)
        cc = presented[:1]
        if cc:
            m = PRESENTED_WITH.search(cc[0].text)
            complaint_value = m.group(1).strip() if m else cc[0].text

    # findings: exam and results sentences, ones mentioning a recognised concept first
    finding_sentences = note.sentences("exam", "results")
    concept_starts = {s.start for spans in concepts.values() for s in spans}
    findings = sorted(finding_sentences, key=lambda s: (s.start not in concept_starts, s.start))

    diagnosis = note.sentences("diagnosis") or _matching(note.sentences(), DIAGNOSIS_SENTENCE)
    recognised = [
        CONCEPTS_BY_KEY[k].label
        for k in concepts
        if CONCEPTS_BY_KEY[k].group in ("acute", "chronic")
    ]

    course = note.sentences("course")

    condition = _matching(_discharge_sentences(note), CONDITION) or death_documented(note)

    dispo = disposition_spans(note)
    category = disposition_category(dispo)

    follow = follow_up_spans(note)

    return {
        "fields": [
            _field("presenting_complaint", "Presenting complaint", cc, value=complaint_value),
            _field(
                "relevant_findings", "Relevant findings", findings,
                limit=8,
            ),
            _field(
                "documented_diagnosis", "Documented diagnosis", diagnosis,
                interpretation=(", ".join(recognised) if recognised and not diagnosis else None),
                basis=(
                    "No diagnosis is stated in the note. These conditions were recognised by keyword "
                    "matching anywhere in the text; they are not a diagnosis."
                ),
            ),
            _field("treatment_course", "Treatment and hospital course", course, limit=10),
            _field("discharge_status", "Discharge status", condition),
            _field(
                "disposition", "Disposition", dispo,
                interpretation=category,
                basis="Category assigned by matching the quoted sentence(s) against disposition keywords.",
            ),
            _field("follow_up_plan", "Follow-up plan", follow),
        ],
        "recognised_concepts": [
            {
                "key": key,
                "label": CONCEPTS_BY_KEY[key].label,
                "group": CONCEPTS_BY_KEY[key].group,
                "evidence": _evidence(spans, 3),
            }
            for key, spans in concepts.items()
        ],
    }


# -----------------------------------------------------------------------------------
# Discharge documentation review
# -----------------------------------------------------------------------------------

def _item(key: str, label: str, status: str, rationale: str, spans: Iterable[Span] = ()) -> dict:
    return {
        "key": key,
        "label": label,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "rationale": rationale,
        "evidence": _evidence(spans, 5),
    }


def documentation_review(note: ParsedNote, concepts: dict[str, list[Span]] | None = None) -> dict:
    concepts = concepts if concepts is not None else detect_concepts(note)
    died = death_documented(note)
    items: list[dict] = []

    # 1. final diagnosis
    dx_section = note.sentences("diagnosis")
    dx = dx_section or _matching(note.sentences(), DIAGNOSIS_SENTENCE)
    if dx:
        items.append(_item("final_diagnosis", "Final diagnosis", DOCUMENTED,
                           "A diagnosis section or explicit diagnosis statement was found.", dx))
    else:
        items.append(_item("final_diagnosis", "Final diagnosis", NOT_FOUND,
                           "No discharge-diagnosis section or explicit diagnosis statement was found."))

    # 2. hospital course and treatment
    course = note.sentences("course")
    if course:
        items.append(_item("hospital_course", "Hospital course and treatment", DOCUMENTED,
                           f"A hospital-course section with {len(course)} sentence(s) was found.", course[:3]))
    else:
        items.append(_item("hospital_course", "Hospital course and treatment", NOT_FOUND,
                           "No hospital-course section was found."))

    # 3. discharge medication instructions
    meds = note.sentences("meds")
    med_lines = [s for s in meds if MED_DOSE.search(s.text)]
    changes = _matching(meds + _discharge_sentences(note) + note.sentences("course"), MED_CHANGE)
    changes = [s for s in changes if s.section in ("meds", "instructions", "disposition", "followup")
               or "medication" in s.text.lower()]
    if died:
        items.append(_item("medications", "Discharge medication instructions", NOT_APPLICABLE,
                           "The note documents that the patient died in hospital.", died))
    elif med_lines and changes:
        items.append(_item("medications", "Discharge medication instructions", DOCUMENTED,
                           f"A discharge medication list ({len(med_lines)} item(s) with dose or frequency) "
                           "and a statement about medication changes were found.", med_lines[:3] + changes[:2]))
    elif med_lines:
        items.append(_item("medications", "Discharge medication instructions", NEEDS_VERIFICATION,
                           f"A discharge medication list ({len(med_lines)} item(s)) is present, but no explicit "
                           "new / changed / stopped medications were identified. Please confirm the changes "
                           "from pre-admission medications are clear.", med_lines[:3]))
    else:
        items.append(_item("medications", "Discharge medication instructions", NOT_FOUND,
                           "No discharge medication list was found."))

    # 4. follow-up provider and timeframe
    follow = follow_up_spans(note)
    providers = [s for s in follow if PROVIDER.search(s.text)]
    timeframes = [s for s in follow if TIMEFRAME.search(s.text)]
    if died:
        items.append(_item("follow_up", "Follow-up provider and timeframe", NOT_APPLICABLE,
                           "The note documents that the patient died in hospital.", died))
    elif providers and timeframes:
        items.append(_item("follow_up", "Follow-up provider and timeframe", DOCUMENTED,
                           "Both a responsible provider and a timeframe are stated.", providers + timeframes))
    elif providers or timeframes:
        missing = "timeframe" if providers else "responsible provider or service"
        items.append(_item("follow_up", "Follow-up provider and timeframe", NEEDS_VERIFICATION,
                           f"Follow-up is mentioned, but no {missing} was identified.", providers or timeframes))
    elif follow:
        items.append(_item("follow_up", "Follow-up provider and timeframe", NEEDS_VERIFICATION,
                           "Follow-up is mentioned without a named provider or a timeframe.", follow))
    else:
        items.append(_item("follow_up", "Follow-up provider and timeframe", NOT_FOUND,
                           "No follow-up arrangement was found."))

    # 5. patient instructions and warning signs
    instr = _matching(_without_history(note), INSTRUCTIONS)
    warn = _matching(_without_history(note), WARNING_SIGNS)
    if died:
        items.append(_item("instructions", "Patient instructions and warning signs", NOT_APPLICABLE,
                           "The note documents that the patient died in hospital.", died))
    elif instr and warn:
        items.append(_item("instructions", "Patient instructions and warning signs", DOCUMENTED,
                           "Discharge instructions and return precautions were both found.", instr + warn))
    elif warn:
        items.append(_item("instructions", "Patient instructions and warning signs", NEEDS_VERIFICATION,
                           "Return precautions were found, but no statement that discharge instructions were "
                           "given.", warn))
    elif instr:
        items.append(_item("instructions", "Patient instructions and warning signs", NEEDS_VERIFICATION,
                           "Instructions are mentioned, but no specific warning signs or return precautions "
                           "were found.", instr))
    else:
        items.append(_item("instructions", "Patient instructions and warning signs", NOT_FOUND,
                           "No patient instructions or warning signs were found."))

    # 6. pending investigations and responsibility
    pending = _matching(_without_history(note), PENDING)
    responsible = [s for s in pending if RESPONSIBILITY.search(s.text)]
    if pending and responsible:
        items.append(_item("pending", "Pending investigations and responsibility", DOCUMENTED,
                           "Pending results are listed with a named person or team to review them.", pending))
    elif pending:
        items.append(_item("pending", "Pending investigations and responsibility", NEEDS_VERIFICATION,
                           "Pending results are mentioned, but nobody is named as responsible for reviewing "
                           "them.", pending))
    else:
        items.append(_item("pending", "Pending investigations and responsibility", NOT_FOUND,
                           "The note makes no statement about pending results. This does not mean none are "
                           "pending; please confirm."))

    # 7. discharge disposition
    dispo = disposition_spans(note)
    category = disposition_category(dispo)
    if dispo:
        items.append(_item("disposition", "Discharge disposition", DOCUMENTED,
                           f"Disposition stated{f' (interpreted as: {category})' if category else ''}.", dispo))
    else:
        items.append(_item("disposition", "Discharge disposition", NOT_FOUND,
                           "No statement of where the patient went after discharge was found."))

    # 8. unresolved clinical issues
    unresolved = _matching(_without_history(note), UNRESOLVED)
    social = [s for k in SOCIAL_CONCEPTS for s in concepts.get(k, [])]
    if unresolved:
        items.append(_item("unresolved", "Unresolved clinical issues", DOCUMENTED,
                           "Outstanding or unresolved issues are explicitly listed.", unresolved))
    elif social:
        items.append(_item("unresolved", "Unresolved clinical issues", NEEDS_VERIFICATION,
                           "The note mentions issues that may continue after discharge (shown below) but does "
                           "not list them as unresolved issues or say who will address them.",
                           sorted(social, key=lambda s: s.start)))
    else:
        items.append(_item("unresolved", "Unresolved clinical issues", NOT_FOUND,
                           "No list of unresolved or outstanding issues was found."))

    counts = {status: sum(1 for i in items if i["status"] == status) for status in STATUS_LABELS}
    return {
        "items": items,
        "counts": counts,
        "disclaimer": (
            "This checklist reflects what is written in the provided note only. 'Not found' means the "
            "documentation was not found, not that the care did not happen. It is not an approval of the "
            "discharge."
        ),
    }


# -----------------------------------------------------------------------------------
# Follow-up and care continuity
# -----------------------------------------------------------------------------------

def _extract(pattern: re.Pattern, spans: Iterable[Span]) -> list[dict]:
    out, seen = [], set()
    for span in spans:
        for match in pattern.finditer(span.text):
            value = match.group(0).strip()
            if value.lower() in seen:
                continue
            seen.add(value.lower())
            out.append({
                "value": value,
                "evidence": span.as_dict(),
            })
    return out


def follow_up(note: ParsedNote) -> dict:
    died = death_documented(note)
    follow = follow_up_spans(note)
    pending = _matching(_without_history(note), PENDING)
    meds = note.sentences("meds")
    med_changes = [s for s in _matching(_discharge_sentences(note) + meds, MED_CHANGE)]
    warn = _matching(_without_history(note), WARNING_SIGNS)

    timeframes = _extract(TIMEFRAME, follow)
    services = _extract(PROVIDER, follow)

    flags: list[str] = []
    if not died:
        if not follow:
            flags.append("No follow-up arrangement is documented.")
        else:
            if not timeframes:
                flags.append("Follow-up is mentioned but no timeframe is documented.")
            if not services:
                flags.append("Follow-up is mentioned but no responsible clinician or service is named.")
        if not warn:
            flags.append("No return precautions or warning signs are documented.")
        if not meds:
            flags.append("No discharge medication list is documented.")
    if pending and not any(RESPONSIBILITY.search(s.text) for s in pending):
        flags.append("Pending results are mentioned without naming who will review them.")
    if not pending:
        flags.append("The note does not say whether any results are pending.")

    return {
        "patient_died": bool(died),
        "arrangements": _evidence(follow),
        "timeframes": timeframes,
        "responsible_services": services,
        "pending_investigations": _evidence(pending),
        "medication_list": _evidence(meds),
        "medication_changes": _evidence(med_changes),
        "return_precautions": _evidence(warn),
        "flags": flags,
    }


def analyse_note(text: str) -> dict:
    """Case summary, documentation review and follow-up for one note."""
    note = parse_note(text)
    concepts = detect_concepts(note)
    return {
        "sections_found": [
            {"key": s.key, "header": s.header, "start": s.start, "end": s.end}
            for s in note.sections
            if s.key != "preamble"
        ],
        "case_summary": case_summary(note, concepts),
        "documentation_review": documentation_review(note, concepts),
        "follow_up": follow_up(note),
        "concept_keys": sorted(concepts),
    }
