"""Keyword detection of clinical concepts in free text.

The concept keys are the same as the risk-factor keys in
:mod:`medxplain.data.clinical_lexicon`, so a concept found in a submitted note
can be matched against the structured ``factors`` column of the case library.
Detection is deliberately simple and transparent (regular expressions, shown to
the user as the matching sentence): it is a retrieval aid, not a diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medxplain.clinical.text import ParsedNote, Span


@dataclass(frozen=True)
class Concept:
    key: str
    label: str
    group: str  # acute | chronic | goals_of_care | functional | social
    pattern: re.Pattern


def _c(key: str, label: str, group: str, pattern: str) -> Concept:
    return Concept(key, label, group, re.compile(pattern, re.IGNORECASE))


CONCEPTS: tuple[Concept, ...] = (
    # acute severity
    _c("septic_shock", "Septic shock", "acute", r"\bseptic shock\b|\bdistributive shock\b|\burosepsis\b"),
    _c("vasopressors", "Vasopressor support", "acute",
       r"\bvasopressors?\b|\bpressors?\b|\bnorepinephrine\b|\blevophed\b|\bvasopressin\b|\bphenylephrine\b"),
    _c("mech_vent", "Mechanical ventilation", "acute",
       r"\bintubated\b|\bintubation\b|\bmechanical(ly)? ventilat\w*|\bextubation\b"),
    _c("cardiac_arrest", "Cardiac arrest", "acute",
       r"\bcardiac arrest\b|\bcardiopulmonary arrest\b|\b(PEA|VF|VT) arrest\b|\bROSC\b|\bACLS\b"),
    _c("aki_dialysis", "Acute kidney injury / dialysis", "acute",
       r"\bacute kidney injury\b|\bAKI\b|\bCVVH\b|\brenal replacement therapy\b|\boliguric\b|\banuri[ac]\b|\burgent hemodialysis\b"),
    _c("ards", "ARDS", "acute", r"\bARDS\b|\bacute respiratory distress syndrome\b"),
    _c("gi_bleed", "GI bleeding", "acute",
       r"\bGI bleed\w*|\bgastrointestinal (hemorrhage|bleed\w*)|\bhematemesis\b|\bmelena\b"),
    _c("encephalopathy", "Encephalopathy / altered mental status", "acute",
       r"\bencephalopathy\b|\bobtunded\b|\bGCS of \d+|\bunresponsive to verbal\b"),
    _c("lactate", "Elevated lactate", "acute",
       r"\blactate (peaked|of|\d)|\bserum lactate\b|\blactic acid(osis)?\b"),
    _c("hypotension", "Hypotension", "acute",
       r"\bhypotensi(ve|on)\b|\bMAP in the [2-5]0s\b|\bmean arterial pressure\b"),
    # chronic comorbidity
    _c("metastatic_cancer", "Metastatic cancer", "chronic",
       r"\bmetastatic\b|\bstage IV\b.*\bcancer\b|\bwidely disseminated\b"),
    _c("chf", "Heart failure", "chronic",
       r"\bCHF\b|\bheart failure\b|\bEF of \d+|\bejection fraction\b"),
    _c("copd", "COPD", "chronic", r"\bCOPD\b|\bchronic obstructive pulmonary disease\b"),
    _c("cirrhosis", "Cirrhosis", "chronic", r"\bcirrhosis\b|\bend-stage liver disease\b|\bChild-Pugh\b"),
    _c("esrd", "End-stage renal disease", "chronic", r"\bESRD\b|\bend[- ]stage renal disease\b"),
    _c("diabetes", "Diabetes", "chronic", r"\bdiabetes\b|\bIDDM\b|\bDM2?\b|\binsulin dependent\b"),
    _c("dementia", "Dementia", "chronic", r"\bdementia\b|\bneurocognitive disorder\b"),
    _c("immunosuppressed", "Immunosuppression", "chronic",
       r"\bimmunosuppress\w*|\bneutropenic\b|\btacrolimus\b|\bmycophenolate\b"),
    # goals of care
    _c("comfort_measures", "Comfort measures / DNR", "goals_of_care",
       r"\bcomfort measures\b|\bCMO\b|\bDNR\b|\bcomfort[- ]focused\b|\bgoals of care\b"),
    _c("hospice", "Hospice", "goals_of_care", r"\bhospice\b|\bend-of-life care\b"),
    # functional / reassuring
    _c("stable_course", "Uncomplicated course", "functional",
       r"\buncomplicated\b|\bhemodynamically (stable|intact)\b|\bstable vital signs\b|\bno perioperative complications\b"),
    _c("ambulating", "Ambulating independently", "functional",
       r"\bambulating independently\b|\bwalking without assistance\b|\bcleared to go home\b"),
    _c("discharge_home", "Discharged home", "functional", r"\bdischarged (to )?home\b"),
    _c("normal_labs", "Labs normalising", "functional",
       r"\bnormalized\b|\breturned to baseline\b|\blactate cleared\b"),
    # social / readmission drivers
    _c("poor_adherence", "Medication non-adherence", "social",
       r"\bnon-?adherence\b|\bnoncomplian\w*|\bmissed clinic appointments\b|\bnot been taking\b"),
    _c("prior_admissions", "Frequent prior admissions", "social",
       r"\b(re)?admissions? in the past\b|\bfrequent readmissions\b|\breturn hospitalizations\b"),
    _c("no_support", "Limited social support", "social",
       r"\blives alone\b|\bno reliable social support\b|\bunstable housing\b|\bno caregiver\b"),
    _c("substance_use", "Substance use", "social",
       r"\balcohol use disorder\b|\bpolysubstance\b|\bwithdrawal seizures?\b|\bleft AMA\b|\bagainst medical advice\b"),
)

CONCEPTS_BY_KEY: dict[str, Concept] = {c.key: c for c in CONCEPTS}


def detect_concepts(note: ParsedNote) -> dict[str, list[Span]]:
    """Concept key -> sentences that mention it, in note order.

    Searched outside the header block only, so administrative fields such as
    "Service: MICU" are not mistaken for findings.
    """
    found: dict[str, list[Span]] = {}
    for sentence in note.sentences():
        if sentence.section == "preamble":
            continue
        for concept in CONCEPTS:
            if concept.pattern.search(sentence.text):
                found.setdefault(concept.key, []).append(sentence)
    return found


def concept_label(key: str) -> str:
    concept = CONCEPTS_BY_KEY.get(key)
    return concept.label if concept else key.replace("_", " ")
