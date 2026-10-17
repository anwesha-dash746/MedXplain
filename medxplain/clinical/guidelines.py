"""Externally sourced guideline pointers for conditions recognised in a note.

Each concept detected by :mod:`medxplain.clinical.concepts` maps to published
guidance from an external body (NICE, the Surviving Sepsis Campaign, GOLD, the
Resuscitation Council UK). Every URL below was checked to resolve to the named
guideline when this table was written; titles are the publishers' own.

The suggestions point the clinician to the guidance. MedXplain does not read the
guidelines or check the case against them, and says so in every suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass

from medxplain.clinical.concepts import concept_label
from medxplain.clinical.text import Span


@dataclass(frozen=True)
class Guideline:
    key: str
    title: str
    publisher: str
    url: str


G = {
    g.key: g
    for g in (
        Guideline("nice_ng51", "Suspected sepsis: recognition, diagnosis and early management (NG51)", "NICE",
                  "https://www.nice.org.uk/guidance/ng51"),
        Guideline("ssc_adult", "Surviving Sepsis Campaign adult guidelines", "Society of Critical Care Medicine",
                  "https://www.sccm.org/survivingsepsiscampaign/guidelines-and-resources/surviving-sepsis-campaign-adult-guidelines"),
        Guideline("nice_ng106", "Chronic heart failure in adults: diagnosis and management (NG106)", "NICE",
                  "https://www.nice.org.uk/guidance/ng106"),
        Guideline("nice_ng115", "Chronic obstructive pulmonary disease in over 16s: diagnosis and management (NG115)", "NICE",
                  "https://www.nice.org.uk/guidance/ng115"),
        Guideline("gold", "Global Strategy for Prevention, Diagnosis and Management of COPD", "GOLD",
                  "https://goldcopd.org/"),
        Guideline("nice_ng148", "Acute kidney injury: prevention, detection and management (NG148)", "NICE",
                  "https://www.nice.org.uk/guidance/ng148"),
        Guideline("nice_ng107", "Renal replacement therapy and conservative management (NG107)", "NICE",
                  "https://www.nice.org.uk/guidance/ng107"),
        Guideline("nice_ng28", "Type 2 diabetes in adults: management (NG28)", "NICE",
                  "https://www.nice.org.uk/guidance/ng28"),
        Guideline("nice_ng17", "Type 1 diabetes in adults: diagnosis and management (NG17)", "NICE",
                  "https://www.nice.org.uk/guidance/ng17"),
        Guideline("nice_ng97", "Dementia: assessment, management and support for people living with dementia and their carers (NG97)",
                  "NICE", "https://www.nice.org.uk/guidance/ng97"),
        Guideline("nice_cg103", "Delirium: prevention, diagnosis and management in hospital and long-term care (CG103)",
                  "NICE", "https://www.nice.org.uk/guidance/cg103"),
        Guideline("nice_ng50", "Cirrhosis in over 16s: assessment and management (NG50)", "NICE",
                  "https://www.nice.org.uk/guidance/ng50"),
        Guideline("nice_cg141", "Acute upper gastrointestinal bleeding in over 16s: management (CG141)", "NICE",
                  "https://www.nice.org.uk/guidance/cg141"),
        Guideline("nice_ng31", "Care of dying adults in the last days of life (NG31)", "NICE",
                  "https://www.nice.org.uk/guidance/ng31"),
        Guideline("resus_uk", "Resuscitation guidelines", "Resuscitation Council UK",
                  "https://www.resus.org.uk/library/2021-resuscitation-guidelines"),
        Guideline("nice_cg76", "Medicines adherence: involving patients in decisions about prescribed medicines (CG76)",
                  "NICE", "https://www.nice.org.uk/guidance/cg76"),
        Guideline("nice_cg115", "Alcohol-use disorders: diagnosis, assessment and management (CG115)", "NICE",
                  "https://www.nice.org.uk/guidance/cg115"),
        Guideline("nice_ng27", "Transition between inpatient hospital settings and community or care home settings (NG27)",
                  "NICE", "https://www.nice.org.uk/guidance/ng27"),
    )
}

#: concept -> (guidelines, short reason the guidance may be relevant)
CONCEPT_GUIDELINES: dict[str, tuple[tuple[str, ...], str]] = {
    "septic_shock": (("nice_ng51", "ssc_adult"), "sepsis recognition and early management"),
    "lactate": (("nice_ng51",), "assessment of possible sepsis"),
    "chf": (("nice_ng106",), "heart failure management"),
    "copd": (("nice_ng115", "gold"), "COPD management"),
    "aki_dialysis": (("nice_ng148",), "acute kidney injury"),
    "esrd": (("nice_ng107",), "renal replacement therapy"),
    "diabetes": (("nice_ng28", "nice_ng17"), "diabetes management (whichever type applies)"),
    "dementia": (("nice_ng97",), "care for people living with dementia"),
    "encephalopathy": (("nice_cg103",), "delirium in hospital, if acute confusion is part of the picture"),
    "cirrhosis": (("nice_ng50",), "cirrhosis assessment and management"),
    "gi_bleed": (("nice_cg141",), "upper GI bleeding"),
    "comfort_measures": (("nice_ng31",), "care in the last days of life"),
    "hospice": (("nice_ng31",), "care in the last days of life"),
    "cardiac_arrest": (("resus_uk",), "resuscitation and post-arrest care"),
    "poor_adherence": (("nice_cg76",), "supporting medicines adherence"),
    "substance_use": (("nice_cg115",), "alcohol-use disorders, if alcohol is involved"),
    "no_support": (("nice_ng27",), "planning discharge when social support is limited"),
    "prior_admissions": (("nice_ng27",), "transitions of care after repeated admissions"),
}

MAX_GUIDELINE_SUGGESTIONS = 5


def guideline_suggestions(concepts: dict[str, list[Span]]) -> list[dict]:
    """One suggestion per recognised concept with mapped guidance, most specific first."""
    out: list[dict] = []
    used: set[str] = set()
    for key, (guideline_keys, topic) in CONCEPT_GUIDELINES.items():
        if key not in concepts or len(out) >= MAX_GUIDELINE_SUGGESTIONS:
            continue
        sources = [G[g] for g in guideline_keys if g not in used]
        if not sources:
            continue  # the same guidance was already suggested for another concept
        used.update(g.key for g in sources)
        quote = concepts[key][0].text
        out.append({
            "kind": "guideline",
            "text": (
                f"{concept_label(key)} is mentioned in this note. Consider checking the plan against "
                f"published guidance on {topic}."
            ),
            "basis": (
                f"Matched by keyword to the sentence “{quote[:120]}{'…' if len(quote) > 120 else ''}”. "
                "MedXplain links to the guidance but has not read it or checked this case against it."
            ),
            "sources": [{"title": g.title, "publisher": g.publisher, "url": g.url} for g in sources],
        })
    return out
