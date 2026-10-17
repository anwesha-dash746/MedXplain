"""Automatically generated suggestions, kept separate from historical findings.

Suggestions come from two inputs, neither of them the historical case library:

* **external guidance** - published guidelines (NICE, Surviving Sepsis
  Campaign, GOLD, Resuscitation Council UK) mapped to conditions recognised in
  this note, see :mod:`medxplain.clinical.guidelines`;
* **documentation gaps** in this note's discharge summary.

They are prompts for the clinician to consider. They never propose a diagnosis,
a drug or a dose, and every suggestion carries the basis it was generated from.
The list is short by design.
"""

from __future__ import annotations

from medxplain.clinical.concepts import detect_concepts
from medxplain.clinical.guidelines import guideline_suggestions
from medxplain.clinical.text import parse_note

#: documentation-review item -> prompt, in priority order (most safety-relevant first)
_REVIEW_PROMPTS: tuple[tuple[str, str], ...] = (
    ("follow_up", "Consider documenting who will follow the patient up and by when."),
    ("pending", "Consider confirming whether any results are pending and who will review them."),
    ("medications", "Consider making clear which medications are new, changed or stopped."),
    ("instructions", "Consider documenting the instructions and warning signs given to the patient."),
    ("disposition", "Consider stating where the patient is going after discharge."),
    ("final_diagnosis", "Consider stating the final diagnosis explicitly."),
    ("unresolved", "Consider listing unresolved issues and who will address them."),
)

MAX_DOCUMENTATION_SUGGESTIONS = 3


def generate_suggestions(text: str, review: dict, similar: dict | None = None) -> list[dict]:
    """Guideline pointers first, then the most important documentation gaps.

    ``similar`` (the historical case search) is accepted for API compatibility
    but deliberately not used: these suggestions are not derived from history.
    """
    suggestions = guideline_suggestions(detect_concepts(parse_note(text)))

    items = {item["key"]: item for item in review.get("items", [])}
    added = 0
    for key, prompt in _REVIEW_PROMPTS:
        item = items.get(key)
        if item is None or item["status"] not in ("not_found", "needs_verification"):
            continue
        if added >= MAX_DOCUMENTATION_SUGGESTIONS:
            break
        suggestions.append({
            "kind": "documentation",
            "text": prompt,
            "basis": (
                f"{item['label']}: not found in this note."
                if item["status"] == "not_found"
                else f"{item['label']}: mentioned but incomplete in this note."
            ),
        })
        added += 1
    return suggestions
