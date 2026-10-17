"""Section and sentence segmentation with character offsets.

Everything the clinical workspace shows is anchored to a span of the note the
clinician submitted, so segmentation never rewrites the text: every section and
sentence carries ``start``/``end`` offsets into the original string, and
``note[start:end]`` is exactly what is displayed as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Canonical section keys and the header spellings that map to them. Headers not
#: listed here (e.g. "Vitals on admission:") are treated as part of the
#: surrounding section rather than starting a new one.
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "chief_complaint": ("chief complaint", "presenting complaint", "reason for admission", "cc"),
    "hpi": ("history of present illness", "hpi"),
    "pmh": ("past medical history", "pmh"),
    "exam": ("physical examination", "physical exam", "exam on admission", "examination"),
    "results": ("pertinent results", "labs", "laboratory data", "laboratory results", "imaging", "results"),
    "course": ("brief hospital course", "hospital course", "clinical course"),
    "diagnosis": (
        "discharge diagnosis", "discharge diagnoses", "final diagnosis", "final diagnoses",
        "primary diagnosis", "secondary diagnosis", "diagnoses", "diagnosis",
    ),
    "meds": ("discharge medications", "medications on discharge", "discharge meds"),
    "disposition": ("discharge disposition and condition", "discharge disposition", "disposition"),
    "condition": ("discharge condition", "condition at discharge", "condition on discharge"),
    "instructions": ("discharge instructions", "patient instructions", "instructions"),
    "followup": ("followup instructions", "follow-up instructions", "follow up instructions",
                 "follow-up", "follow up", "followup"),
    "pending": ("pending results", "pending labs", "pending studies", "results pending"),
}

SECTION_LABELS: dict[str, str] = {
    "chief_complaint": "Chief Complaint",
    "hpi": "History of Present Illness",
    "pmh": "Past Medical History",
    "exam": "Physical Examination",
    "results": "Pertinent Results",
    "course": "Hospital Course",
    "diagnosis": "Discharge Diagnosis",
    "meds": "Discharge Medications",
    "disposition": "Discharge Disposition",
    "condition": "Discharge Condition",
    "instructions": "Discharge Instructions",
    "followup": "Follow-up",
    "pending": "Pending Results",
    "preamble": "Header",
}

_ALIAS_TO_KEY = {alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases}
_HEADER_RE = re.compile(r"(?m)^[ \t]*([A-Za-z][A-Za-z /&()'-]{1,60}?)[ \t]*:")
_LIST_ITEM_RE = re.compile(r"^\s*(\d+[.)]|[-*•])\s+")
_ABBREV_BEFORE_PERIOD = re.compile(
    r"(?:\b(?:Mr|Ms|Mrs|Dr|St|vs|etc|approx|e\.g|i\.e|Mr\./Ms|No)|\b[A-Z])$"
)


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    section: str

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "section": self.section,
            "section_label": SECTION_LABELS.get(self.section, self.section),
        }


@dataclass
class Section:
    key: str
    header: str
    start: int  # start of the section's content (after the header colon)
    end: int
    sentences: list[Span] = field(default_factory=list)


@dataclass
class ParsedNote:
    text: str
    sections: list[Section]

    def section(self, key: str) -> Section | None:
        for s in self.sections:
            if s.key == key:
                return s
        return None

    def sentences(self, *keys: str) -> list[Span]:
        """Sentences from the given sections in note order; every section if none given."""
        wanted = set(keys)
        return [
            sentence
            for s in self.sections
            if not wanted or s.key in wanted
            for sentence in s.sentences
        ]

    def has(self, key: str) -> bool:
        s = self.section(key)
        return bool(s and s.sentences)


def parse_note(text: str) -> ParsedNote:
    """Split a note into recognised sections, each with offset-anchored sentences."""
    headers: list[tuple[str, str, int, int]] = []
    for match in _HEADER_RE.finditer(text):
        name = re.sub(r"\s+", " ", match.group(1).strip().lower())
        key = _ALIAS_TO_KEY.get(name)
        if key:
            headers.append((key, match.group(1).strip(), match.start(), match.end()))

    sections: list[Section] = []
    first_start = headers[0][2] if headers else len(text)
    if first_start > 0:
        sections.append(Section("preamble", "", 0, first_start))
    for i, (key, header, _h_start, content_start) in enumerate(headers):
        end = headers[i + 1][2] if i + 1 < len(headers) else len(text)
        sections.append(Section(key, header, content_start, end))

    for section in sections:
        section.sentences = _sentences(text, section.start, section.end, section.key)
    return ParsedNote(text=text, sections=sections)


def _sentences(text: str, start: int, end: int, section: str) -> list[Span]:
    """Sentence spans inside text[start:end]; numbered/bulleted lines stay whole."""
    out: list[Span] = []
    cursor = start
    for line in text[start:end].splitlines(keepends=True):
        line_start, line_end = cursor, cursor + len(line)
        cursor = line_end
        if not line.strip():
            continue
        if _LIST_ITEM_RE.match(line):
            out.extend(_trimmed(text, line_start, line_end, section))
            continue
        piece_start = line_start
        for i in range(line_start, line_end):
            ch = text[i]
            if ch in ".!?" and (i + 1 >= line_end or text[i + 1].isspace()):
                if ch == "." and _ABBREV_BEFORE_PERIOD.search(text[piece_start:i]):
                    continue
                out.extend(_trimmed(text, piece_start, i + 1, section))
                piece_start = i + 1
        out.extend(_trimmed(text, piece_start, line_end, section))
    return out


def _trimmed(text: str, start: int, end: int, section: str) -> list[Span]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start < 2:
        return []
    return [Span(text[start:end], start, end, section)]
