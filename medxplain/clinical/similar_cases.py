"""Retrieval of comparable historical cases from the processed corpus.

Matching uses two signals, both reported back to the user as the criteria that
were applied:

* **structured fields** - the ``factors`` column of the corpus (clinical
  concepts recorded for each admission), matched against concepts detected in
  the submitted note, plus age band when the note states an age;
* **text similarity** - TF-IDF cosine similarity over the full note text.

Every number returned (case counts, outcome rates, recommendation frequencies)
is computed from the matched records at request time. Recommendations are
sentences *quoted* from those records, grouped into categories by keyword;
nothing is generated. Frequencies describe documentation habits in the
library, not whether a recommendation worked.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medxplain.clinical.concepts import CONCEPTS_BY_KEY, concept_label, detect_concepts
from medxplain.clinical.extraction import disposition_category, disposition_spans
from medxplain.clinical.text import parse_note
from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.clinical.cases")

#: concept groups that count as clinically relevant structured match criteria
MATCH_GROUPS = ("acute", "chronic", "goals_of_care")

#: Plan / recommendation statements a clinician documents, grouped for counting.
#: Only sentences from course and discharge sections are considered.
RECOMMENDATION_CATEGORIES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("Outpatient follow-up arranged", r"\bfollow[- ]?up\b|\bappointment\b|\bPCP\b"),
        ("Discharge instructions or return precautions given",
         r"\bdischarge instructions\b|\breturn for\b|\bnew or worsening\b|\breturn precautions\b"),
        ("Physical / occupational therapy involved", r"\bphysical therapy\b|\boccupational therapy\b"),
        ("Dietary recommendation from nutrition", r"\bnutrition\b.*\brecommend\w*|\bheart healthy diet\b"),
        ("Speech and swallow diet recommendation", r"\bswallow\b"),
        ("Wound care recommendation", r"\bwound care\b"),
        ("Pharmacy medication reconciliation", r"\bmedication reconciliation\b"),
        ("Social work / case management for discharge planning", r"\bsocial work\b|\bcase management\b"),
        ("Code status / goals-of-care discussion", r"\bcode status\b|\bgoals of care\b|\bfamily meeting\b"),
        ("Transition to comfort-focused care", r"\bcomfort measures\b|\bCMO\b|\bcomfort[- ]focused\b|\bDNR\b"),
        ("Hospice referral or enrolment", r"\bhospice\b"),
        ("Palliative care involvement", r"\bpalliative care\b"),
        ("Infection-control precautions", r"\bcontact precautions\b"),
        ("Interpreter services used", r"\binterpreter\b"),
    )
)
_RECOMMENDATION_SECTIONS = ("course", "disposition", "condition", "instructions", "followup")

_AGE_RE = re.compile(r"\b(\d{1,3})[- ]?(?:year|yr)s?[- ]?old\b", re.IGNORECASE)
_SEX_RE = re.compile(r"\bSex:\s*([MF])\b|\b\d{1,3}[- ]?year[- ]?old (male|female|man|woman)\b", re.IGNORECASE)


@dataclass
class _Library:
    note_ids: list[str]
    texts: list[str]
    factors: list[set[str]]
    mortality: np.ndarray
    readmission: np.ndarray | None
    ages: np.ndarray | None
    sexes: list[str] | None
    splits: list[str] | None
    vectorizer: object
    matrix: object


class CaseLibrary:
    """Lazily-loaded, thread-safe index over the processed corpus."""

    def __init__(self, corpus_path: Path, dataset_report_path: Path | None = None,
                 max_matches: int = 30, min_similarity: float = 0.10, duplicate_similarity: float = 0.97):
        self.corpus_path = Path(corpus_path)
        self.dataset_report_path = dataset_report_path
        self.max_matches = max_matches
        self.min_similarity = min_similarity
        self.duplicate_similarity = duplicate_similarity
        self._lib: _Library | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # -- loading ---------------------------------------------------------------------
    def source(self) -> dict:
        source = "unknown"
        if self.dataset_report_path and Path(self.dataset_report_path).exists():
            try:
                source = json.loads(Path(self.dataset_report_path).read_text()).get("source", source)
            except Exception:  # pragma: no cover - a malformed report only loses the label
                pass
        synthetic = source == "synthetic"
        if self._lib is not None and self._lib.note_ids and self._lib.note_ids[0].startswith("SYN"):
            synthetic = True
        return {
            "name": source,
            "synthetic": synthetic,
            "label": (
                "Synthetic demonstration data: computer-generated notes, not real patients"
                if synthetic else f"Historical case library ({source})"
            ),
        }

    def _ensure_loaded(self) -> _Library | None:
        if self._lib is not None or self._load_error is not None:
            return self._lib
        with self._lock:
            if self._lib is not None or self._load_error is not None:
                return self._lib
            if not self.corpus_path.exists():
                self._load_error = f"no case library found at {self.corpus_path}"
                return None
            try:
                import pandas as pd
                from sklearn.feature_extraction.text import TfidfVectorizer

                frame = pd.read_parquet(self.corpus_path)
                if "text" not in frame.columns or "mortality" not in frame.columns:
                    self._load_error = "the case library has no 'text' or 'mortality' column"
                    return None
                texts = frame["text"].astype(str).tolist()
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english",
                    token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9/-]+\b",
                )
                matrix = vectorizer.fit_transform(texts)
                self._lib = _Library(
                    note_ids=frame["note_id"].astype(str).tolist() if "note_id" in frame else
                    [str(i) for i in range(len(frame))],
                    texts=texts,
                    factors=[_parse_factors(v) for v in frame["factors"]] if "factors" in frame else
                    [set() for _ in texts],
                    mortality=frame["mortality"].to_numpy(dtype=int),
                    readmission=frame["readmission_30d"].to_numpy(dtype=int)
                    if "readmission_30d" in frame else None,
                    ages=frame["age"].to_numpy(dtype=float) if "age" in frame else None,
                    sexes=frame["sex"].astype(str).tolist() if "sex" in frame else None,
                    splits=frame["split"].astype(str).tolist() if "split" in frame else None,
                    vectorizer=vectorizer,
                    matrix=matrix,
                )
                LOGGER.info("case library loaded: %d notes from %s", len(texts), self.corpus_path)
            except Exception as exc:  # pragma: no cover - reported to the client instead
                LOGGER.exception("could not load case library")
                self._load_error = f"could not load the case library: {exc}"
        return self._lib

    # -- query -----------------------------------------------------------------------
    def query(self, text: str, n_examples: int = 5) -> dict:
        lib = self._ensure_loaded()
        if lib is None:
            return {
                "available": False,
                "reason": self._load_error,
                "data_needed": (
                    "A library of de-identified historical discharge notes with recorded outcomes "
                    "(e.g. in-hospital mortality, 30-day readmission) and, ideally, structured clinical "
                    "fields. Build the demonstration library with scripts/00_build_dataset.py."
                ),
                "source": self.source(),
            }

        note = parse_note(text)
        detected = detect_concepts(note)
        key_factors = sorted(
            k for k in detected if CONCEPTS_BY_KEY[k].group in MATCH_GROUPS
        )
        age, sex = _patient_age(text), _patient_sex(text)

        from sklearn.metrics.pairwise import linear_kernel

        sims = linear_kernel(lib.vectorizer.transform([text]), lib.matrix).ravel()

        n_total = len(lib.texts)
        mask = np.ones(n_total, dtype=bool)
        duplicates = sims >= self.duplicate_similarity
        mask &= ~duplicates
        mask &= sims >= self.min_similarity

        criteria: list[dict] = []
        required = 0
        overlap = np.zeros(n_total, dtype=int)
        if key_factors:
            required = min(2, len(key_factors))
            wanted = set(key_factors)
            overlap = np.array([len(wanted & f) for f in lib.factors])
            mask &= overlap >= required
            criteria.append({
                "type": "structured",
                "label": (
                    f"Recorded with at least {required} of the conditions found in this note"
                    if len(key_factors) > required
                    else "Recorded with the condition(s) found in this note"
                ),
                "detail": ", ".join(concept_label(k) for k in key_factors),
            })
        else:
            criteria.append({
                "type": "structured",
                "label": "No acute or chronic conditions were recognised in this note",
                "detail": "Matching falls back to text similarity alone.",
            })

        age_band = None
        if age is not None and lib.ages is not None:
            band_mask = np.abs(lib.ages - age) <= 10
            if (mask & band_mask).sum() >= 5:
                mask &= band_mask
                age_band = (age - 10, age + 10)
                criteria.append({
                    "type": "structured",
                    "label": f"Age {age - 10}–{age + 10}",
                    "detail": f"The note states the patient is {age}.",
                })
            else:
                criteria.append({
                    "type": "structured",
                    "label": "Age not used",
                    "detail": f"Fewer than 5 cases also fell within 10 years of age {age}, so age was not applied.",
                })

        criteria.append({
            "type": "text",
            "label": f"Similar wording (text similarity of at least {self.min_similarity:.2f})",
            "detail": "TF-IDF cosine similarity over the full note text.",
        })

        candidate_idx = np.flatnonzero(mask)
        jaccard = np.array([
            len(set(key_factors) & lib.factors[i]) / max(len(set(key_factors) | (lib.factors[i] & _match_keys())), 1)
            for i in candidate_idx
        ]) if key_factors else np.zeros(len(candidate_idx))
        score = 0.6 * jaccard + 0.4 * sims[candidate_idx]
        order = candidate_idx[np.argsort(-score, kind="stable")]
        n_eligible = int(len(order))
        matched = order[: self.max_matches]
        if n_eligible > self.max_matches:
            criteria.append({
                "type": "limit",
                "label": f"The {self.max_matches} closest of {n_eligible} eligible cases",
                "detail": "Ranked by condition overlap (60%) and text similarity (40%).",
            })

        source = self.source()
        base = {
            "available": True,
            "source": source,
            "library_size": n_total,
            "excluded_duplicates": int(duplicates.sum()),
            "criteria": criteria,
            "query_concepts": [{"key": k, "label": concept_label(k)} for k in key_factors],
            "patient_age": age,
            "patient_sex": sex,
            "n_eligible": n_eligible,
            "n_matched": int(len(matched)),
        }
        if len(matched) == 0:
            base.update({
                "outcomes": [], "recommendations": [], "examples": [],
                "reason": "No case in the library met the matching criteria.",
            })
            return base

        base["outcomes"] = self._outcomes(lib, matched)
        base["recommendations"] = self._recommendations(lib, matched)
        base["examples"] = [
            self._example(lib, int(i), float(sims[i]), key_factors) for i in matched[:n_examples]
        ]
        return base

    # -- aggregation -----------------------------------------------------------------
    @staticmethod
    def _outcomes(lib: _Library, idx: np.ndarray) -> list[dict]:
        out = []
        died = int(lib.mortality[idx].sum())
        out.append({
            "key": "mortality",
            "label": "Died in hospital",
            "count": died,
            "n": int(len(idx)),
            "rate": died / len(idx),
            "library_rate": float(lib.mortality.mean()),
        })
        if lib.readmission is not None:
            survivors = idx[lib.mortality[idx] == 0]
            if len(survivors):
                readm = int(lib.readmission[survivors].sum())
                out.append({
                    "key": "readmission_30d",
                    "label": "Readmitted within 30 days (among those discharged alive)",
                    "count": readm,
                    "n": int(len(survivors)),
                    "rate": readm / len(survivors),
                    "library_rate": float(lib.readmission[lib.mortality == 0].mean()),
                })
        return out

    @staticmethod
    def _recommendations(lib: _Library, idx: np.ndarray) -> list[dict]:
        counts: dict[str, set[int]] = {}
        quotes: dict[str, list[dict]] = {}
        for i in idx:
            parsed = parse_note(lib.texts[i])
            for sentence in parsed.sentences(*_RECOMMENDATION_SECTIONS):
                for label, pattern in RECOMMENDATION_CATEGORIES:
                    if pattern.search(sentence.text):
                        counts.setdefault(label, set()).add(int(i))
                        bucket = quotes.setdefault(label, [])
                        if len(bucket) < 3 and all(q["text"] != sentence.text for q in bucket):
                            bucket.append({"text": sentence.text, "note_id": lib.note_ids[i]})
        n = len(idx)
        rows = [
            {
                "category": label,
                "n_cases": len(cases),
                "n": n,
                "frequency": len(cases) / n,
                "examples": quotes[label],
            }
            for label, cases in counts.items()
        ]
        return sorted(rows, key=lambda r: -r["n_cases"])

    @staticmethod
    def _example(lib: _Library, i: int, similarity: float, key_factors: list[str]) -> dict:
        parsed = parse_note(lib.texts[i])
        dispo = disposition_spans(parsed)
        return {
            "note_id": lib.note_ids[i],
            "similarity": round(similarity, 3),
            "shared_conditions": [concept_label(k) for k in key_factors if k in lib.factors[i]],
            "age": int(lib.ages[i]) if lib.ages is not None else None,
            "sex": lib.sexes[i] if lib.sexes is not None else None,
            "split": lib.splits[i] if lib.splits is not None else None,
            "died_in_hospital": bool(lib.mortality[i]),
            "readmitted_30d": bool(lib.readmission[i]) if lib.readmission is not None else None,
            "disposition": disposition_category(dispo),
            "text": lib.texts[i],
        }


def _match_keys() -> set[str]:
    return {k for k, c in CONCEPTS_BY_KEY.items() if c.group in MATCH_GROUPS}


def _parse_factors(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [v.strip() for v in value.split(",") if v.strip()]
    try:
        return {str(v) for v in value}
    except TypeError:
        return set()


def _patient_age(text: str) -> int | None:
    m = _AGE_RE.search(text)
    if not m:
        return None
    age = int(m.group(1))
    return age if 0 < age < 120 else None


def _patient_sex(text: str) -> str | None:
    m = _SEX_RE.search(text)
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or "").lower()
    return "F" if raw in ("f", "female", "woman") else "M" if raw in ("m", "male", "man") else None
