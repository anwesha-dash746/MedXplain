"""Synthetic discharge-summary generator used for prototyping and for the
explanation-quality experiments that need gold rationales.

Why this exists
---------------
MIMIC-III access takes weeks and cannot be redistributed, so the repository has
to be runnable by anyone who clones it. This module produces notes that imitate
the surface structure of MIMIC-III discharge summaries (section headers,
de-identification placeholders, abbreviations, boilerplate) and carry a *known*
generative process:

    logit = intercept + age_term + sum(weight_f for each present risk factor f)
    label ~ Bernoulli(sigmoid(logit))

Because the label is sampled rather than thresholded, the Bayes-optimal AUROC is
below 1.0, so models cannot trivially saturate the benchmark.

Two properties make this corpus useful beyond a smoke test:

1.  Every injected risk-factor sentence is recorded with its character span and
    its true log-odds weight. Those spans are gold rationales, which lets us
    score explanation *plausibility* automatically (token-F1 / AUPRC against
    the gold spans) rather than only by eye.
2.  Distractor sentences with zero weight are injected in every section, so an
    explainer that latches onto section boilerplate is measurably penalised.

Nothing here is a substitute for real clinical data: the generator is a
prototype harness, and headline results should be reported on MIMIC-III.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from medxplain.data.clinical_lexicon import (
    ALL_FACTORS,
    DISTRACTORS,
    SECTION_HEADERS,
    SECTION_ORDER,
    RiskFactor,
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class SyntheticCorpusConfig:
    """Knobs for :func:`generate_corpus`."""

    n_notes: int = 6000
    seed: int = 13
    #: shifts the intercept so the realised positive rate lands near this value
    target_positive_rate: float = 0.16
    target_readmission_rate: float = 0.20
    #: extra unexplained variance, so text does not fully determine the label
    label_noise: float = 0.05
    min_distractors: int = 3
    max_distractors: int = 7
    #: fraction of notes that also receive filler paragraphs to stress long-note handling
    long_note_fraction: float = 0.25
    n_med_lines: tuple[int, int] = (4, 9)


@dataclass
class GoldSpan:
    """A rationale span: where a risk factor appears and how much it truly matters."""

    start: int
    end: int
    factor: str
    text: str
    mortality_weight: float
    readmission_weight: float


@dataclass
class SyntheticNote:
    note_id: str
    text: str
    mortality: int
    readmission_30d: int
    mortality_logit: float
    readmission_logit: float
    age: int
    sex: str
    factors: list[str]
    gold_spans: list[GoldSpan] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "note_id": self.note_id,
            "text": self.text,
            "mortality": self.mortality,
            "readmission_30d": self.readmission_30d,
            "mortality_logit": round(self.mortality_logit, 4),
            "readmission_logit": round(self.readmission_logit, 4),
            "age": self.age,
            "sex": self.sex,
            "factors": list(self.factors),
            "gold_spans": [
                {
                    "start": s.start,
                    "end": s.end,
                    "factor": s.factor,
                    "text": s.text,
                    "mortality_weight": s.mortality_weight,
                    "readmission_weight": s.readmission_weight,
                }
                for s in self.gold_spans
            ],
        }


# --------------------------------------------------------------------------------------
# Surface realisation helpers
# --------------------------------------------------------------------------------------

_SERVICES = (
    "MEDICINE", "SURGERY", "CARDIOTHORACIC", "NEUROLOGY", "MICU", "CCU", "TRAUMA SICU",
)
_CHIEF_COMPLAINTS = (
    "shortness of breath", "chest pain", "altered mental status", "fever and chills",
    "abdominal pain", "weakness and falls", "hypotension", "nausea and vomiting",
    "productive cough", "syncope",
)
_ALLERGIES = ("Penicillins", "Sulfa (Sulfonamide Antibiotics)", "No Known Allergies", "Codeine", "Iodine")
_FILLER = (
    "The patient was maintained on telemetry monitoring for the duration of the stay "
    "without any documented arrhythmic events.",
    "Serial abdominal examinations were performed and remained benign throughout.",
    "Daily weights and strict intake and output were recorded by the nursing staff.",
    "The primary team communicated with the family by telephone on several occasions "
    "to provide updates on the clinical trajectory.",
    "Speech and swallow evaluation recommended a dysphagia level 2 diet with thin liquids.",
    "Wound care was consulted and recommended dressing changes every other day.",
    "Pharmacy performed medication reconciliation on admission and again prior to discharge.",
    "Glycemic control was managed with a sliding scale insulin protocol and fingersticks "
    "before meals and at bedtime.",
    "Venous thromboembolism prophylaxis was continued throughout the admission per protocol.",
    "The patient was evaluated by social work regarding disposition planning and insurance "
    "coverage for post-acute services.",
    "Repeat imaging was obtained on the third hospital day and compared with the prior study; "
    "findings were discussed with the covering team.",
    "Interpreter services were used for all substantive discussions with the patient and family.",
    "Code status was addressed on admission and revisited during the stay.",
    "The patient remained on contact precautions pending clearance of surveillance cultures.",
)


def _deid(rng: random.Random) -> dict[str, str]:
    """MIMIC-style de-identification placeholders."""
    return {
        "date": f"[**{rng.randint(2100, 2200)}-{rng.randint(1, 12)}-{rng.randint(1, 28)}**]",
        "hospital": f"[**Hospital1 {rng.randint(1, 99)}**]",
        "name": f"[**Known lastname {rng.randint(100, 9999)}**]",
        "location": f"[**Location (un) {rng.randint(10, 999)}**]",
        "doctor": f"[**First Name8 (NamePattern2) **] [**Last Name (NamePattern1) {rng.randint(10, 999)}**]",
    }


def _pick_factors(rng: random.Random) -> list[RiskFactor]:
    """Sample which risk factors this admission has, with mild positive correlation.

    Real patients cluster: someone in septic shock is more likely to also be
    ventilated. We model that with a per-patient acuity multiplier rather than
    a full covariance structure, which is enough to keep the task non-trivially
    separable while staying interpretable.
    """
    acuity = rng.lognormvariate(0.0, 0.55)
    chosen: list[RiskFactor] = []
    for factor in ALL_FACTORS:
        p = factor.prevalence
        if factor.mortality_weight > 0:
            p = min(0.95, p * acuity)
        else:  # protective findings are less likely in a high-acuity admission
            p = min(0.95, p / max(acuity, 1e-6))
        if rng.random() < p:
            chosen.append(factor)
    # guarantee at least one signal-bearing sentence so no note is pure boilerplate
    if not chosen:
        chosen.append(rng.choice(ALL_FACTORS))
    return chosen


def _age_terms(age: int) -> tuple[float, float]:
    """Age contribution to (mortality, readmission) log-odds, centred at 65."""
    mortality = 0.030 * (age - 65)
    readmission = 0.012 * (age - 65)
    return mortality, readmission


def _build_sections(
    rng: random.Random,
    cfg: SyntheticCorpusConfig,
    factors: Sequence[RiskFactor],
    age: int,
    sex: str,
    deid: dict[str, str],
) -> tuple[dict[str, list[tuple[str, RiskFactor | None]]], str]:
    """Assemble per-section sentence lists, each tagged with its factor (or None)."""
    sections: dict[str, list[tuple[str, RiskFactor | None]]] = {s: [] for s in SECTION_ORDER}

    complaint = rng.choice(_CHIEF_COMPLAINTS)
    sections["hpi"].append(
        (
            f"Mr./Ms. {deid['name']} is a {age} year old {'male' if sex == 'M' else 'female'} "
            f"who presented to the {deid['hospital']} emergency department on {deid['date']} "
            f"with {complaint}.",
            None,
        )
    )

    for factor in factors:
        # The surface form is never reworded, so the gold rationale span is
        # exactly the injected phrase; variety comes from the multiple forms per
        # factor and from sentence shuffling rather than from generated glue text.
        form = rng.choice(factor.surface_forms)
        sentence = form[0].upper() + form[1:] + "."
        sections[factor.section].append((sentence, factor))

    # vitals line, always present, mostly neutral
    sections["exam"].insert(
        0,
        (
            f"Vitals on admission: T {rng.uniform(36.0, 39.4):.1f} C, HR {rng.randint(58, 132)}, "
            f"BP {rng.randint(82, 168)}/{rng.randint(44, 96)}, RR {rng.randint(12, 34)}, "
            f"SpO2 {rng.randint(84, 100)} percent on {rng.choice(['room air', '2L NC', '6L NC', 'NRB'])}.",
            None,
        ),
    )
    sections["labs"].insert(
        0,
        (
            f"WBC {rng.uniform(2.0, 26.0):.1f}, Hct {rng.uniform(20.0, 45.0):.1f}, "
            f"Plt {rng.randint(40, 420)}, Cr {rng.uniform(0.5, 6.4):.1f}, "
            f"Na {rng.randint(126, 148)}, K {rng.uniform(2.9, 6.1):.1f}.",
            None,
        ),
    )

    # distractors
    n_dist = rng.randint(cfg.min_distractors, cfg.max_distractors)
    for _ in range(n_dist):
        sec = rng.choice(SECTION_ORDER)
        sections[sec].append((rng.choice(DISTRACTORS[sec]), None))

    # discharge medication list (shuffled first, then numbered, so the list reads
    # in order the way a real discharge summary does)
    n_meds = rng.randint(*cfg.n_med_lines)
    meds = rng.sample(DISTRACTORS["meds"], k=min(n_meds, len(DISTRACTORS["meds"])))
    sections["meds"] = [(f"{i + 1}. {m}", None) for i, m in enumerate(meds)]

    if rng.random() < cfg.long_note_fraction:
        for filler in rng.sample(_FILLER, k=rng.randint(4, 9)):
            sections["course"].append((filler, None))

    # interleave signal and distractor sentences so position is not a shortcut,
    # keeping the opening line of each section pinned in place
    for sec in ("hpi", "exam", "labs", "course", "disposition"):
        if len(sections[sec]) > 2:
            head, tail = sections[sec][:1], sections[sec][1:]
            rng.shuffle(tail)
            sections[sec] = head + tail

    header = (
        f"Admission Date: {deid['date']}       Discharge Date: {deid['date']}\n"
        f"Date of Birth: {deid['date']}        Sex: {sex}\n"
        f"Service: {rng.choice(_SERVICES)}\n"
        f"Allergies: {rng.choice(_ALLERGIES)}\n"
        f"Attending: {deid['doctor']}\n"
        f"Chief Complaint: {complaint}\n"
    )
    return sections, header


def _render(
    sections: dict[str, list[tuple[str, RiskFactor | None]]], header: str
) -> tuple[str, list[GoldSpan]]:
    """Flatten sections into a note, recording character spans of factor sentences."""
    parts: list[str] = [header]
    spans: list[GoldSpan] = []
    cursor = len(header)

    for sec in SECTION_ORDER:
        if not sections[sec]:
            continue
        title = f"\n{SECTION_HEADERS[sec]}:\n"
        parts.append(title)
        cursor += len(title)
        for sentence, factor in sections[sec]:
            line = sentence + ("\n" if sec == "meds" else " ")
            if factor is not None:
                start = cursor
                end = cursor + len(sentence)
                spans.append(
                    GoldSpan(
                        start=start,
                        end=end,
                        factor=factor.key,
                        text=sentence,
                        mortality_weight=factor.mortality_weight,
                        readmission_weight=factor.readmission_weight,
                    )
                )
            parts.append(line)
            cursor += len(line)
        parts.append("\n")
        cursor += 1

    text = "".join(parts)
    # sanity: spans must line up with the rendered text
    for span in spans:
        assert text[span.start : span.end] == span.text, "gold span misaligned"
    return text, spans


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calibrate_intercept(
    logits: Sequence[float], target_rate: float, lo: float = -12.0, hi: float = 12.0
) -> float:
    """Bisect for the intercept that makes mean sigmoid(logit + b) == target_rate."""
    for _ in range(80):
        mid = (lo + hi) / 2
        rate = sum(_sigmoid(z + mid) for z in logits) / len(logits)
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def generate_corpus(cfg: SyntheticCorpusConfig | None = None) -> list[dict]:
    """Generate ``cfg.n_notes`` synthetic discharge summaries with gold rationales."""
    cfg = cfg or SyntheticCorpusConfig()
    rng = random.Random(cfg.seed)

    drafts: list[tuple[SyntheticNote, float, float]] = []
    for i in range(cfg.n_notes):
        age = int(min(95, max(19, rng.gauss(66, 16))))
        sex = rng.choice(("M", "F"))
        factors = _pick_factors(rng)
        deid = _deid(rng)
        sections, header = _build_sections(rng, cfg, factors, age, sex, deid)
        text, spans = _render(sections, header)

        age_m, age_r = _age_terms(age)
        mort_logit = age_m + sum(f.mortality_weight for f in factors)
        readm_logit = age_r + sum(f.readmission_weight for f in factors)
        mort_logit += rng.gauss(0.0, cfg.label_noise * 10)
        readm_logit += rng.gauss(0.0, cfg.label_noise * 10)

        note = SyntheticNote(
            note_id=f"SYN{i:06d}",
            text=text,
            mortality=0,
            readmission_30d=0,
            mortality_logit=mort_logit,
            readmission_logit=readm_logit,
            age=age,
            sex=sex,
            factors=[f.key for f in factors],
            gold_spans=spans,
        )
        drafts.append((note, mort_logit, readm_logit))

    # calibrate prevalence to the configured base rates, then sample labels
    b_m = _calibrate_intercept([d[1] for d in drafts], cfg.target_positive_rate)
    b_r = _calibrate_intercept([d[2] for d in drafts], cfg.target_readmission_rate)

    rows: list[dict] = []
    for note, ml, rl in drafts:
        note.mortality_logit = ml + b_m
        note.readmission_logit = rl + b_r
        note.mortality = int(rng.random() < _sigmoid(note.mortality_logit))
        note.readmission_30d = int(rng.random() < _sigmoid(note.readmission_logit))
        rows.append(note.to_row())
    return rows


# --------------------------------------------------------------------------------------
# Paraphrasing, for the explanation-stability evaluation
# --------------------------------------------------------------------------------------

_GENERIC_PARAPHRASES: tuple[tuple[str, str], ...] = (
    ("The patient", "The pt"),
    ("presented to", "arrived at"),
    ("was admitted", "was brought in"),
    ("year old", "y/o"),
    ("emergency department", "ED"),
    ("hospital course", "inpatient course"),
    ("was started on", "was initiated on"),
    ("follow up", "follow-up"),
    ("discharged home", "sent home"),
    ("no evidence of", "no signs of"),
)


def paraphrase_note(text: str, seed: int = 0, max_edits: int = 8) -> str:
    """Meaning-preserving rewrite: abbreviation swaps and equivalent phrasings.

    Used by the stability evaluation — a faithful explainer should assign nearly
    the same importance to the same clinical content after these edits.
    """
    rng = random.Random(seed)
    candidates: list[tuple[str, str]] = list(_GENERIC_PARAPHRASES)
    for factor in ALL_FACTORS:
        candidates.extend(factor.paraphrases)
    rng.shuffle(candidates)

    edits = 0
    out = text
    for src, dst in candidates:
        if edits >= max_edits:
            break
        if src in out:
            out = out.replace(src, dst, 1)
            edits += 1
    return out


def near_duplicate_note(text: str, seed: int = 0) -> str:
    """Formatting-only perturbation (whitespace/casing), semantics untouched."""
    rng = random.Random(seed)
    lines = text.split("\n")
    out = []
    for line in lines:
        if rng.random() < 0.25:
            line = "  " + line
        if rng.random() < 0.10 and line and line[0].isalpha():
            line = line[0].lower() + line[1:]
        out.append(line)
    return "\n".join(out)


def iter_examples(rows: Iterable[dict], n: int = 6) -> list[dict]:
    """Pick a spread of low/medium/high-risk notes for the frontend case-study mode."""
    ranked = sorted(rows, key=lambda r: r["mortality_logit"])
    if len(ranked) <= n:
        return list(ranked)
    idx = [int(i * (len(ranked) - 1) / (n - 1)) for i in range(n)]
    return [ranked[i] for i in idx]
