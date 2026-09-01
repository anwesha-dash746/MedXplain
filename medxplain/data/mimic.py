"""MIMIC-III cohort construction.

MIMIC-III cannot be redistributed, so this module reads the CSVs from a local
PhysioNet download and emits exactly the same schema the synthetic generator
produces (``note_id, text, mortality, readmission_30d, subject_id, ...``). Every
downstream stage is therefore dataset-agnostic: swapping the prototype corpus
for the real cohort is a config change, not a code change.

Cohort definition (documented here because reviewers will ask for it verbatim)
-----------------------------------------------------------------------------
* **Unit of analysis**: one hospital admission (``HADM_ID``).
* **Note selection**: ``NOTEEVENTS`` rows with ``CATEGORY == 'Discharge summary'``.
  Where an admission has an addendum, the report and addenda are concatenated in
  ``CHARTDATE`` order. Rows with ``ISERROR`` set are dropped.
* **Primary outcome — in-hospital mortality**: ``ADMISSIONS.HOSPITAL_EXPIRE_FLAG``.
* **Secondary outcome — 30-day readmission**: an unplanned admission for the same
  ``SUBJECT_ID`` beginning within 30 days of the index discharge. Admissions of
  type ``ELECTIVE`` are not counted as readmissions, and index admissions ending
  in death are excluded from the readmission cohort (they cannot be readmitted).
* **Exclusions**: age < 18 at admission; admissions with no discharge summary;
  organ-donor accounts (``DIAGNOSIS`` containing 'ORGAN DONOR').

Leakage control
---------------
The discharge summary is written at the end of the stay and, for mortality,
contains the outcome explicitly ("patient expired", "pronounced dead", time of
death). Training on that is label leakage. :func:`mask_outcome_leakage`
removes those phrases, and the flag that controls it is recorded in the results
file so the ablation ("with vs. without leakage masking") can be reported —
this is a known pitfall in MIMIC mortality papers and is worth stating openly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.data.mimic")


@dataclass
class MimicConfig:
    root: Path
    min_age: int = 18
    readmission_window_days: int = 30
    mask_leakage: bool = True
    min_chars: int = 200
    #: cap on notes, useful for a fast pilot run on the real cohort
    limit: int | None = None


#: Phrases that state the mortality outcome directly. Replaced with a neutral
#: token rather than deleted, so the surrounding narrative stays readable.
LEAKAGE_PATTERNS: tuple[str, ...] = (
    r"\bpatient\s+(?:was\s+)?pronounced\s+dead\b",
    r"\bpronounced\s+(?:dead|deceased)\b",
    r"\btime\s+of\s+death\b",
    r"\bpatient\s+expired\b",
    r"\bexpired\s+on\s+(?:the\s+)?(?:morning|evening|night)?\b",
    r"\bdeath\s+was\s+pronounced\b",
    r"\bpatient\s+(?:passed\s+away|deceased)\b",
    r"\bcause\s+of\s+death\b",
    r"\bautopsy\b",
    r"\bpost[- ]mortem\b",
    r"\bexpiration\s+note\b",
    r"\bdischarge\s+disposition:\s*expired\b",
)
_LEAKAGE_RE = re.compile("|".join(LEAKAGE_PATTERNS), flags=re.IGNORECASE)


def mask_outcome_leakage(text: str, token: str = "<REDACTED-OUTCOME>") -> str:
    """Blank out phrases that state the mortality label outright."""
    return _LEAKAGE_RE.sub(token, text)


def _read_csv(root: Path, name: str, **kwargs) -> pd.DataFrame:
    """Read a MIMIC table, tolerating the .csv / .csv.gz packaging variants."""
    for candidate in (root / f"{name}.csv", root / f"{name}.csv.gz", root / f"{name.lower()}.csv"):
        if candidate.exists():
            LOGGER.info("reading %s", candidate.name)
            return pd.read_csv(candidate, **kwargs)
    raise FileNotFoundError(
        f"{name}.csv not found under {root}. Point --mimic-root at the directory "
        "containing ADMISSIONS.csv, PATIENTS.csv and NOTEEVENTS.csv."
    )


def build_cohort(cfg: MimicConfig) -> pd.DataFrame:
    """Build the admission-level cohort with both outcome labels."""
    root = Path(cfg.root)
    admissions = _read_csv(
        root,
        "ADMISSIONS",
        parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"],
        usecols=[
            "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME",
            "ADMISSION_TYPE", "HOSPITAL_EXPIRE_FLAG", "DIAGNOSIS",
        ],
    )
    patients = _read_csv(root, "PATIENTS", parse_dates=["DOB"], usecols=["SUBJECT_ID", "DOB", "GENDER"])

    df = admissions.merge(patients, on="SUBJECT_ID", how="left")

    # MIMIC-III shifts DOB for patients over 89; those ages come out ~300 and are
    # conventionally reported as 90.
    age_days = (df["ADMITTIME"] - df["DOB"]).dt.total_seconds() / (3600 * 24)
    df["age"] = np.floor(age_days / 365.242).clip(lower=0)
    df.loc[df["age"] > 89, "age"] = 90

    before = len(df)
    df = df[df["age"] >= cfg.min_age]
    df = df[~df["DIAGNOSIS"].fillna("").str.contains("ORGAN DONOR", case=False)]
    LOGGER.info("age/donor exclusions removed %d admissions", before - len(df))

    df["mortality"] = df["HOSPITAL_EXPIRE_FLAG"].astype(int)
    df["readmission_30d"] = _readmission_labels(df, cfg.readmission_window_days)

    notes = _load_discharge_summaries(root, limit=cfg.limit)
    df = df.merge(notes, on="HADM_ID", how="inner")
    df = df[df["text"].str.len() >= cfg.min_chars]

    if cfg.mask_leakage:
        df["text"] = df["text"].map(mask_outcome_leakage)

    out = pd.DataFrame(
        {
            "note_id": df["HADM_ID"].astype(str).map(lambda h: f"MIMIC{h}"),
            "subject_id": df["SUBJECT_ID"].astype(int),
            "hadm_id": df["HADM_ID"].astype(int),
            "text": df["text"],
            "mortality": df["mortality"].astype(int),
            "readmission_30d": df["readmission_30d"].astype(int),
            "age": df["age"].astype(int),
            "sex": df["GENDER"].fillna("U"),
            "admission_type": df["ADMISSION_TYPE"],
            "los_days": (df["DISCHTIME"] - df["ADMITTIME"]).dt.total_seconds() / 86400,
        }
    ).reset_index(drop=True)

    LOGGER.info(
        "cohort: %d admissions | mortality %.3f | readmission %.3f",
        len(out), out["mortality"].mean(), out["readmission_30d"].mean(),
    )
    return out


def _readmission_labels(df: pd.DataFrame, window_days: int) -> np.ndarray:
    """Unplanned readmission within ``window_days`` of discharge."""
    ordered = df.sort_values(["SUBJECT_ID", "ADMITTIME"]).copy()
    ordered["next_admit"] = ordered.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
    ordered["next_type"] = ordered.groupby("SUBJECT_ID")["ADMISSION_TYPE"].shift(-1)

    # elective follow-up admissions are planned care, not readmission failures
    elective = ordered["next_type"].eq("ELECTIVE")
    ordered.loc[elective, "next_admit"] = pd.NaT

    gap = (ordered["next_admit"] - ordered["DISCHTIME"]).dt.total_seconds() / 86400
    label = (gap >= 0) & (gap <= window_days)
    # a patient who died in hospital cannot be readmitted
    label &= ordered["HOSPITAL_EXPIRE_FLAG"].eq(0)
    return label.astype(int).reindex(df.index).to_numpy()


def _load_discharge_summaries(root: Path, limit: int | None = None) -> pd.DataFrame:
    """One concatenated discharge summary per admission."""
    notes = _read_csv(
        root,
        "NOTEEVENTS",
        usecols=["HADM_ID", "CHARTDATE", "CATEGORY", "DESCRIPTION", "ISERROR", "TEXT"],
        low_memory=False,
    )
    notes = notes[notes["CATEGORY"].str.strip().eq("Discharge summary")]
    notes = notes[notes["ISERROR"].isna()]
    notes = notes.dropna(subset=["HADM_ID"])
    notes = notes.sort_values(["HADM_ID", "CHARTDATE"])
    if limit is not None:
        keep = notes["HADM_ID"].drop_duplicates().head(limit)
        notes = notes[notes["HADM_ID"].isin(keep)]
    grouped = notes.groupby("HADM_ID")["TEXT"].apply(lambda s: "\n\n".join(s.astype(str)))
    return grouped.rename("text").reset_index().assign(HADM_ID=lambda d: d["HADM_ID"].astype(int))
