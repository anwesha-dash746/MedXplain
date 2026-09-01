"""Clinical vocabulary used by the synthetic note generator.

Each *risk factor* carries a log-odds weight toward in-hospital mortality and a
separate weight toward 30-day readmission, plus the surface forms that express
it in a discharge summary. The surface forms are what a faithful explanation
should highlight, so they double as gold rationales on the prototype corpus.

Weights are hand-set to be clinically plausible in *direction and relative
magnitude* (loosely informed by APACHE/SOFA-style severity scoring and the
readmission literature), not calibrated to any real cohort. They exist to make
the prototype corpus learnable and its explanations checkable; every number
reported for a real cohort comes from MIMIC-III instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskFactor:
    """A clinical concept that shifts outcome log-odds and has textual surface forms."""

    key: str
    section: str
    mortality_weight: float
    readmission_weight: float
    prevalence: float
    surface_forms: tuple[str, ...]
    #: equivalent phrasings, used to build paraphrases for the stability evaluation
    paraphrases: tuple[tuple[str, str], ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------------------
# High-severity acute findings (strong mortality signal)
# --------------------------------------------------------------------------------------
ACUTE_SEVERITY: tuple[RiskFactor, ...] = (
    RiskFactor(
        "septic_shock", "course", 1.65, 0.35, 0.11,
        (
            "patient developed septic shock requiring norepinephrine",
            "course complicated by septic shock with escalating pressor requirement",
            "septic shock secondary to urosepsis, started on levophed",
        ),
        (("septic shock", "distributive shock from sepsis"), ("norepinephrine", "levophed")),
    ),
    RiskFactor(
        "vasopressors", "course", 1.05, 0.20, 0.13,
        (
            "required vasopressor support for persistent hypotension",
            "on multiple pressors including vasopressin and phenylephrine",
            "unable to wean off pressors during the ICU stay",
        ),
        (("vasopressor support", "pressor support"),),
    ),
    RiskFactor(
        "mech_vent", "course", 1.20, 0.25, 0.16,
        (
            "intubated on hospital day 1 for hypoxic respiratory failure",
            "mechanically ventilated with FiO2 requirement of 80 percent",
            "prolonged mechanical ventilation with failed extubation trial",
        ),
        (("intubated", "endotracheally intubated"), ("mechanically ventilated", "on mechanical ventilation")),
    ),
    RiskFactor(
        "cardiac_arrest", "course", 1.95, 0.15, 0.05,
        (
            "PEA arrest with ROSC after 12 minutes of CPR",
            "suffered an in-hospital cardiac arrest requiring ACLS",
            "VF arrest, defibrillated twice with return of spontaneous circulation",
        ),
        (("cardiac arrest", "cardiopulmonary arrest"), ("CPR", "cardiopulmonary resuscitation")),
    ),
    RiskFactor(
        "aki_dialysis", "course", 0.95, 0.55, 0.10,
        (
            "acute kidney injury progressing to anuria, CVVH initiated",
            "oliguric renal failure requiring urgent hemodialysis",
            "AKI on CKD with initiation of renal replacement therapy",
        ),
        (("AKI", "acute kidney injury"), ("hemodialysis", "HD")),
    ),
    RiskFactor(
        "ards", "course", 1.15, 0.20, 0.06,
        (
            "developed ARDS with P/F ratio of 90 requiring proning",
            "bilateral infiltrates consistent with acute respiratory distress syndrome",
        ),
        (("ARDS", "acute respiratory distress syndrome"),),
    ),
    RiskFactor(
        "gi_bleed", "course", 0.70, 0.45, 0.07,
        (
            "massive upper GI bleed requiring 6 units of PRBC",
            "recurrent hematemesis with hemodynamic instability",
        ),
        (("GI bleed", "gastrointestinal hemorrhage"), ("PRBC", "packed red blood cells")),
    ),
    RiskFactor(
        "encephalopathy", "course", 0.60, 0.30, 0.14,
        (
            "persistent encephalopathy, unresponsive to verbal stimuli",
            "obtunded with GCS of 7 on admission",
            "acute metabolic encephalopathy without return to baseline mental status",
        ),
        (("encephalopathy", "altered mental status"), ("obtunded", "poorly responsive")),
    ),
    RiskFactor(
        "lactate", "labs", 0.85, 0.10, 0.12,
        (
            "lactate peaked at 7.8 mmol/L",
            "serum lactate 6.2 with a persistent anion gap acidosis",
        ),
        (("lactate", "serum lactic acid"),),
    ),
    RiskFactor(
        "hypotension", "exam", 0.55, 0.20, 0.15,
        (
            "MAP in the 50s despite aggressive fluid resuscitation",
            "hypotensive to 78/40 on arrival to the floor",
        ),
        (("hypotensive", "with low blood pressure"), ("MAP", "mean arterial pressure")),
    ),
)

# --------------------------------------------------------------------------------------
# Chronic comorbidity burden (moderate mortality signal, strong readmission signal)
# --------------------------------------------------------------------------------------
COMORBIDITIES: tuple[RiskFactor, ...] = (
    RiskFactor(
        "metastatic_cancer", "hpi", 1.35, 0.60, 0.07,
        (
            "metastatic adenocarcinoma with widespread hepatic involvement",
            "stage IV non-small cell lung cancer on palliative chemotherapy",
            "known metastatic disease to bone and brain",
        ),
        (("metastatic", "widely disseminated"),),
    ),
    RiskFactor(
        "chf", "hpi", 0.45, 0.85, 0.18,
        (
            "history of chronic systolic heart failure with EF of 25 percent",
            "known CHF with recurrent volume overload admissions",
            "decompensated heart failure, NYHA class IV",
        ),
        (("CHF", "congestive heart failure"), ("EF", "ejection fraction")),
    ),
    RiskFactor(
        "copd", "hpi", 0.35, 0.70, 0.17,
        (
            "severe COPD on 3L home oxygen",
            "COPD with frequent exacerbations, GOLD stage IV",
        ),
        (("COPD", "chronic obstructive pulmonary disease"), ("home oxygen", "home O2")),
    ),
    RiskFactor(
        "cirrhosis", "hpi", 0.95, 0.65, 0.06,
        (
            "decompensated cirrhosis with refractory ascites",
            "Child-Pugh C cirrhosis complicated by hepatic encephalopathy",
        ),
        (("cirrhosis", "end-stage liver disease"),),
    ),
    RiskFactor(
        "esrd", "hpi", 0.60, 0.80, 0.08,
        (
            "ESRD on MWF hemodialysis",
            "end stage renal disease with a failing AV fistula",
        ),
        (("ESRD", "end stage renal disease"),),
    ),
    RiskFactor(
        "diabetes", "hpi", 0.15, 0.35, 0.28,
        (
            "insulin dependent diabetes mellitus type 2, poorly controlled",
            "IDDM with peripheral neuropathy and retinopathy",
        ),
        (("diabetes mellitus", "DM"), ("insulin dependent", "insulin requiring")),
    ),
    RiskFactor(
        "dementia", "hpi", 0.55, 0.45, 0.09,
        (
            "advanced dementia, nonverbal at baseline",
            "Alzheimer type dementia with recurrent aspiration",
        ),
        (("dementia", "major neurocognitive disorder"),),
    ),
    RiskFactor(
        "immunosuppressed", "hpi", 0.70, 0.40, 0.06,
        (
            "on chronic immunosuppression following renal transplant",
            "neutropenic on tacrolimus and mycophenolate",
        ),
        (("immunosuppression", "immunosuppressive therapy"),),
    ),
)

# --------------------------------------------------------------------------------------
# Disposition / care-goal statements (very strong mortality signal in real notes)
# --------------------------------------------------------------------------------------
GOALS_OF_CARE: tuple[RiskFactor, ...] = (
    RiskFactor(
        "comfort_measures", "disposition", 2.10, -0.90, 0.06,
        (
            "family meeting held and goals of care transitioned to comfort measures only",
            "CMO status with palliative care following",
            "made DNR/DNI and transitioned to comfort-focused care",
        ),
        (("comfort measures only", "CMO"), ("DNR/DNI", "do not resuscitate and do not intubate")),
    ),
    RiskFactor(
        "hospice", "disposition", 1.55, -0.70, 0.05,
        (
            "discharged to inpatient hospice",
            "hospice consult obtained, patient enrolled prior to discharge",
        ),
        (("hospice", "end-of-life care"),),
    ),
)

# --------------------------------------------------------------------------------------
# Protective / reassuring findings (negative weight)
# --------------------------------------------------------------------------------------
PROTECTIVE: tuple[RiskFactor, ...] = (
    RiskFactor(
        "stable_course", "course", -0.85, -0.55, 0.30,
        (
            "hospital course was uncomplicated and the patient remained hemodynamically stable",
            "afebrile with stable vital signs throughout the admission",
            "tolerated the procedure well with no perioperative complications",
        ),
        (("hemodynamically stable", "hemodynamically intact"),),
    ),
    RiskFactor(
        "ambulating", "disposition", -0.70, -0.60, 0.26,
        (
            "ambulating independently and tolerating a regular diet at discharge",
            "working with physical therapy, cleared to go home without services",
        ),
        (("ambulating independently", "walking without assistance"),),
    ),
    RiskFactor(
        "discharge_home", "disposition", -0.60, -0.35, 0.34,
        (
            "discharged home in stable condition with outpatient follow up in 2 weeks",
            "discharged to home with family, follow up arranged with PCP",
        ),
        (("PCP", "primary care physician"),),
    ),
    RiskFactor(
        "normal_labs", "labs", -0.40, -0.25, 0.30,
        (
            "white count normalized to 7.2 and lactate cleared to 1.1",
            "renal function returned to baseline with creatinine of 0.9",
        ),
        (("white count", "WBC"),),
    ),
)

# --------------------------------------------------------------------------------------
# Readmission-specific drivers (weak on mortality, strong on readmission)
# --------------------------------------------------------------------------------------
READMISSION_DRIVERS: tuple[RiskFactor, ...] = (
    RiskFactor(
        "poor_adherence", "hpi", 0.10, 0.95, 0.13,
        (
            "multiple missed clinic appointments and reported medication non-adherence",
            "has not been taking the prescribed diuretics",
        ),
        (("non-adherence", "noncompliance"),),
    ),
    RiskFactor(
        "prior_admissions", "hpi", 0.20, 1.05, 0.15,
        (
            "three admissions in the past six months for the same complaint",
            "frequent readmissions over the last year",
        ),
        (("readmissions", "return hospitalizations"),),
    ),
    RiskFactor(
        "no_support", "disposition", 0.25, 0.85, 0.10,
        (
            "lives alone with no reliable social support at home",
            "unstable housing situation, no caregiver available",
        ),
        (("social support", "social supports"),),
    ),
    RiskFactor(
        "substance_use", "hpi", 0.30, 0.75, 0.11,
        (
            "ongoing alcohol use disorder with prior withdrawal seizures",
            "active polysubstance use, left AMA during a prior admission",
        ),
        (("AMA", "against medical advice"),),
    ),
)

ALL_FACTORS: tuple[RiskFactor, ...] = (
    ACUTE_SEVERITY + COMORBIDITIES + GOALS_OF_CARE + PROTECTIVE + READMISSION_DRIVERS
)

FACTORS_BY_KEY: dict[str, RiskFactor] = {f.key: f for f in ALL_FACTORS}

# --------------------------------------------------------------------------------------
# Distractor content: clinically real but outcome-neutral (weight 0). A model that
# highlights these instead of the factors above is producing implausible rationales.
# --------------------------------------------------------------------------------------
DISTRACTORS: dict[str, tuple[str, ...]] = {
    "hpi": (
        "The patient reports no recent travel or sick contacts.",
        "Family history is notable for hypertension in both parents.",
        "Patient is a retired schoolteacher who enjoys gardening.",
        "Immunizations are up to date including annual influenza vaccine.",
        "No known drug allergies documented in the chart.",
    ),
    "exam": (
        "HEENT: normocephalic, atraumatic, pupils equal and reactive.",
        "Skin: warm and dry without rashes or lesions.",
        "Extremities: no clubbing or cyanosis noted.",
        "Neck: supple, no lymphadenopathy appreciated.",
    ),
    "labs": (
        "Chest radiograph reviewed with the radiology team.",
        "EKG showed normal sinus rhythm at a rate of 78.",
        "Blood cultures were drawn prior to antibiotic administration.",
        "Urinalysis was sent from a clean catch specimen.",
    ),
    "course": (
        "Physical therapy and occupational therapy were consulted early in the stay.",
        "Nutrition evaluated the patient and recommended a heart healthy diet.",
        "Case management followed throughout the admission.",
        "The patient was seen daily by the primary team.",
    ),
    "meds": (
        "Acetaminophen 650 mg PO q6h PRN pain",
        "Docusate sodium 100 mg PO BID",
        "Pantoprazole 40 mg PO daily",
        "Heparin 5000 units SC TID for DVT prophylaxis",
        "Atorvastatin 40 mg PO qHS",
        "Multivitamin one tablet PO daily",
    ),
    "disposition": (
        "Discharge instructions were reviewed with the patient and family.",
        "A follow up appointment was scheduled with the primary team.",
        "The patient was advised to return for any new or worsening symptoms.",
    ),
}

#: Expansions applied by the preprocessing step and reused for paraphrase generation.
ABBREVIATIONS: dict[str, str] = {
    "sob": "shortness of breath",
    "cp": "chest pain",
    "n/v": "nausea and vomiting",
    "htn": "hypertension",
    "dm2": "diabetes mellitus type 2",
    "cad": "coronary artery disease",
    "afib": "atrial fibrillation",
    "uti": "urinary tract infection",
    "pna": "pneumonia",
    "arf": "acute renal failure",
    "wbc": "white blood cell count",
    "hct": "hematocrit",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "po": "by mouth",
    "iv": "intravenous",
    "prn": "as needed",
    "bid": "twice daily",
    "tid": "three times daily",
    "qhs": "at bedtime",
    "sc": "subcutaneous",
    "ed": "emergency department",
    "icu": "intensive care unit",
    "los": "length of stay",
    "hd": "hospital day",
    "s/p": "status post",
    "y/o": "year old",
    "pt": "patient",
    "hx": "history",
    "tx": "treatment",
    "dx": "diagnosis",
}

SECTION_HEADERS: dict[str, str] = {
    "hpi": "History of Present Illness",
    "exam": "Physical Examination",
    "labs": "Pertinent Results",
    "course": "Brief Hospital Course",
    "meds": "Discharge Medications",
    "disposition": "Discharge Disposition and Condition",
}

SECTION_ORDER: tuple[str, ...] = ("hpi", "exam", "labs", "course", "meds", "disposition")
