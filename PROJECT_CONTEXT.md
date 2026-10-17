# MedXplain — Complete Project Context

**An explainable transformer framework for clinical risk prediction from unstructured clinical notes, with a clinician-facing web application.**

This file is the single, self-contained reference for the project: what it is, how the data were built, every method and design decision, every number the project has produced, the features of the application, and the limitations. It is written so that a reader with no other access to the code could understand the work and write a research paper from it. Every statistic below is copied from the project's own output files (`results/metrics/*.json`) or from measurements made on the corpus; where a number is approximate or a result has not been produced, this is stated.

> **Status as of 17 October 2026.** Prediction models trained and evaluated; application working end to end. Explanation-quality evaluations (faithfulness, stability, plausibility) and the statistical-significance tests are implemented but **have not been run**, so no numbers for them exist yet. All results are on a **synthetic** corpus, not real patients.

---

## Table of contents

1. [One-paragraph summary](#1-one-paragraph-summary)
2. [Problem and motivation](#2-problem-and-motivation)
3. [Contributions](#3-contributions)
4. [Data](#4-data)
5. [Preprocessing and long-document handling](#5-preprocessing-and-long-document-handling)
6. [Models](#6-models)
7. [Training protocol](#7-training-protocol)
8. [Decision threshold and calibration](#8-decision-threshold-and-calibration)
9. [Explainability methods](#9-explainability-methods)
10. [Explanation evaluation framework](#10-explanation-evaluation-framework)
11. [Results](#11-results)
12. [Key findings and discussion](#12-key-findings-and-discussion)
13. [The application (features)](#13-the-application-features)
14. [System architecture and API](#14-system-architecture-and-api)
15. [Engineering issues found and fixed](#15-engineering-issues-found-and-fixed)
16. [Testing](#16-testing)
17. [Limitations](#17-limitations)
18. [Ethics, privacy and safety](#18-ethics-privacy-and-safety)
19. [Reproducibility](#19-reproducibility)
20. [Future work](#20-future-work)
21. [Suggested paper outline](#21-suggested-paper-outline)
22. [Repository map](#22-repository-map)
23. [Glossary](#23-glossary)
24. [References](#24-references)

---

## 1. One-paragraph summary

MedXplain reads a free-text hospital discharge summary, predicts the risk of **in-hospital mortality**, and explains the prediction at the level of individual words. It compares four pretrained transformer encoders (ClinicalBERT, BioBERT, PubMedBERT, general BERT) against three tuned TF-IDF baselines (logistic regression, random forest, XGBoost) on a 3,000-note synthetic corpus whose generative process is known, which gives *gold-standard rationales* for evaluating explanations. Six explanation methods (raw attention, attention rollout, gradient-weighted attention, integrated gradients, occlusion, SHAP) share one output format so they can be compared on faithfulness, stability and plausibility. The work is delivered as a full-stack application (FastAPI + Next.js) with a prediction view, a clinician-facing results workspace (comparable previous cases, guideline-linked suggestions, a calibration-aware risk assessment, model evaluation details) and a peer-consultation page. Headline results: on 429 held-out test notes, a TF-IDF + logistic regression baseline reaches AUROC 0.862 / AUPRC 0.632; ClinicalBERT restricted to 128 tokens reaches only 0.690 / 0.382 (mean of 2 seeds) and is badly miscalibrated; the same ClinicalBERT retrained to read 512 tokens (essentially the whole note) reaches **0.807 / 0.503** and becomes well calibrated (ECE 0.044, Brier skill score 0.177).

---

## 2. Problem and motivation

- Clinical notes hold most of what a clinician observed (severity, trajectory, goals-of-care discussions) in unstructured text that coded fields miss.
- Transformer language models pretrained on clinical or biomedical text can extract this signal, but produce an unexplained number. Clinicians will not act on a risk score they cannot interrogate.
- The research question is therefore not *"can a transformer predict mortality from a note?"* but **"can the explanation of that prediction be shown to be trustworthy, and at what cost?"**
- Most clinical explainable-AI (XAI) work uses one explanation method on one model and validates it by visual inspection. MedXplain compares several methods on the same model and the same notes, along three independent axes (faithfulness, stability, plausibility), with controls.

NLP content at each layer:

| Layer | NLP role |
|---|---|
| Input | Free-text clinical narrative, median ~327 WordPiece tokens (~188 words), heavy jargon and abbreviation |
| Preprocessing | Tokenisation, de-identification placeholders, abbreviation handling, long-document truncation |
| Model | Domain-adapted masked language model fine-tuned for sequence classification |
| Task | Binary text classification under class imbalance (16.6% positive) |
| Explanation | Token- and word-level attribution anchored to the original text |
| Evaluation | ERASER-style rationale evaluation: faithfulness, stability, plausibility |

---

## 3. Contributions

1. **Head-to-head comparison of six explanation methods** on the same model and notes, scored on faithfulness, stability and plausibility (framework implemented; runs pending).
2. **A random-explanation control** attached to every faithfulness number, so "masking text hurts" is not mistaken for explanation quality.
3. **Explanation stability** under meaning-preserving edits (paraphrase, whitespace, irrelevant-sentence insertion): rarely reported in clinical XAI.
4. **Sign agreement** as a metric: does a method mark protective findings as protective? Quantifies what unsigned attention cannot express.
5. **A synthetic corpus with exact gold rationales** (character span and true log-odds weight of every injected risk factor), enabling automatic plausibility scoring.
6. **Empirical evidence on context length:** reading ~75 words (128 tokens) versus ~300 words (512 tokens) moves the same backbone from AUROC 0.68 to 0.81, and from badly miscalibrated to well calibrated.
7. **A documented train/serve truncation mismatch** (training on head + tail, serving head only), found and fixed. A concrete, reportable pitfall for deployed clinical NLP.
8. **A working clinician-facing system** with calibration-aware risk display, honest labelling of provenance (historical records vs model output vs generated suggestions), and explicit statements of what attention can and cannot claim.

---

## 4. Data

### 4.1 Why a synthetic corpus

MIMIC-III (the standard source of real ICU discharge summaries) requires PhysioNet credentialing and cannot be redistributed. A repository that runs only for credentialed users cannot be reviewed or reproduced, so `medxplain/data/synthetic.py` generates notes that imitate the **surface structure** of MIMIC-III discharge summaries with a **known generative process**. A MIMIC-III adapter (`medxplain/data/mimic.py`) emits the identical schema, so switching to real data is a configuration change. **No MIMIC-III run has been performed.**

### 4.2 Generative process

```
logit(mortality)   = intercept_m + 0.030·(age − 65) + Σ mortality_weight(f)   for each factor f present
logit(readmission) = intercept_r + 0.012·(age − 65) + Σ readmission_weight(f)
label ~ Bernoulli(sigmoid(logit)),  with extra label noise (label_noise = 0.05)
```

- **Labels are sampled, not thresholded**, so the best achievable AUROC is below 1.0 and models cannot saturate the benchmark.
- **Intercepts are calibrated by bisection** so the realised rates hit the targets (16% mortality, 20% readmission).
- **Patient acuity:** a per-admission log-normal acuity multiplier (σ = 0.55) raises the probability of every harmful factor and lowers the probability of protective ones, so a patient in septic shock is also more likely to be ventilated.
- Each factor's sentence is inserted verbatim (one of 2–3 phrasings), and **its exact character span and true weight are recorded as a gold rationale**.
- **Distractors:** 3–7 clinically realistic but outcome-neutral sentences (weight 0) are inserted per note from a pool of 26, so an explainer that highlights boilerplate is penalised.
- **Long notes:** 25% of notes also get 4–9 filler sentences (e.g. "Pharmacy performed medication reconciliation…") to stress long-document handling.
- Each note has 4–9 discharge medication lines, vitals and a basic lab line.

### 4.3 Note structure

Sections: Chief Complaint, History of Present Illness, Physical Examination (with a vitals line), Pertinent Results, Brief Hospital Course, Discharge Medications, Discharge Disposition and Condition. The raw notes also carry MIMIC-style headers (`Admission Date:`, `Service:`, `Attending:`) and `[** … **]` de-identification placeholders. Among the 3,000 processed notes, Brief Hospital Course appears in 2,788 and Discharge Disposition in 2,570; the other sections appear in all notes.

### 4.4 Risk factors (28)

Weights are hand-set to be clinically plausible in **direction and relative magnitude** (loosely informed by APACHE/SOFA-style severity and the readmission literature), **not calibrated to any real cohort**.

| Family | Factor | Section | Mortality weight | Readmission weight | Prevalence |
|---|---|---|---|---|---|
| Acute severity | septic_shock | course | +1.65 | +0.35 | 0.11 |
| Acute severity | vasopressors | course | +1.05 | +0.20 | 0.13 |
| Acute severity | mech_vent | course | +1.20 | +0.25 | 0.16 |
| Acute severity | cardiac_arrest | course | +1.95 | +0.15 | 0.05 |
| Acute severity | aki_dialysis | course | +0.95 | +0.55 | 0.10 |
| Acute severity | ards | course | +1.15 | +0.20 | 0.06 |
| Acute severity | gi_bleed | course | +0.70 | +0.45 | 0.07 |
| Acute severity | encephalopathy | course | +0.60 | +0.30 | 0.14 |
| Acute severity | lactate | labs | +0.85 | +0.10 | 0.12 |
| Acute severity | hypotension | exam | +0.55 | +0.20 | 0.15 |
| Chronic comorbidity | metastatic_cancer | hpi | +1.35 | +0.60 | 0.07 |
| Chronic comorbidity | chf | hpi | +0.45 | +0.85 | 0.18 |
| Chronic comorbidity | copd | hpi | +0.35 | +0.70 | 0.17 |
| Chronic comorbidity | cirrhosis | hpi | +0.95 | +0.65 | 0.06 |
| Chronic comorbidity | esrd | hpi | +0.60 | +0.80 | 0.08 |
| Chronic comorbidity | diabetes | hpi | +0.15 | +0.35 | 0.28 |
| Chronic comorbidity | dementia | hpi | +0.55 | +0.45 | 0.09 |
| Chronic comorbidity | immunosuppressed | hpi | +0.70 | +0.40 | 0.06 |
| Goals of care | comfort_measures | disposition | +2.10 | −0.90 | 0.06 |
| Goals of care | hospice | disposition | +1.55 | −0.70 | 0.05 |
| Protective | stable_course | course | −0.85 | −0.55 | 0.30 |
| Protective | ambulating | disposition | −0.70 | −0.60 | 0.26 |
| Protective | discharge_home | disposition | −0.60 | −0.35 | 0.34 |
| Protective | normal_labs | labs | −0.40 | −0.25 | 0.30 |
| Readmission drivers | poor_adherence | hpi | +0.10 | +0.95 | 0.13 |
| Readmission drivers | prior_admissions | hpi | +0.20 | +1.05 | 0.15 |
| Readmission drivers | no_support | disposition | +0.25 | +0.85 | 0.10 |
| Readmission drivers | substance_use | hpi | +0.30 | +0.75 | 0.11 |

Prevalence is the base probability before the acuity multiplier is applied.

### 4.5 Corpus statistics (realised)

| Statistic | Value |
|---|---|
| Notes / patients | 3,000 / 3,000 (one admission per patient) |
| Primary outcome | In-hospital mortality, 16.6% overall (497 deaths) |
| Secondary outcome | 30-day readmission, 19.6% overall |
| Age | mean 65.7, range 19–95 |
| Sex | 1,534 M / 1,466 F |
| Characters per note | mean 1,284, median 1,189, max 2,859 |
| WordPiece tokens per note (Bio_ClinicalBERT tokenizer, 500-note sample) | mean 341, SD 62, median 327, p90 440, p95 458, max 531 |
| Notes over 256 tokens / over 512 tokens | 95.8% / 0.6% |
| Words per note (400-note sample) | median 188, p95 290, max 344 |
| Tokens per word | 1.70 |
| Gold rationale spans | mean 4.52 per note; every note has at least one |

**Splits** (stratified, patient-disjoint; leakage check: 0 duplicate texts and 0 shared patients across splits):

| Split | Notes | Deaths | Mortality rate | Readmissions | Readmission rate |
|---|---|---|---|---|---|
| Train | 2,142 | 355 | 16.6% | 420 | 19.6% |
| Validation | 429 | 71 | 16.6% | 95 | 22.1% |
| Test | 429 | 71 | 16.6% | 73 | 17.0% |

For a fair comparison, **all models (baselines and transformers) are trained on the same 1,000-note stratified subsample** of the training split (`train.max_train_notes: 1000`). Validation and test are never subsampled.

### 4.6 MIMIC-III path (implemented, not run)

Admission-level cohort; discharge summaries with addenda concatenated; error rows dropped; age ≥ 18; mortality from `HOSPITAL_EXPIRE_FLAG`; readmission is an unplanned admission within 30 days (elective admissions and in-hospital deaths excluded). Two leakage guards:

- **Label leakage:** `mask_outcome_leakage()` redacts 12 phrases that state the outcome ("patient expired", "time of death", "autopsy", "Discharge Disposition: EXPIRED"), with a flag recorded so the with/without ablation can be reported.
- **Identity leakage:** patient-grouped splitting on `subject_id`; `check_leakage()` asserts patient disjointness and no duplicate note text.

`configs/mimic_full.yaml` is the intended full protocol: 512 tokens, full fine-tuning, 6 epochs, 5 seeds.

---

## 5. Preprocessing and long-document handling

- **De-identification placeholders become typed tokens** (`<DATE>`, `<PLACE>`, `<NAME>`) rather than being deleted, which would shift character offsets.
- **Administrative headers are stripped** (`Admission Date:`, `Service:`, `Attending:`). They are pure boilerplate, and removing them cut mean length from about 459 to 341 tokens.
- **Abbreviation expansion is off by default** (it changes token boundaries and therefore explanations) and is available as an ablation.
- **No lowercasing**; whitespace collapsed; rule lines stripped.
- **Head + tail truncation** (`truncation_strategy: head_tail`, `head_fraction: 0.35`): when a note exceeds the model's window, keep the first 35% and last 65% of the token budget. Plain right-truncation would throw away the hospital course and discharge disposition, where goals-of-care language (the strongest mortality signal) sits.
- **Sliding-window chunking** with pooled window probabilities (`max`, `mean` or `noisy_or`) is implemented for full-document scoring but is not used by the served models.

With a 128-token window the model reads 126 body tokens, about **75 words, roughly 38% of a typical note**. No note in the corpus fits entirely. With a 512-token window it reads 510 body tokens, about **300 words**, and **99.2% of notes are read in full**.

---

## 6. Models

### 6.1 Transformer backbones

| Key | Hugging Face checkpoint | Pretraining corpus | Parameters |
|---|---|---|---|
| `clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes (on top of BioBERT) | 108.3 M |
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` | PubMed abstracts + PMC full text | 108.3 M |
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMed, with a from-scratch biomedical vocabulary | 109.5 M |
| `bert` | `bert-base-uncased` | BooksCorpus + Wikipedia (general domain) | 109.5 M |

`bert` is the **control** that isolates the benefit of domain-specific pretraining. Architecture (`medxplain/models/transformer.py`): encoder → `[CLS]` pooling → dropout 0.1 → linear 2-class head. It's a thin custom wrapper rather than `AutoModelForSequenceClassification`, so that attention tensors (for rollout) and `inputs_embeds` (for integrated gradients) are exposed.

### 6.2 Baselines (`medxplain/models/baselines.py`)

TF-IDF features (up to 60,000), each classifier **tuned on the same validation split** the transformers use and **calibrated post hoc** (Platt / sigmoid) on validation:

| Baseline | Selected hyper-parameters |
|---|---|
| TF-IDF + Logistic Regression | C = 0.25, n-grams (1, 2), min_df = 3 |
| TF-IDF + Random Forest | 400 trees, max depth 24, min samples per leaf 1 |
| TF-IDF + XGBoost | 300 trees, learning rate 0.1, max depth 4 (seed 13) or 6 (seed 42), `scale_pos_weight` for imbalance |

Logistic regression also provides an **exact** explanation (TF-IDF value × coefficient, summing to the decision function): a faithful reference point.

---

## 7. Training protocol

Hand-written loop (`medxplain/train/trainer.py`), CPU only.

| Setting | 128-token pilot (4 backbones) | 512-token ClinicalBERT |
|---|---|---|
| Max sequence length | 128 | 512 |
| Truncation | head + tail (35% / 65%) | head + tail (rarely needed) |
| Frozen encoder layers | bottom 8 of 12 | bottom 8 of 12 |
| Trainable parameters | 28.9 M of ~108 M | 28.9 M of 108.3 M |
| Training notes | 1,000 | 1,000 |
| Epochs | 2 | 2 |
| Batch size | 16 | 4 × 4 gradient-accumulation steps (effective 16) |
| Learning rate (encoder / head) | 3e-5 / 3e-4 | 3e-5 / 3e-4 |
| Weight decay / warm-up | 0.01 / 10% | 0.01 / 10% |
| Loss | class-weighted cross-entropy, weights [0.60, 3.01] | same |
| Model selection | validation AUPRC (patience 1) | validation AUPRC |
| Seeds | 13, 42 (ClinicalBERT); 13 (others) | 13 |
| Training time (CPU) | ClinicalBERT ~31 min/seed; BioBERT 58; PubMedBERT 18; BERT 16 | 74 min |

Design decisions:

- **Select on validation AUPRC**, not loss or accuracy. At 16% prevalence, loss is dominated by the majority class.
- **Discriminative learning rates.** The new head gets 10× the encoder rate.
- **Class imbalance as an experimental variable.** Plain, class-weighted and focal loss are all available in the config.
- **Seed is the outer loop**, so an interrupted run still gives a complete single-seed comparison.
- **Compute constraint:** an Intel Core i5-1235U laptop CPU (2 performance + 8 efficiency cores, no GPU). Software: Python 3.11.9, PyTorch 2.8.0 (CPU), transformers 4.56.2, scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0.

---

## 8. Decision threshold and calibration

- **Threshold ("alert level").** The probability at or above which a note is flagged. It is chosen on **validation** only, never on test. The default policy maximises F1 on validation. For the served 512-token model it was re-selected as **the highest threshold that catches ≥ 80% of validation deaths**, giving **0.174**. The previous best-F1 threshold (0.504) is preserved in the results file (`previous_operating_point`). The tool `scripts/set_alert_level.py` performs this re-selection.
- **Calibration** is measured on test by expected calibration error (ECE, 10 bins), Brier score, and Brier skill score (BSS, improvement over always predicting the base rate; BSS < 0 means worse than that).
- **Display rule in the application:** a score is shown as a percentage chance **only if ECE ≤ 0.05 and BSS > 0** for the served checkpoint's seed. Otherwise the app shows it only as above or below the alert level, with an explanation. All 128-token transformers fail this rule; the 512-token model passes.
- **Risk bands** shown in the UI are relative to the alert level *t*: high if p ≥ max(0.75, 1.5t); elevated if p ≥ t; borderline if p ≥ 0.5t; otherwise low.

---

## 9. Explainability methods

All methods return the same `Explanation` payload (`medxplain/explain/payload.py`) at two granularities: **token level** (model subwords, used by the faithfulness metrics) and **word level** (subwords merged back to words, with character offsets into the original note, used by the UI). Subword scores are summed into words by default.

| Method | Signed? | Cost (model evaluations) | What it computes |
|---|---|---|---|
| `attention_raw` | No | 1 forward pass | Mean over heads of the last layer's `[CLS]` attention row |
| `attention_rollout` (default) | No | 1 forward pass | Attention propagated through all layers with residuals: Â = norm(0.5·mean_heads(A) + 0.5·I), R = Â_L ⋯ Â_1, importance = R[CLS, :] (Abnar & Zuidema 2020) |
| `attention_grad` | Yes | 1 forward + 1 backward | Attention weighted by the gradient of the predicted class (Chefer et al. 2021) |
| `integrated_gradients` | Yes | `n_steps` (16) forward + backward | Path integral from an all-`[MASK]` baseline (special tokens kept), Riemann midpoint; reports completeness error |
| `occlusion` | Yes | 1 forward pass per word (cap 350 words in the API) | Drop in log-odds when each word is replaced by `[MASK]` |
| `shap` | Yes | `max_evals` (API 200; evaluation 150) | Partition SHAP with a text masker, on the **log-odds** scale so values stay additive |

Key points for a paper:

- **Attention is unsigned and class-agnostic.** A high rollout score means "the model used this word", not "this word raised the risk". The API marks such payloads `signed: false`, and the UI renders them in a single colour, with no "protective" panel.
- **The IG baseline is all-`[MASK]`**, not the zero embedding: a zero vector is not a realistic text input. Completeness (attributions sum to F(x) − F(x′)) gives a built-in check, reported as `convergence_delta`.
- **SHAP on log-odds**, because Shapley additivity breaks on the probability scale once the sigmoid saturates.
- **Explanations describe exactly the input the model scored.** With head + tail truncation, the skipped middle of a long note is reported as `unread_span` and shown greyed out, never silently blank. Occlusion only masks words the model actually reads.
- **Directional language in the UI** says a word "raised" or "lowered the model's score", not the patient's risk, and states that highlights are not evidence of cause.

---

## 10. Explanation evaluation framework

Implemented in `medxplain/eval/` and `scripts/04–06`. **None of these have been run yet.** The configuration specifies 25 test notes for faithfulness, 15 for stability, 25 for plausibility and 12 for human review.

### 10.1 Faithfulness (ERASER; DeYoung et al. 2020)

- **Comprehensiveness** = p(ŷ | x) − p(ŷ | x without the top-k tokens). Higher is better.
- **Sufficiency** = p(ŷ | x) − p(ŷ | only the top-k tokens). Lower is better.
- Averaged over k ∈ {1, 5, 10, 20, 50}% of tokens. Deletion and insertion curves and their AUCs.
- **Removal = masking, not deletion**, to avoid a sequence-length artefact.
- **A random-explanation control** gives `comprehensiveness_over_random`.
- Tokens are ranked by **signed score toward the predicted class**.

### 10.2 Stability

The note is re-explained after (a) whitespace/casing-only edits, (b) abbreviation and synonym paraphrase, and (c) insertion of an irrelevant sentence. Reported: Spearman rank correlation, top-10 Jaccard overlap, prediction shift, and **distractor mass** (share of importance landing on the inserted sentence).

### 10.3 Plausibility

- **Automatic (synthetic):** token-F1 and rationale AUPRC against gold spans, Spearman correlation with the true weights, **sign agreement**, and off-rationale mass.
- **Human (any corpus):** a blinded review packet (HTML form + CSV), with methods relabelled A/B/C and reshuffled per note, a 1–5 Likert rubric, and Krippendorff's α (ordinal) for inter-rater agreement.

### 10.4 Statistical testing (`medxplain/eval/significance.py`, run by `scripts/03_evaluate.py`; not yet run)

Headline model selected on validation and tested once. Paired tests (DeLong for AUROC; stratified paired bootstrap, 1,000 resamples, for AUPRC), Holm–Bonferroni across baseline comparisons, and seed-level domain-adaptation tests against BERT with Cohen's d.

---

## 11. Results

All results are on the **synthetic** test split (n = 429, 71 deaths, prevalence 16.6%). The no-skill AUPRC equals prevalence (0.166). Thresholds are chosen on validation. "±" is the standard deviation across seeds; a single value means one seed.

### 11.1 Main comparison

| Model | Window | Seeds | AUROC | AUPRC | AUPRC lift | F1 | Brier | Brier skill | ECE |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | full note | 2* | **0.862** | **0.632** | 3.82× | 0.519 | **0.094** | 0.320 | **0.022** |
| TF-IDF + XGBoost | full note | 2 | 0.849 ± 0.013 | 0.582 ± 0.017 | 3.45–3.59× | 0.517 ± 0.031 | 0.104 | 0.243–0.255 | 0.026–0.043 |
| TF-IDF + Random Forest | full note | 2 | 0.843 ± 0.001 | 0.554 ± 0.003 | 3.34–3.36× | 0.510 ± 0.002 | 0.104 | 0.250 | 0.044–0.045 |
| **ClinicalBERT, 512 tokens** | ~300 words | 1 | **0.807** | **0.503** | 3.04× | 0.443† | **0.114** | **0.177** | **0.044** |
| PubMedBERT, 128 tokens | ~75 words | 1 | 0.728 | 0.420 | 2.54× | 0.419 | 0.190 | −0.376 | 0.258 |
| ClinicalBERT, 128 tokens | ~75 words | 2 | 0.690 ± 0.009 | 0.382 ± 0.005 | 2.29–2.33× | 0.356 ± 0.002 | 0.178 | −0.30 / −0.28 | 0.217 / 0.207 |
| BioBERT, 128 tokens | ~75 words | 1 | 0.685 | 0.355 | 2.15× | 0.366 | 0.190 | −0.376 | 0.244 |
| BERT-base (general), 128 tokens | ~75 words | 1 | 0.663 | 0.336 | 2.03× | 0.380 | 0.207 | −0.500 | 0.277 |

\* Logistic regression is deterministic, so both seeds give identical results (SD 0).
† F1 at the recall-80% operating point (threshold 0.174). At the original best-F1 threshold (0.504) it was 0.454.

### 11.2 Validation scores (used for model selection)

| Model | Validation AUROC | Validation AUPRC |
|---|---|---|
| ClinicalBERT 512 (seed 13) | 0.809 | 0.592 |
| ClinicalBERT 128 (seeds 13 / 42) | 0.724 / 0.735 | 0.374 / 0.362 |
| PubMedBERT 128 | 0.729 | 0.375 |
| BioBERT 128 | 0.724 | 0.340 |
| BERT 128 | 0.671 | 0.326 |

Validation trajectory of ClinicalBERT 512: after epoch 1, AUROC 0.796 / AUPRC 0.546; after epoch 2, 0.809 / 0.592.

### 11.3 Effect of context length (same backbone, same recipe, seed 13)

| ClinicalBERT seed 13 | 128 tokens | 512 tokens | Change |
|---|---|---|---|
| Test AUROC | 0.684 | 0.807 | +0.123 |
| Test AUPRC | 0.378 | 0.503 | +0.125 |
| Brier | 0.180 | 0.114 | −0.066 |
| Brier skill score | −0.304 | +0.177 | from worse than base rate to better |
| ECE | 0.217 | 0.044 | calibration check failed → passed |

### 11.4 Operating points of the served model (ClinicalBERT 512)

| Threshold policy | Threshold | Test recall | Test precision | Specificity | Flagged | Confusion (TP / FN / FP / TN) |
|---|---|---|---|---|---|---|
| Best F1 on validation (original) | 0.504 | 0.38 | 0.56 | 0.94 | ~11% | 27 / 44 / 21 / 337 |
| **Recall ≥ 80% on validation (served)** | **0.174** | **0.82** | **0.30** | 0.63 | 45% | 58 / 13 / 133 / 225 |

Descriptive test-set operating points (thresholds read off the test set, for illustration only): recall 0.51 → precision 0.45, flagged 19%; recall 0.70 → precision 0.38, flagged 31%; recall 0.80 → precision 0.33, flagged 41%; recall 0.90 → precision 0.27, flagged 54%.

For comparison, the 128-token ClinicalBERT (seed 13, threshold 0.395) has recall 0.72, precision 0.24, specificity 0.55 (TP 51 / FN 20 / FP 163 / TN 195).

### 11.5 Illustrative single-note behaviour (anecdotal)

On the case-study note "60y – comfort measures" (true outcome: died, 413 tokens):

| Configuration | What the model read | Score | Flagged? |
|---|---|---|---|
| 128-token model, head-only serving (original bug) | first 126 tokens (complaint, history, exam) | 0.538 | yes (alert 0.395) |
| 128-token model, head + tail serving (fixed) | start + end, incl. "transitioned to comfort-focused care" | 0.818 | yes |
| 512-token model, alert 0.504 | whole note | 0.37 | no |
| 512-token model, alert 0.174 (served) | whole note | 0.37 | yes |

The 512-token model scores all four surviving case-study patients at 0.03–0.09 (not flagged) and the two who died at 0.37 and 0.70 (both flagged at 0.174). These are single examples, not evidence of performance.

### 11.6 Not yet available

Faithfulness, stability, plausibility, human review and significance tests: **no results files exist for the real experiment**. The application shows these as "Not evaluated yet". Figures and LaTeX tables (`scripts/07_figures_tables.py`) have not been generated for it (`results/figures/` and `results/tables/` are empty).

**Do not use `results/smoke/`.** That folder does contain faithfulness, stability, plausibility and significance results, plus figures and tables. They come from `configs/smoke.yaml`, a tiny automated test run on a handful of notes that only checks the pipeline runs end to end. They are not experimental results and must not be reported.

---

## 12. Key findings and discussion

1. **A tuned bag-of-words baseline beats every transformer in this setting.** TF-IDF + LR (AUROC 0.862, AUPRC 0.632) outperforms even the 512-token ClinicalBERT (0.807 / 0.503). Plausible reasons: (a) only 1,000 training notes; (b) only the top 4 of 12 encoder layers trained, for 2 epochs; (c) the synthetic labels are a near-linear function of the presence of fixed phrases, which favours a linear model over n-grams. This matches the small-data clinical NLP literature: domain-adapted transformers do not dominate a tuned linear baseline without enough data, context and fine-tuning.
2. **Context length matters more than backbone choice.** Moving ClinicalBERT from 128 to 512 tokens (+0.12 AUROC, +0.13 AUPRC) dwarfs the spread between backbones at 128 tokens (0.663–0.728 AUROC). At 128 tokens the model never sees most of the hospital course, where the strongest risk factors are injected.
3. **Short-context transformers are badly miscalibrated.** All 128-token transformers have negative Brier skill (worse than predicting the base rate) and ECE 0.21–0.28. The 512-token model is well calibrated (ECE 0.044, BSS 0.177) without any post-hoc calibration. The baselines were Platt-calibrated and are also well calibrated.
4. **Domain pretraining at 128 tokens:** PubMedBERT (0.728) > ClinicalBERT (0.690) ≈ BioBERT (0.685) > BERT-base (0.663). This is consistent with a domain-adaptation benefit, but rests on one or two seeds and no significance test, so it should not be claimed without the planned paired tests.
5. **The alert level is a clinical choice, not a technical detail.** On the same model, the default best-F1 threshold catches 38% of deaths with 56% precision; the recall-80% threshold catches 82% with 30% precision and flags 45% of patients. The application exposes this and records the policy used.
6. **Train/serve mismatches are silent.** The served models originally scored notes with head-only truncation while having been trained and evaluated on head + tail. Published metrics described a different input from the one users saw (see §15).

---

## 13. The application (features)

A working full-stack system (Next.js 14 + Tailwind + Recharts frontend on port 3000; FastAPI backend on port 8000). Tabs: **Analyse | Results | Peer Consult | About**.

### 13.1 Analyse

- Paste or upload a discharge note, or load one of 6 case-study notes from the test split.
- Choose a model (default: ClinicalBERT 512) and an explanation method, or compare several methods at once.
- Output:
  - a **risk card**: a semicircular gauge with the alert level marked, a plain-language headline, an "in plain words" explanation (as "about N in 100" only when calibrated), and a 0–100% bar with patient and alert-level markers;
  - a **highlighted note**: red raised and blue lowered the model's score for signed methods; purple for attention, which is unsigned; unread parts greyed out;
  - a **top contributing words** chart;
  - a **"How to read this"** panel in plain language, with technical details collapsible;
  - truncation and other warnings, and an in-memory session log.

### 13.2 Results (clinical workspace for the analysed note)

- **Clinical Insights from Previous Cases:** retrieves comparable historical cases from the 3,000-note library.
  - **Matching:** cases must share at least 2 of the acute, chronic or goals-of-care conditions recognised in the note (keyword detection over 28 concepts); age within ±10 years when at least 5 such cases exist; TF-IDF (1–2-gram) cosine ≥ 0.10.
  - **Ranking:** 0.6 × condition Jaccard + 0.4 × cosine; the top 30 are kept, and near-duplicates (cosine ≥ 0.97, i.e. the note itself) are excluded.
  - **Shown:** the number of matches, the criteria, recorded outcomes (deaths, and readmissions among survivors) against library-wide rates, recommendation categories quoted from matched notes with their frequencies, and the five closest cases with their full source text.
  - **Labelling:** the section is labelled as synthetic demonstration data, and frequencies are described as not implying effectiveness.
- **AI suggestions** (orange box, separate from historical data). Rule-generated, never patient-specific treatment advice:
  - up to 5 **external guideline pointers**: recognised conditions mapped to 18 published guidelines (NICE NG51, NG106, NG115, NG148, NG107, NG28, NG17, NG97, CG103, NG50, CG141, NG31, CG76, CG115, NG27; the Surviving Sepsis Campaign; GOLD; Resuscitation Council UK), each link verified to resolve;
  - up to 3 **documentation-gap prompts** (missing follow-up, pending results, medication changes).
- **Patient-specific risk assessment:** flagged / not-flagged verdict, calibration status, contributing-words chart and "how to read this". The full prediction and highlighted note live on Analyse.
- **Need another clinical perspective?** The "Ask a Peer Physician" button opens Peer Consult with the note prefilled.
- **Technical Details: Model Evaluation and Validation** (collapsible): a plain-language summary, dataset tiles, every model's scores with a "score out of 10" view, ROC and PR curves, a metric glossary, and "Not evaluated yet" placeholders for the explanation-quality sections.
- **Removed by design:** an earlier version also showed rule-based Case Summary, Discharge Documentation Review and Follow-up sections. They were removed from the UI at the user's request (the backend still computes them to feed the documentation prompts).

### 13.3 Peer Consult (frontend-only)

- A note box (prefilled from the latest analysed note), a clinical question, and **two questions per browser session** (counter "Questions used: X/2"; the limit survives a page reload).
- Each question returns five fixed responses from named physician profiles: Internal Medicine (12 years), Emergency Medicine (10), Respiratory Medicine (9), Cardiology (14) and General Medicine (8).
- **Response set:** A (diagnosis and interpretation) or B (treatment and discharge), chosen by keyword counts in the question. A tie or no match gives Set A.
- **Responses are static prompts**, identical for every case of a question type; a footnote states they are automatically generated and not a review of the case by a physician.
- Nothing is sent to any server.
- A separate, authenticated Peer Consult backend (SQLite, encrypted case content, verification gate, consultation workflow; see §14) exists in the code but is not shown in the current UI.

### 13.4 About

Explains the purpose, how to read explanations honestly (attention ≠ attribution; the threshold isn't 50%; truncation is visible), and the limitations.

---

## 14. System architecture and API

```
Next.js 14 + Tailwind + Recharts (:3000)  ──REST/JSON──►  FastAPI (:8000)  ──►  PyTorch + Hugging Face
                                                             │
                                                             ├─ inference service (LRU cache of 128, guard rails 20–40,000 chars)
                                                             ├─ clinical analysis (sections, concepts, case library, suggestions)
                                                             └─ peer consult store (SQLite, Fernet-encrypted fields)
```

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict` | Score a note; returns probability, threshold, risk band, calibration status, explanations, warnings |
| POST | `/api/predict/batch` | Score several notes without explanations |
| GET | `/api/health`, `/api/models`, `/api/methods`, `/api/examples` | Service status and metadata |
| GET / DELETE | `/api/history` | In-memory session log |
| GET | `/api/results` | Published metrics files for the Technical Details view |
| GET / DELETE | `/api/cache` | Prediction cache statistics / clear |
| POST | `/api/clinical/analyze` | Comparable cases, suggestions, extraction for one note |
| — | `/api/peer/*` (16 routes) | Authenticated Peer Consult backend: register, login, logout, profile, verification, verified directory, dashboard, consultations (create, list, detail, accept, decline, cancel, close, messages, history) |

Peer Consult backend security, for completeness:
- **Passwords:** PBKDF2-SHA256 hashes.
- **Sessions:** bearer tokens stored only as SHA-256 hashes, with a 12-hour expiry.
- **Encryption at rest:** case text and messages encrypted with Fernet.
- **Audit log:** holds no case content.
- **Verification gate:** only `verified` accounts (from a licensing authority) can use it. **No licensing provider is integrated**, so real accounts cannot be verified. An opt-in development override exists (`MEDXPLAIN_PEER_ALLOW_DEV_VERIFICATION=1` plus `scripts/peer_consult_admin.py`).
- **Retention:** finished consultations are purged after 365 days.

---

## 15. Engineering issues found and fixed

1. **Train/serve truncation mismatch.** Training and evaluation used head + tail truncation (`ClinicalNoteDataset._encode`), but the API and all explainers used head-only truncation. The served model never read the hospital course or disposition it was trained on, and the published metrics did not describe the served behaviour. **Fix:** one shared encoder (`encode_for_model`, `encode_with_offsets` in `payload.py`) reproduces the training truncation exactly, with a test asserting token-for-token equality; unread text is reported and greyed out.
2. **Short context.** Even with the correct truncation, 128 tokens covered about 38% of a typical note. **Fix:** retrained ClinicalBERT at 512 tokens (§11.3) and installed it alongside the 128-token model (`scripts/promote_checkpoint.py`), so all earlier results are preserved.
3. **Calibration was over-claimed.** The API described the probability as "calibrated" when ECE was 0.22 and BSS was negative. **Fix:** the calibration status is read from the evaluation files per checkpoint seed, and percentages are shown only when the criteria pass.
4. **Occlusion word cap.** The explanation tested at most 200 words (34% of notes are longer). **Fix:** raised to 350 (the longest note is 344 words), and words the model does not read are skipped.
5. **Unreadable validation errors** in the UI ("[object Object]"). **Fix:** errors are formatted per field.

---

## 16. Testing

| Suite | Count | Result (latest run) |
|---|---|---|
| Backend (`python -m pytest backend/tests`) | 191 tests | 190 pass; 1 intermittent failure (`test_an_oracle_explanation_beats_a_reversed_one`, about 1 in 3 runs) |
| Frontend logic (`npm test`, Node's test runner) | 9 tests | 9 pass |
| Frontend type-check (`tsc --noEmit`) | — | clean |

The intermittent backend test uses a tiny transformer with **unseeded random weights**, so its outcome varies run to run; it also failed before any recent explanation changes. Seeding the fixture in `backend/tests/conftest.py` would make it deterministic.

The backend tests cover:
- gold-span alignment and character offsets surviving tokenisation and truncation;
- **served encoding = training encoding**;
- patient-disjoint splitting and subword→word aggregation;
- IG completeness, and occlusion matching real confidence drops;
- metric formulas against closed forms;
- the full API surface, including errors;
- the clinical analysis (extraction, retrieval counts, suggestions, guideline mapping);
- Peer Consult: authentication, verification gate, workflow transitions, permissions, encryption at rest, persistence, and the audit log holding no case content.

Manual browser checks covered every tab, the Results → Peer Consult prefill, both response sets, the two-question limit, and phone-width (400 px) layout.

---

## 17. Limitations

- **Synthetic data.** All numbers describe the pipeline on generated notes whose labels depend on fixed phrases. They are **not clinical performance**, and the risk weights are plausible in direction, not calibrated effect sizes.
- **Compute-limited training.** CPU only; 1,000 training notes; 8 of 12 layers frozen; 2 epochs; one or two seeds per model. The intended protocol (`mimic_full.yaml`: full fine-tuning, 512 tokens, 5 seeds, real data) has not been run.
- **Single seed** for the 512-token model and three of the 128-token models. No confidence intervals or significance tests yet.
- **Explanation-quality evaluations not yet run.** The core research contribution (faithfulness, stability, plausibility comparison) has no results.
- **Attention is class-agnostic** by construction, which limits what an attention-only system can claim.
- **SHAP is slow on CPU** (tens of seconds per note).
- **Retrieval and suggestions are keyword-based.** Unusual phrasing is missed; guideline links mean "relevant topic", not "checked against the guideline".
- **Peer Consult responses are static** and not written for the case. The authenticated workflow lacks a real licence-verification integration.
- **Not a medical device.** It isn't validated for clinical use.

---

## 18. Ethics, privacy and safety

- **Note text is never written to disk by the prediction API.** The session log is in memory only. Peer Consult (frontend) keeps notes in the browser tab.
- **Calibration-gated display:** percentages are shown only when evaluation supports them.
- **Provenance labelling** throughout: historical records, model prediction, interpretation, and AI suggestion are visually distinct.
- **Directional honesty:** attention is never shown as "raises" or "lowers risk"; signed attributions describe the *model's score*, not causation.
- **Threshold transparency:** the alert level and the policy that set it are recorded and shown.
- **No regulatory compliance is claimed.** No real patient data are included or required.

---

## 19. Reproducibility

```bash
pip install -r backend/requirements.txt && (cd frontend && npm install)
python scripts/00_build_dataset.py          # corpus, splits, data report
python scripts/01_train_baselines.py        # TF-IDF baselines
python scripts/02_train_transformers.py     # 4 backbones at 128 tokens (configs/default.yaml)
# 512-token ClinicalBERT into a separate folder, then install it:
python scripts/02_train_transformers.py --set "model.backbones=[clinicalbert]" --set model.max_length=512 \
  --set "experiment.seeds=[13]" --set experiment.results_dir=results/clinicalbert_512 \
  --set train.batch_size=4 --set train.gradient_accumulation_steps=4 --set train.eval_batch_size=8
python scripts/promote_checkpoint.py results/clinicalbert_512 clinicalbert clinicalbert_512 --display-name "Bio_ClinicalBERT (512 tokens)"
python scripts/set_alert_level.py clinicalbert_512 --target-recall 0.8
python scripts/03_evaluate.py ; python scripts/04_faithfulness.py ; python scripts/05_stability.py
python scripts/06_plausibility.py ; python scripts/07_figures_tables.py
# app
python -m uvicorn backend.app.main:app --port 8000
cd frontend && npm run dev                  # http://localhost:3000
```

Guarantees:
- The resolved config is saved with the results.
- Every metrics file embeds an `environment` block.
- Seeds are fixed and passed through Python, NumPy and Torch.
- Figures and tables are regenerated from the saved metrics only.
- Test probabilities and validation probabilities for every run are saved in `results/predictions/*.npz`, so thresholds and operating points can be recomputed without retraining.

---

## 20. Future work

1. Run the explanation evaluations (scripts 04–06) on the 512-token model, and the significance tests (script 03).
2. Train every backbone at 512 tokens with ≥ 3 seeds, then run full fine-tuning on a GPU.
3. Run on MIMIC-III with leakage masking, and report the with/without-masking ablation.
4. Carry out the blinded clinician plausibility study with the provided packet.
5. Use sliding-window inference for notes over 512 tokens, with calibrated pooling.
6. Study threshold policy as part of clinical deployment (alert burden vs missed deaths).
7. Integrate a real licensing-authority verification service for the authenticated Peer Consult.

---

## 21. Suggested paper outline

1. **Title:** e.g. *"Faithful, Stable and Plausible? Evaluating Explanations of Transformer Mortality Prediction from Discharge Summaries"*.
2. **Abstract:** adapt §1.
3. **Introduction:** §2; contributions from §3.
4. **Related work:** clinical BERT models (Alsentzer, Lee, Gu); attention as explanation (Jain & Wallace; Wiegreffe & Pinter); attribution (Sundararajan; Lundberg & Lee; Abnar & Zuidema; Chefer); ERASER (DeYoung); MIMIC-III mortality prediction.
5. **Data:** §4 (generative process, the risk-factor table, corpus statistics, splits, leakage checks).
6. **Methods:** §5 preprocessing and truncation, §6 models, §7 training, §8 threshold and calibration, §9 explanation methods, §10 evaluation.
7. **Experiments and results:** Tables 11.1, 11.3, 11.4; ROC/PR curves from `results/metrics`; explanation results once run.
8. **Discussion:** §12 (baseline vs transformer, context length, calibration, threshold trade-off, train/serve mismatch).
9. **System:** §13–14 as a system-demonstration contribution.
10. **Limitations and ethics:** §17–18.
11. **Conclusion and future work:** §20.

An IEEE LaTeX template for the paper is already in `paper/medxplain_ieee.tex`.

---

## 22. Repository map

```
medxplain/                    research package
  data/                       synthetic.py, clinical_lexicon.py, mimic.py, preprocess.py, splits.py, dataset.py
  models/                     transformer.py (wrapper, long-note inference), baselines.py, losses.py
  train/trainer.py            training loop
  explain/                    attention.py, gradients.py, shap_explainer.py (SHAP + occlusion), payload.py (shared encoder + payload)
  eval/                       metrics.py, faithfulness.py, stability.py, plausibility.py, significance.py
  clinical/                   text.py (sections/sentences), concepts.py (28 concepts), extraction.py,
                              similar_cases.py (case library), guidelines.py (external guidance), suggestions.py
backend/app/                  FastAPI: main.py, config.py, schemas.py, schemas_peer.py,
                              routers/ (predict, meta, clinical, peer), services/ (inference, registry, calibration, peer_*)
backend/tests/                pytest suite (191 tests)
frontend/                     Next.js app: app/page.tsx, components/ (ResultPanel, ClinicalWorkspace, ResultsView,
                              StaticPeerConsult, PlainLanguage, …), lib/ (api, types, staticPeer), tests/
scripts/                      00–07 pipeline, promote_checkpoint.py, set_alert_level.py, peer_consult_admin.py
configs/                      default.yaml (CPU pilot), mimic_full.yaml (full protocol), smoke.yaml (CI)
results/metrics/              dataset_report.json, baselines.json, transformers.json (incl. clinicalbert_512)
results/checkpoints/          served models (git-ignored)
results/clinicalbert_512/     512-token training run (log, metrics, predictions)
paper/                        IEEE LaTeX template
README.md, explanation.md     earlier documentation (some numbers there predate the 512-token run)
```

---

## 23. Glossary

- **AUROC:** probability that a randomly chosen death is scored above a randomly chosen survivor (0.5 = chance, 1 = perfect).
- **AUPRC:** area under the precision–recall curve; chance level equals prevalence (0.166 here). More informative than AUROC for rare outcomes.
- **AUPRC lift:** AUPRC divided by prevalence.
- **F1:** harmonic mean of precision and recall at the chosen threshold.
- **Brier score:** mean squared error of the predicted probabilities (lower is better).
- **Brier skill score (BSS):** 1 − Brier / Brier of always predicting prevalence. BSS < 0 means worse than the base rate.
- **ECE:** expected calibration error, the average gap between predicted and observed rates across 10 probability bins.
- **Recall / sensitivity:** share of deaths flagged. **Precision / PPV:** share of flags that were deaths. **Specificity:** share of survivors not flagged.
- **Alert level / threshold:** the probability at which a note is flagged.
- **Head + tail truncation:** keep the start and end of a long note, drop the middle.
- **Signed attribution:** a score whose sign says whether a word pushed the model's output up or down.
- **Comprehensiveness / sufficiency:** ERASER faithfulness metrics (§10.1).
- **Gold rationale:** the text span known, by construction, to carry a risk factor.

---

## 24. References

- Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL*.
- Alsentzer, E. et al. (2019). Publicly Available Clinical BERT Embeddings. *Clinical NLP Workshop*.
- Chefer, H., Gur, S. & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*.
- DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). Comparing the Areas under Two or More Correlated ROC Curves. *Biometrics*.
- DeYoung, J. et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL*.
- Gu, Y. et al. (2021). Domain-Specific Language Model Pretraining for Biomedical NLP. *ACM HEALTH*.
- Jain, S. & Wallace, B. C. (2019). Attention is not Explanation. *NAACL*.
- Johnson, A. E. W. et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*.
- Lee, J. et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics*.
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
- Wiegreffe, S. & Pinter, Y. (2019). Attention is not not Explanation. *EMNLP*.
- Guidelines linked by the application: NICE (NG51, NG106, NG115, NG148, NG107, NG28, NG17, NG97, CG103, NG50, CG141, NG31, CG76, CG115, NG27); Surviving Sepsis Campaign (SCCM); GOLD COPD report; Resuscitation Council UK.
