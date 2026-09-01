# MedXplain — Technical Explanation

**An Explainable Transformer Framework for Clinical Risk Prediction from Unstructured Clinical Notes**

This document explains what has been built, why each design decision was made, and
what the current evidence does and does not support. It is written for a technical
reader who wants to interrogate the methodology rather than be sold on it.

---

## 1. The problem, and why it is an NLP problem

Clinical notes — discharge summaries, progress notes — contain the richest
information in the medical record, but it is locked in unstructured free text.
Structured fields (labs, vitals, ICD codes) capture a fraction of what the
clinician actually observed. A model that reads the narrative has access to
severity, trajectory and goals-of-care language that no coded field contains.

Transformer language models pretrained on clinical text extract that signal well.
They also produce a bare number with no justification, and a clinician will not
act on an unexplained risk score. **The research question is therefore not "can a
transformer predict mortality from a note" — that is settled — but "can the
explanation of that prediction be shown to be trustworthy?"**

The NLP content sits at every layer:

| Layer | NLP role |
|---|---|
| Input | Free-text clinical narrative, ~340 wordpieces median, heavy domain jargon and abbreviation |
| Preprocessing | Tokenization, de-identification artifacts, abbreviation handling, long-document truncation/chunking |
| Model | Domain-adapted masked LM (ClinicalBERT / BioBERT / PubMedBERT) fine-tuned for sequence classification |
| Task | Binary text classification under severe class imbalance |
| Explanation | **Token-level attribution** — the explanation is anchored to language, not to tabular features |
| Evaluation | Rationale evaluation borrowed from ERASER; explanation faithfulness, stability and plausibility |

The prediction target is **in-hospital mortality** (primary); **30-day
readmission** is carried as a secondary label throughout and can be made primary
with one config flag.

---

## 2. Data

### 2.1 Why there is a synthetic corpus at all

MIMIC-III requires PhysioNet credentialing and CITI training (weeks of lead time)
and **cannot be redistributed**. A repository that only runs for credentialed
users cannot be reviewed, reproduced, or handed to a collaborator.

So `medxplain/data/synthetic.py` generates notes that imitate the *surface
structure* of MIMIC-III discharge summaries — section headers, `[**Hospital1 18**]`
de-identification placeholders, medication lists, clinical abbreviations,
boilerplate — with a **known generative process**:

```
logit  = intercept + age_term + Σ  weight(risk factor present)
label  ~ Bernoulli(sigmoid(logit))
```

Two properties are deliberate:

1. **The label is sampled, not thresholded.** The Bayes-optimal AUROC is
   therefore strictly below 1.0 and models cannot saturate the benchmark. A
   deterministic label would make every method look perfect and the comparison
   meaningless.
2. **The intercept is calibrated by bisection** so the realised positive rate
   lands on a configured target (16% mortality, 20% readmission), reproducing the
   rare-event regime that makes this problem hard.

`medxplain/data/clinical_lexicon.py` defines 24 risk factors across five
families — acute severity (septic shock, mechanical ventilation, cardiac arrest,
ARDS), chronic comorbidity (metastatic cancer, cirrhosis, ESRD, CHF), goals of
care (comfort measures, hospice), protective findings (uncomplicated course,
ambulating, discharged home), and readmission-specific social drivers
(non-adherence, prior admissions, no social support). Each carries a
mortality weight, a readmission weight, a prevalence, and 2–3 surface
realisations. Weights are hand-set to be clinically plausible in *direction and
relative magnitude*, not calibrated to a real cohort.

Patient-level correlation is modelled with a per-admission log-normal **acuity
multiplier**: a patient in septic shock is more likely to also be ventilated, and
protective findings become less likely as acuity rises. This is enough structure
to make the task non-trivially separable without an unwieldy full covariance model.

### 2.2 The property that makes this corpus scientifically useful

**Every injected risk-factor sentence is recorded with its exact character span
and its true log-odds weight.**

That gives us *gold rationales* for free — something that normally requires
expensive human annotation (this is precisely what the ERASER benchmark had to
commission). Consequences:

- Explanation **plausibility** can be scored automatically (token-F1, rationale
  AUPRC against the gold spans).
- We can ask a question no real dataset permits: does the highest-scoring word
  correspond to the *largest true weight*? (`weight_spearman`)
- We can test **sign agreement**: does the method mark protective findings as
  protective, or merely as "important"? Attention cannot express direction at
  all, and this measures the cost of that.

Additionally, **distractor sentences with zero weight** are injected into every
section ("Case management followed throughout the admission", "Parking validation
was provided"). An explainer that latches onto fluent clinical boilerplate is
measurably penalised via `off_rationale_mass` and `distractor_mass`.

Realised corpus: 3,000 notes, 16.6% mortality, median 327 wordpieces, mean 4.5
gold rationale spans per note.

**Honest framing:** this corpus characterises *the pipeline and the relative
behaviour of explanation methods*. It is not clinical evidence. Headline clinical
numbers must come from MIMIC-III.

### 2.3 The MIMIC-III path

`medxplain/data/mimic.py` builds the real cohort from a local PhysioNet download
and emits the identical schema, so every downstream stage is dataset-agnostic.
The cohort definition is documented in the module docstring for verbatim
transcription into a paper: admission-level unit of analysis, discharge summaries
with addenda concatenated in chart order, `ISERROR` rows dropped, age ≥ 18, organ
donor accounts excluded, mortality from `HOSPITAL_EXPIRE_FLAG`, readmission as an
unplanned admission within 30 days (elective excluded, in-hospital deaths
excluded from the readmission cohort).

Two issues are handled explicitly because both are common ways clinical NLP
results turn out to be inflated:

**Label leakage.** A discharge summary for a patient who died usually says so —
"patient expired", "time of death", "autopsy", "Discharge Disposition: EXPIRED".
Training on that is not prediction. `mask_outcome_leakage()` redacts twelve such
patterns, and the flag is recorded in the results file so the with/without
ablation can be reported rather than hidden.

**Identity leakage.** MIMIC patients have repeat admissions. If a patient appears
in both train and test, the model can memorise the patient rather than learn
clinical signal. Splitting is therefore `StratifiedGroupKFold` on `subject_id`,
and `check_leakage()` asserts both patient disjointness and absence of duplicate
note text across splits.

### 2.4 Preprocessing decisions

Two rules govern `medxplain/data/preprocess.py`:

1. **No lossy normalisation before explanation.** The text the model sees is the
   text the UI highlights, so cleaning never silently deletes content a clinician
   would expect highlighted. De-ID placeholders become typed tokens (`<DATE>`,
   `<PLACE>`, `<NAME>`) rather than being deleted, because deleting them would
   shift every downstream character offset.
2. **Optional steps stay optional.** Abbreviation expansion changes token
   boundaries and therefore changes explanations, so it is **off by default** and
   exposed as an ablation rather than baked in.

One measured decision: MIMIC-style administrative headers (`Admission Date:`,
`Service:`, `Attending:`) are stripped. They are pure boilerplate and consumed
~60 wordpieces — roughly 25% of the notes' length. Removing them cut the mean
from 459 to 341 tokens, which materially changes what survives truncation.

**Long documents.** 96% of notes exceed 256 tokens. Two mechanisms:

- **`head_tail` truncation** for training: keep the first 35% and last 65% of the
  budget. Plain right-truncation discards the discharge disposition, which is
  where goals-of-care language (the single strongest mortality signal in real
  notes) lives.
- **Sliding-window chunking with pooled logits** for inference, so a full
  document can be scored. `max` pooling is the default because it matches the
  clinical semantics — one window describing a cardiac arrest should drive risk
  up regardless of surrounding routine text.

---

## 3. Models

### 3.1 Backbones compared

| Key | Checkpoint | Pretraining corpus |
|---|---|---|
| `clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes |
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` | PubMed abstracts + PMC full text |
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | PubMed, **from-scratch** vocabulary |
| `bert` | `bert-base-uncased` | BooksCorpus + Wikipedia |

`bert-base` is not filler — it is the **control that isolates the benefit of
domain adaptation**. Without it, "ClinicalBERT gets 0.87 AUROC" says nothing
about whether clinical pretraining helped. PubMedBERT is interesting separately
because it trains its WordPiece vocabulary from scratch on biomedical text rather
than inheriting general-domain subwords, so `norepinephrine` is not shredded into
five pieces.

### 3.2 Architecture

A deliberately thin wrapper (`medxplain/models/transformer.py`), *not*
`AutoModelForSequenceClassification`, for two reasons:

1. We need `output_attentions` plumbed through inference without paying for it
   during training, and we need the raw per-layer attention tensors for rollout.
2. We need `inputs_embeds` accepted so integrated gradients can interpolate in
   embedding space.

Encoder → pooling (`cls` / `mean` / `cls_mean`) → dropout → linear head.

### 3.3 Training protocol

`medxplain/train/trainer.py` is hand-written rather than `transformers.Trainer`
so the choices a reviewer asks about are visible in one file.

- **Class imbalance.** Three options selectable by config: unweighted CE (the
  control), inverse-frequency weighted CE, and focal loss (γ=2). At 16.6%
  positive the weights are [0.60, 3.01]. Treating imbalance as an experimental
  variable rather than a hidden default matters, because the choice moves F1
  substantially.
- **Model selection on validation AUPRC, not loss or accuracy.** This is the most
  consequential single decision in the training loop. With a 16% positive rate,
  validation loss is dominated by the majority class and routinely selects a
  checkpoint that ranks the minority class *worse*. AUPRC is the headline
  rare-event metric, so it is what early stopping optimises.
- **Discriminative learning rates.** The randomly-initialised head gets 10× the
  encoder LR. One shared LR either under-trains the head or destroys the
  pretrained representation.
- **Multi-seed.** Every backbone is trained across multiple seeds and reported as
  mean ± SD. Single-run results are a standard rejection reason.
- **Seed is the outer loop.** One pass covers every backbone, so an interrupted
  run yields a complete single-seed comparison rather than three seeds of one
  model and nothing for the others.

### 3.4 Baselines

Publication credibility rests on the baselines being *tuned*, not handicapped:

- TF-IDF + Logistic Regression (C and n-gram range searched)
- TF-IDF + Random Forest
- TF-IDF + XGBoost (`scale_pos_weight` for imbalance)

All three are tuned on **the same validation split** the transformers use for
early stopping, so model selection is not a confound. All are fitted on
**exactly the same training notes** as the transformers. Each also exposes
`explain_tokens()` in the shared payload format — for logistic regression this is
*exact* (contribution = tfidf value × coefficient, summing to the decision
function), which gives a genuinely faithful reference point to compare
transformer explanations against.

---

## 4. Explainability — the core contribution

Six methods, all returning an identical `Explanation` payload
(`medxplain/explain/payload.py`) so they can be swapped in the API, rendered by
the same UI, and scored by the same metrics.

| Method | Signed? | Cost | What it computes |
|---|---|---|---|
| `attention_raw` | no | 1 forward | mean over heads of last layer's `[CLS]` row |
| `attention_rollout` | no | 1 forward | attention propagated across all layers |
| `attention_grad` | yes | 1 fwd + 1 bwd | attention weighted by class gradient |
| `integrated_gradients` | yes | `n_steps` fwd+bwd | path integral from a masked baseline |
| `occlusion` | yes | 1 fwd per word | Δ log-odds when each word is masked |
| `shap` | yes | hundreds fwd | partition SHAP over the note |

### 4.1 Attention rollout

Raw last-layer attention is what most clinical XAI systems ship and what "Attention
is not Explanation" (Jain & Wallace, 2019) criticises. Rather than take a side, we
implement both and measure the difference on the same model.

Rollout (Abnar & Zuidema, 2020) accounts for residual connections and layer
composition:

```
Â_l = normalise( 0.5 · mean_heads(A_l) + 0.5 · I )
R   = Â_L · Â_{L-1} · … · Â_1
importance = R[0, :]        # the [CLS] row
```

The `0.5·I` term models the residual stream; without it, rollout attributes
importance that in fact bypassed the attention mechanism entirely.

**Attention rollout is class-agnostic.** It produces the identical map whether
the model predicts high or low risk — it shows what the `[CLS]` representation
was *built from*, not which class the evidence supports. This is a real
limitation, it is stated in the API payload (`signed: false`), and it is the
reason the signed methods are included alongside it.

### 4.2 SHAP

Partition SHAP with a `Text` masker: O(n log n) model evaluations instead of the
2ⁿ a naive Shapley computation needs. Two decisions worth defending:

- **Computed on log-odds, not probability.** Shapley values are additive, and
  additivity only holds on a scale where the model is locally linear. On the
  probability scale the values stop summing to the prediction once the sigmoid
  saturates — which is exactly the high-risk regime we care about.
- **The masking baseline is a config value**, because replacing tokens with
  `[MASK]` versus deleting them produces systematically different attributions.
  That is a methodological choice, not an implementation detail.

### 4.3 Integrated Gradients

```
IG_i = (x_i − x'_i) · ∫₀¹ ∂F(x' + α(x − x')) / ∂x_i  dα
```

approximated by Riemann midpoint over `n_steps`. The baseline `x'` is the
all-`[MASK]` sequence with special tokens preserved, **not** the zero embedding:
a zero vector is not a point in the model's input distribution for text, and
using it makes attributions dominated by how far each token's embedding happens
to sit from the origin.

IG satisfies **completeness** — attributions sum to `F(x) − F(x')` — which gives a
built-in correctness check. We report `convergence_delta` on every explanation,
and there is a test asserting that increasing `n_steps` reduces it. Attention has
no analogous self-check.

### 4.4 The payload contract

Two granularities are carried simultaneously:

- **token level** — the model's own subword units, which is what faithfulness
  metrics must operate on (they must delete exactly what the model consumes);
- **word level** — subwords merged back into whitespace words with character
  offsets into the *original* note, which is what a clinician reads.

This avoids a subtle and common bug: scoring an explanation on wordpieces while
displaying it on words, so the number in the table describes something different
from the picture in the figure. Subword scores are summed by default —
"norepinephrine" should not lose credit because the tokenizer split it five ways.

---

## 5. Evaluation methodology

### 5.1 Predictive performance

AUROC, **AUPRC** (treated as the headline number), F1, precision, recall,
specificity, balanced accuracy, Brier score, Brier skill score, and ECE.

AUPRC leads the discussion because at 16% prevalence AUROC is inflated by the
ease of ranking abundant negatives; two models with equal AUROC can differ
substantially in the precision a clinician sees at a usable alert threshold. The
no-skill floor (= prevalence) is carried in every output so a reader judges lift
rather than a raw number. We also report **operating points** — precision and
alert rate at 50/70/80/90% recall — which is the form a clinical service actually
needs.

### 5.2 Faithfulness (ERASER)

The central question: does the explanation describe what the model actually did,
or merely something that looks sensible?

- **Comprehensiveness** = `p(ŷ|x) − p(ŷ|x \ top-k)`. Remove the highlighted
  tokens; a faithful explanation should hurt. Higher is better.
- **Sufficiency** = `p(ŷ|x) − p(ŷ|top-k only)`. Keep only the highlighted tokens;
  a faithful explanation should nearly reproduce the prediction. Lower is better.

Averaged over k ∈ {1, 5, 10, 20, 50}% of tokens, since a single k is easy to
cherry-pick. Three implementation choices make the numbers defensible:

1. **Removal means masking, not deletion.** Deleting tokens shortens the sequence
   and shifts every position, so part of the measured drop would be a *length
   artefact* rather than evidence removal.
2. **A random-explanation control is computed alongside every result.** Masking
   any 20% of a note lowers confidence somewhat. The scientifically interesting
   quantity is how much *more* the explanation's tokens hurt than random ones do,
   reported as `comprehensiveness_over_random`. Papers that omit this control are
   reporting an effect confounded with "masking text hurts".
3. **Ranking uses signed scores toward the predicted class**, not absolute value.
   Ranking by |score| would mix in the strongest evidence *against* the
   prediction and understate comprehensiveness.

Plus **deletion/insertion curves** and their AUCs: a faithful explanation gives a
low deletion AUC (confidence collapses as its top tokens go) and a high insertion
AUC (confidence recovers quickly as they return).

### 5.3 Stability

Rarely reported in clinical XAI, and a genuine gap. A clinician who pastes the
same note with "SOB" and with "shortness of breath" should not get different
highlights. Three perturbations of increasing severity:

- **near-duplicate** — whitespace/casing only; any change is pure noise
- **paraphrase** — abbreviation and synonym swaps that preserve clinical meaning
- **distractor** — an irrelevant sentence appended; importance landing on it is
  wasted attribution, reported directly as `distractor_mass`

Measured three ways because they answer different questions: rank correlation
(does the ordering survive?), top-k Jaccard (does the clinician see the same
highlights?), and prediction shift (did the model itself move, which would make
explanation change legitimate?).

### 5.4 Plausibility

- **Automatic** (synthetic corpus): token-F1 and rationale AUPRC against gold
  spans, plus weight correlation and sign agreement.
- **Human** (any corpus): `scripts/06_plausibility.py` emits a **blinded,
  per-note-shuffled review packet** — an HTML form plus a CSV rating sheet.
  Method names are replaced with neutral labels A/B/C and reshuffled for every
  note, so a rater cannot learn "B is always SHAP". Raters score clinical
  sensibility and completeness on a 1–5 Likert rubric. Scoring computes per-method
  means and **Krippendorff's α for ordinal data** — chosen over Fleiss' κ because
  it handles missing ratings and respects the ordering of a Likert scale (κ treats
  a 1-vs-5 and a 4-vs-5 disagreement as equally severe, which understates
  agreement on ordinal rubrics).

### 5.5 Statistical rigour

- The headline model is selected on **validation** AUPRC and evaluated on test
  **once**. Choosing the winner by test score is a standard way to over-report.
- **Paired** tests, because both models saw identical patients: DeLong (1988,
  fast implementation of Sun & Xu 2014) for correlated AUROCs; paired stratified
  bootstrap for AUPRC where no closed-form variance exists; paired permutation as
  a third option.
- **Holm–Bonferroni** across the family of baseline comparisons — uniformly more
  powerful than plain Bonferroni at the same family-wise error rate.
- Domain adaptation tested separately at the **seed level** against BERT-base,
  with Cohen's *d* alongside *p*, since with 3 seeds a p-value alone is nearly
  uninformative.
- Bootstrap CIs are **stratified by class**, because with a rare outcome an
  unstratified bootstrap produces folds with no positives where AUROC is
  undefined, silently biasing the interval.

---

## 6. The system

A working full-stack application, not a notebook — which strengthens the "system
contribution" angle that applied venues reward.

```
Next.js 14 + Tailwind + Recharts  (:3000)
  note input · risk gauge · highlighted note · importance chart
  method toggle/compare · results dashboard · session log
                    │ REST/JSON
FastAPI  (:8000)
  /predict /models /methods /examples /results /history /predict/batch
  cleaning · inference · explanation · LRU cache · guard rails
                    │
PyTorch + HuggingFace
  fine-tuned encoder · 6 explainers · attention extraction
```

Design points worth mentioning to a reviewer:

- **The UI cannot lie about direction.** Signed methods render on a diverging
  red/blue scale; unsigned methods render on a single hue, and the "protective
  words" panel is suppressed entirely. Showing a low-attention word in
  "protective blue" would invent a claim attention cannot make.
- **Highlight intensity is scaled to the note's own peak**, not a fixed absolute
  range — attributions differ by orders of magnitude across methods, so a fixed
  scale renders most notes uniformly blank.
- **The threshold is drawn on the gauge.** For a rare outcome the operating point
  is nowhere near 50%; a gauge without it invites "under half the dial = safe".
- **Truncation is surfaced.** If a note exceeds the context window the response
  says so, because the highlights then describe only the part the model read.
- **Note text is never written to disk.** The session log is an in-memory ring
  buffer that dies with the process — appropriate for a tool people paste
  clinical text into.
- Character offsets from the backend drive the highlighting, rather than
  re-tokenising in the browser, so highlights land exactly on what was scored.

---

## 7. Current status and results

### Completed and verified

- 45-module research package; FastAPI backend (10 endpoints); Next.js frontend
  (TypeScript compiles clean); 8 pipeline scripts; 3 config profiles; Docker +
  compose; Makefile; README.
- **98 tests passing** — covering gold-span alignment, character offsets
  surviving tokenisation and truncation, patient-disjoint splitting, subword→word
  aggregation, IG completeness, occlusion scores matching real confidence drops,
  metric definitions against closed forms, and the full API surface.

### Baseline results (test n = 429, 16.6% positive, 1,000 training notes)

| Model | AUROC | AUPRC |
|---|---|---|
| TF-IDF + Logistic Regression | 0.862 | 0.632 |
| TF-IDF + XGBoost | 0.849 ± 0.013 | 0.582 ± 0.017 |
| TF-IDF + Random Forest | 0.843 ± 0.001 | 0.554 ± 0.003 |

### Transformer results — in progress

4 backbones × 2 seeds currently training. First epoch of ClinicalBERT reached
validation AUROC 0.698 / AUPRC 0.343 — **below the TF-IDF baseline**.

This is expected and should be reported honestly rather than tuned away. See §8.

---

## 8. The compute constraint, stated plainly

Development hardware is **CPU-only** (no GPU), at roughly 10 seconds per training
step. The full protocol — 4 backbones × 3 seeds × 512 tokens × full fine-tuning —
would take approximately 18 hours. To get an end-to-end result, the pilot was
reduced to:

| Setting | Pilot | Full protocol (`configs/mimic_full.yaml`) |
|---|---|---|
| Training notes | 1,000 | full split (~50k admissions on MIMIC) |
| Max sequence length | 128 | 512 |
| Encoder layers frozen | bottom 8 of 12 | 0 (full fine-tuning) |
| Epochs | 2 | 6 with early stopping |
| Seeds | 2 | 5 |

**This is why the transformer currently underperforms TF-IDF, and the reason is
not mysterious.** With 1,000 training examples, only the top four encoder layers
trainable, and a 128-token window that cannot reach the hospital course of a
327-token note, the model is closer to a linear probe on partially-adapted
features than to a fine-tuned classifier. Meanwhile TF-IDF + LR has full access
to every word in the document and only ~8k parameters to fit — it is exactly the
regime where the classical baseline is expected to win.

This is a well-documented result in small-data clinical NLP, and it is worth
presenting as a finding rather than hiding: **domain-adapted transformers do not
dominate a tuned bag-of-words baseline until you have enough data and enough
context length to justify them.** The honest claim available from the pilot is
about the *explanation methods*, which are compared on the same model and are
unaffected by that model's absolute accuracy.

The fair comparison requires a GPU run of `configs/mimic_full.yaml`. Both configs
are in the repository so the difference is explicit rather than a footnote.

---

## 9. What is genuinely novel here

Most clinical XAI work uses **one** explanation method, on **one** model, and
validates it by **visual inspection**. The defensible contributions of this work:

1. **A head-to-head comparison of six explanation methods on the same model and
   the same notes**, across three independent axes — faithfulness, stability and
   plausibility. These can disagree, and showing where they disagree is the
   result.
2. **A random-explanation control on every faithfulness number.** Without it,
   comprehensiveness partly measures "masking text hurts".
3. **Explanation stability under meaning-preserving edits**, which is almost
   never reported and is a precondition for clinical use.
4. **The cost/faithfulness trade-off measured directly.** SHAP needs hundreds of
   forward passes; occlusion needs one per word; attention needs one. If occlusion
   matches SHAP on faithfulness at a fraction of the cost, that is the deployable
   choice and the paper should say so. `fig6_cost_vs_faithfulness.png` plots it.
5. **Sign agreement as a metric**, quantifying what is lost by using an unsigned
   attribution method — a concrete, measurable statement of the attention critique.
6. **A working system**, which applied venues weight favourably as evidence of
   real-world applicability.

---

## 10. Limitations

- **Not a medical device.** Research prototype, not validated for clinical use.
- The pilot is trained on synthetic data. Its numbers characterise the pipeline
  and the relative behaviour of explanation methods, **not clinical performance**.
- Gold-rationale plausibility scoring exists only for the synthetic corpus. On
  MIMIC-III, plausibility must be human-rated — the blinded review harness is
  provided for exactly that, but the study has not been run.
- The current transformer results are compute-constrained and below the classical
  baseline; the GPU run is outstanding.
- Attention rollout is class-agnostic by construction. This is a property of the
  method, not a bug, but it limits what an attention-only system can claim.
- SHAP at a usable budget costs tens of seconds per note on CPU. Whether that is
  deployable at the bedside is itself one of the findings.
- The synthetic generator's risk weights are clinically plausible but not
  empirically calibrated; they establish an ordering, not effect sizes.

---

## 11. Reproducing everything

```bash
pip install -r backend/requirements.txt && (cd frontend && npm install)

python scripts/00_build_dataset.py      # corpus, splits, data report
python scripts/01_train_baselines.py    # TF-IDF baselines
python scripts/02_train_transformers.py --resume
python scripts/03_evaluate.py           # metrics, calibration, significance
python scripts/04_faithfulness.py       # comprehensiveness / sufficiency
python scripts/05_stability.py          # paraphrase agreement
python scripts/06_plausibility.py       # gold rationales + review packet
python scripts/07_figures_tables.py     # 8 figures, 4 LaTeX tables
```

or `make pipeline`, or `./scripts/run_all.sh`.

Reproducibility guarantees:

1. `results/config_used.yaml` records the exact resolved configuration.
2. Every metrics file embeds an `environment` block (package versions, device).
3. Seeds are fixed in config and threaded through Python, NumPy and Torch.
4. `07_figures_tables.py` regenerates every figure and table **from saved metrics
   alone** — it never retrains, so a figure cannot drift from the numbers it plots.

---

## References

- Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.*
- Alsentzer et al. (2019). *Publicly Available Clinical BERT Embeddings.*
- Chefer, Gur & Wolf (2021). *Transformer Interpretability Beyond Attention Visualization.*
- DeLong, DeLong & Clarke-Pearson (1988). *Comparing the Areas under Two or More Correlated ROC Curves.*
- DeYoung et al. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models.*
- Gu et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical NLP.*
- Jain & Wallace (2019). *Attention is not Explanation.*
- Johnson et al. (2016). *MIMIC-III, a freely accessible critical care database.*
- Lee et al. (2020). *BioBERT: a pre-trained biomedical language representation model.*
- Lin et al. (2017). *Focal Loss for Dense Object Detection.*
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
- Sundararajan, Taly & Yan (2017). *Axiomatic Attribution for Deep Networks.*
- Wiegreffe & Pinter (2019). *Attention is not not Explanation.*
