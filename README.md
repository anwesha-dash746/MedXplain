# MedXplain

**An explainable transformer framework for clinical risk prediction from unstructured clinical notes.**

MedXplain reads a free-text clinical note (a discharge summary, say), predicts a
clinical risk outcome, and shows which words drove that prediction — through a
working web application, not a notebook. It is built to support a publication:
multiple baselines, multi-seed training, statistical testing, and an explanation
evaluation that measures *faithfulness* rather than assuming it.

```
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND  Next.js + Tailwind + Recharts        :3000        │
│  note input · risk gauge · highlighted note · importance     │
│  chart · method toggle · results page · session log          │
└───────────────────────────┬──────────────────────────────────┘
                            │ REST/JSON
┌───────────────────────────▼──────────────────────────────────┐
│  BACKEND  FastAPI                               :8000        │
│  /predict · /models · /methods · /examples · /results        │
│  cleaning · inference · explanation · LRU cache              │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  MODEL  PyTorch + HuggingFace                                │
│  Bio_ClinicalBERT / BioBERT / PubMedBERT / BERT-base         │
│  attention rollout · SHAP · integrated gradients · occlusion │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. dependencies
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# 2. build the corpus (synthetic by default; no data access required)
python scripts/00_build_dataset.py

# 3. train — baselines are fast, transformers are not
python scripts/01_train_baselines.py
python scripts/02_train_transformers.py --resume

# 4. evaluate
python scripts/03_evaluate.py       # metrics, calibration, significance
python scripts/04_faithfulness.py   # comprehensiveness / sufficiency
python scripts/05_stability.py      # agreement under paraphrase
python scripts/06_plausibility.py   # gold rationales + human review packet
python scripts/07_figures_tables.py # figures and LaTeX tables

# 5. run the app
python -m uvicorn backend.app.main:app --port 8000   # terminal 1
cd frontend && npm run dev                            # terminal 2
```

Then open <http://localhost:3000>. API docs are at <http://localhost:8000/docs>.

`make pipeline` runs steps 2–4 in order; `make dev` runs both services.
`docker compose up --build` runs the stack in containers (checkpoints are
mounted, not baked into the image).

---

## What is actually being measured

### The prediction task

In-hospital mortality from the discharge summary is the primary outcome;
30-day readmission is carried as a secondary label throughout and can be made
primary with `--set experiment.primary_outcome=readmission_30d`.

### The explanation methods

Six methods, all returning the identical payload shape so they can be compared
on the same metrics and rendered by the same UI:

| Method | Signed? | Cost | What it measures |
|---|---|---|---|
| `attention_raw` | no | 1 forward pass | mean last-layer attention from `[CLS]` |
| `attention_rollout` | no | 1 forward pass | attention rolled out across layers with residuals (Abnar & Zuidema, 2020) |
| `attention_grad` | yes | 1 fwd + 1 bwd | attention weighted by the class gradient (Chefer et al., 2021) |
| `integrated_gradients` | yes | `n_steps` passes | path integral from a masked baseline (Sundararajan et al., 2017) |
| `occlusion` | yes | 1 pass per word | drop in log-odds when each word is masked |
| `shap` | yes | hundreds of passes | partition SHAP over the note |

**Signed matters.** Attention has no direction: a high rollout score means "the
model used this word", not "this word raised the risk". The API says so in the
payload (`signed: false`), and the UI renders unsigned methods on a single-hue
scale precisely so a reader cannot mistake a low-attention word for a protective
finding. This distinction is quietly wrong in a lot of clinical XAI work.

### Faithfulness, not vibes

An explanation that looks sensible can still be a poor description of what the
model did. `scripts/04_faithfulness.py` measures, per ERASER (DeYoung et al., 2020):

- **Comprehensiveness** — how far the model's confidence falls when the top-*k*
  highlighted tokens are masked. Higher is better.
- **Sufficiency** — how much confidence survives when *only* the top-*k* tokens
  are kept. Lower is better.
- **Deletion / insertion curves** and their areas.

Three implementation choices make those numbers defensible:

1. Removal means **masking, not deletion** — deleting tokens shortens the
   sequence, so part of the measured drop would be a length artefact.
2. Every result carries a **random-explanation control**. Masking any 20% of a
   note lowers confidence somewhat; what matters is how much *more* the
   explanation's tokens hurt, reported as `comprehensiveness_over_random`.
3. Tokens are ranked by **signed score toward the predicted class**, not by
   absolute value, which would mix in the strongest evidence *against* the
   prediction.

### Stability and plausibility

- **Stability** (`05`): the same note is re-explained after abbreviation and
  synonym paraphrasing, after whitespace-only edits, and after an irrelevant
  sentence is appended. Agreement is reported as rank correlation, top-*k*
  Jaccard, and `distractor_mass` — the share of importance that lands on the
  inserted irrelevant sentence.
- **Plausibility** (`06`): on the synthetic corpus the generator records the
  character span and true log-odds weight of every risk factor it injected, so
  explanations are scored automatically against gold rationales (token-F1,
  rationale AUPRC, correlation with the true weights, and **sign agreement** —
  does the method mark protective findings as protective?). For any real cohort
  no such ground truth exists, so the script also emits a blinded, shuffled
  **human review packet** (`results/human_review/review_form.html` plus a CSV)
  and scores completed reviews with Krippendorff's alpha.

### Statistical rigour

- Every transformer is trained across multiple seeds; the tables report
  **mean ± SD**, never a single run.
- The headline model is chosen on **validation** AUPRC and evaluated on test
  once. Picking the winner by test score is a common way to over-report.
- Comparisons use a **paired** test — bootstrap for AUPRC, DeLong for AUROC —
  because both models saw identical patients, and **Holm** adjustment across the
  family of baseline comparisons.
- Domain adaptation is tested separately at the **seed level** against the
  BERT-base control, with Cohen's *d* reported alongside *p*, since 3 seeds make
  a *p*-value alone nearly uninformative.

---

## Data

### Synthetic corpus (default, no access required)

`medxplain/data/synthetic.py` generates notes that imitate the surface structure
of MIMIC-III discharge summaries — section headers, `[**de-identification**]`
placeholders, abbreviations, medication lists, boilerplate — with a known
generative process:

```
logit = intercept + age_term + Σ weight(risk factor present)
label ~ Bernoulli(sigmoid(logit))
```

Because the label is *sampled* rather than thresholded, the Bayes-optimal AUROC
is below 1.0 and models cannot saturate the benchmark. Distractor sentences with
zero weight are injected into every section, so an explainer that latches onto
boilerplate is measurably penalised.

**This corpus characterises the pipeline, not clinical performance.** Numbers
from it belong in a paper only as a controlled explainability experiment; the
headline clinical results come from MIMIC-III.

### MIMIC-III

`medxplain/data/mimic.py` builds the real cohort from a local PhysioNet download
and emits the same schema, so switching is a config change:

```bash
python scripts/00_build_dataset.py \
  --set data.source=mimic \
  --set data.mimic_root=/path/to/mimic-iii-clinical-database-1.4
python scripts/02_train_transformers.py --config configs/mimic_full.yaml
```

The cohort definition (admission-level unit, note selection, both outcome
definitions, exclusions) is documented in that module's docstring so it can be
transcribed into the paper verbatim.

**Label leakage is handled explicitly.** A discharge summary for a patient who
died usually says so — "patient expired", "time of death", "autopsy". Training on
that is leakage, not prediction. `mask_outcome_leakage()` removes those phrases
and the flag is recorded in the results file so the with/without ablation can be
reported. This is a known pitfall in MIMIC mortality papers and is worth stating
openly rather than hoping no reviewer asks.

**Splits are patient-disjoint.** A patient with two admissions must not straddle
train and test, or the reported AUROC includes identity memorisation.
`check_leakage()` asserts both patient disjointness and the absence of duplicate
note text across splits.

---

## Repository layout

```
medxplain/                 the research package
  data/                    synthetic generator, MIMIC adapter, preprocessing, splits
    clinical_lexicon.py    risk factors, weights, surface forms, distractors
    synthetic.py           corpus generation with gold rationale spans
    mimic.py               MIMIC-III cohort construction + leakage masking
    preprocess.py          cleaning, chunking, length statistics
    splits.py              stratified/grouped splitting + leakage checks
  models/                  transformer wrapper, TF-IDF baselines, losses
  train/trainer.py         fine-tuning loop, early stopping on validation AUPRC
  explain/                 attention rollout, SHAP, integrated gradients, occlusion
    payload.py             the Explanation contract shared by everything
  eval/                    metrics, faithfulness, stability, significance, plausibility
backend/                   FastAPI service + tests
frontend/                  Next.js dashboard
scripts/                   00..07, the pipeline in order
configs/                   default.yaml (CPU pilot), mimic_full.yaml, smoke.yaml
results/                   metrics, figures, tables, checkpoints (git-ignored)
notebooks/                 exploratory analysis
paper/                     generated tables and figure references
```

---

## Configuration

Everything is driven by `configs/default.yaml`, and the resolved configuration
is written to `results/config_used.yaml` beside the metrics, so any reported
number is traceable to the settings that produced it. Override any key inline:

```bash
python scripts/02_train_transformers.py \
  --set model.backbones='[clinicalbert]' \
  --set train.epochs=4 \
  --set model.max_length=512 \
  --set model.freeze_layers=0
```

Three profiles ship:

| Profile | Purpose |
|---|---|
| `configs/default.yaml` | reduced-scale pilot that completes on a CPU |
| `configs/mimic_full.yaml` | full protocol for MIMIC-III on a GPU: 512 tokens, full fine-tuning, 5 seeds |
| `configs/smoke.yaml` | tiny end-to-end run for CI |

### The CPU pilot is compute-constrained, and says so

The default profile trains at `max_length=128` with the bottom six encoder
layers frozen, for two epochs. That is a deliberate compromise so the pipeline
runs end to end on a laptop; it is **not** the protocol a paper should report.
`configs/mimic_full.yaml` restores full fine-tuning at 512 tokens over five
seeds. Both are in the repository so the difference is explicit rather than
buried in a footnote.

---

## API

`POST /api/predict`

```json
{ "text": "Hospital Course: ...", "model": "clinicalbert", "method": "shap" }
```

returns

```json
{
  "probability": 0.71,
  "predicted_label": 1,
  "decision_threshold": 0.34,
  "risk_band": "elevated",
  "outcome": "in-hospital mortality",
  "explanations": [{
    "method": "shap",
    "signed": true,
    "words": [{ "text": "shock", "score": 0.83, "start": 142, "end": 147 }],
    "top_positive": [...],
    "top_negative": [...],
    "n_model_evaluations": 200,
    "compute_ms": 24310.5
  }],
  "truncated": false,
  "warnings": []
}
```

Two fields exist to stop the response being misread:

- `decision_threshold` is selected on validation, never on test, and for a rare
  outcome it is nowhere near 0.5. `predicted_label` is
  `probability >= decision_threshold`.
- `signed` says whether negative scores mean "lowers the risk".

Other endpoints: `/api/health`, `/api/models`, `/api/methods`, `/api/examples`,
`/api/history`, `/api/results`, `/api/predict/batch`.
`compare_methods` on `/api/predict` returns several explanations of one
prediction in a single call.

Note text is never written to disk — the session log is an in-memory ring buffer
that dies with the process.

---

## Tests

```bash
python -m pytest backend/tests -q
```

The suite runs against a tiny randomly-initialised transformer rather than a real
checkpoint, so it is fast and needs no trained artefacts. It covers the things
that actually break: gold-span alignment, character offsets surviving
tokenisation and truncation, patient-disjoint splitting, subword→word
aggregation, integrated gradients' completeness property, occlusion scores
matching real confidence drops, metric definitions against closed forms, and the
full API surface including validation and error paths.

---

## Reproducing a result

1. `results/config_used.yaml` records the exact configuration.
2. `results/metrics/*.json` records every number plus `environment` (package
   versions, device).
3. Seeds are fixed in the config and threaded through Python, NumPy and Torch.
4. `scripts/07_figures_tables.py` regenerates every figure and table from the
   saved metrics alone — it never retrains, so figures cannot drift from the
   numbers they claim to plot.

---

## Limitations

- **Not a medical device.** Research prototype; not validated for clinical use.
- The shipped pilot is trained on synthetic data. Its numbers describe the
  pipeline and the relative behaviour of explanation methods, not clinical
  performance.
- Gold-rationale plausibility scoring only exists for the synthetic corpus. On
  MIMIC-III, plausibility must be human-rated — the review harness is provided
  for exactly that.
- Attention rollout is class-agnostic: it produces the same map whichever class
  is predicted. This is a property of the method, not a bug, and is why the
  signed methods are included alongside it.
- SHAP at a usable evaluation budget costs tens of seconds per note on CPU.
  Whether that is deployable is itself one of the findings, and
  `fig6_cost_vs_faithfulness.png` plots the trade-off directly.

## References

- Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.*
- Chefer, Gur & Wolf (2021). *Transformer Interpretability Beyond Attention Visualization.*
- DeYoung et al. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models.*
- Jain & Wallace (2019). *Attention is not Explanation.*
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
- Sundararajan, Taly & Yan (2017). *Axiomatic Attribution for Deep Networks.*
- Alsentzer et al. (2019). *Publicly Available Clinical BERT Embeddings.*
- Lee et al. (2020). *BioBERT.* / Gu et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical NLP.*
- Johnson et al. (2016). *MIMIC-III, a freely accessible critical care database.*
