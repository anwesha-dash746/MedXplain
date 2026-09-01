# MedXplain: Explainable Transformer-Based Clinical Risk Prediction

**An Explainable Transformer-Based Framework for Clinical Risk Prediction from Unstructured Clinical Notes**

*End-to-end project: full-stack application + publication-grade model, built for a Scopus-indexed IEEE/Springer conference submission.*

---

## 1. Project Summary

MedXplain is an NLP system that reads unstructured clinical notes (e.g., hospital discharge summaries) and predicts a clinical risk outcome — such as in-hospital mortality risk or 30-day readmission risk — using transformer-based language models. Unlike typical "black box" deep learning models, MedXplain also **explains its predictions** by highlighting the exact words and phrases in the note that most influenced the risk score, using attention visualization and SHAP, surfaced through a full interactive web application rather than a static script or notebook.

**Core idea in one line:** *Predict patient risk from free-text notes, and show the clinician exactly which words drove that prediction — in a real, usable interface.*

Because this project is also meant to produce a publishable result, every phase below is written with **reproducibility, rigorous baselines, and defensible evaluation** in mind, since these are what reviewers at IEEE/Springer Scopus-indexed venues will scrutinize.

---

## 2. Problem Statement

- Clinical notes contain rich, high-value information about a patient's condition, but it's locked in unstructured free text (not spreadsheets or coded fields).
- Transformer models like ClinicalBERT and BioBERT can extract predictive signal from this text effectively.
- However, these models are opaque — they output a risk score with no justification.
- Clinicians are unlikely to trust or act on a prediction they cannot verify or understand.
- **Goal:** build a system that is both accurate (strong predictive performance) and interpretable (clear, faithful reasoning), closing the trust gap between AI predictions and clinical decision-making — packaged as a usable tool, not just a model checkpoint.

---

## 3. Where NLP Fits In

| Component | NLP Role |
|---|---|
| Input data | Free-text clinical notes — unstructured natural language |
| Preprocessing | Tokenization, handling medical abbreviations/jargon, truncation/chunking for long documents |
| Core model | ClinicalBERT / BioBERT / PubMedBERT — transformer language models pretrained on medical text |
| Task | Text classification (binary or multi-label) — a core NLP task |
| Explainability | Token/word-level attribution (attention weights, SHAP values) — explanation is anchored to language, not generic tabular features |

---

## 4. Key Concepts & Terminology (for reference)

- **Transformer**: A neural network architecture that processes sequences of text using self-attention, weighing the importance of different words relative to each other.
- **BERT**: Bidirectional Encoder Representations from Transformers — a pretrained language model that understands context from both directions in a sentence.
- **ClinicalBERT / BioBERT / PubMedBERT**: Domain-adapted BERT variants pretrained on clinical notes or biomedical literature, giving them better understanding of medical terminology than generic BERT.
- **Fine-tuning**: Taking a pretrained model and training it further on a specific labeled dataset/task (e.g., mortality prediction).
- **Attention visualization**: Displaying which tokens the model "attended to" most heavily when making a prediction.
- **SHAP (SHapley Additive exPlanations)**: A model-agnostic method (game-theoretic) that assigns each input token a contribution score toward the final prediction.
- **Faithfulness (in explainability)**: Whether an explanation actually reflects the model's real decision process, as opposed to just looking plausible to a human. Measured via tests like comprehensiveness and sufficiency (see Section 9).
- **AUROC**: Area Under the ROC Curve — measures how well a binary classifier separates positive and negative classes.
- **F1-score**: Harmonic mean of precision and recall — useful when classes are imbalanced (e.g., mortality is rare).

---

## 5. Dataset Options

| Dataset | Description | Access |
|---|---|---|
| **MIMIC-III** | De-identified ICU discharge summaries + outcome labels (mortality, readmission) from Beth Israel Deaconess Medical Center | Free PhysioNet account + CITI "Data or Specimens Only Research" training (approval can take days–weeks) |
| **i2b2/n2c2 datasets** | Shared-task datasets for clinical NLP (de-identification, NER, relation extraction) | Requires Data Use Agreement, usually faster than MIMIC |
| **Synthetic/public Kaggle datasets** | Symptom-disease text datasets, synthetic clinical notes | No approval needed — good for early prototyping only, not for the final paper-quality results (reviewers will expect a recognized clinical benchmark like MIMIC-III) |

**Recommendation:** Apply for MIMIC-III access immediately (longest lead time). Prototype the pipeline on a public/synthetic dataset in parallel, then switch to MIMIC-III for all final reported results — publication reviewers expect a standard, citable clinical dataset.

---

## 6. System Architecture (Full-Stack, End-to-End)

This is a real application with a frontend, backend, and model-serving layer — not a single script.

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Next.js)                │
│  - Clinician-style dashboard UI                                │
│  - Note input (paste text / upload file)                       │
│  - Risk score display (gauge/badge, confidence interval)       │
│  - Highlighted note view (color-coded by SHAP/attention score) │
│  - Token-level importance chart (bar chart of top contributors)│
│  - Model/explanation-method toggle (attention vs. SHAP)        │
│  - Prediction history / session log                            │
└───────────────────────────┬─────────────────────────────────┘
                             │ REST/JSON (HTTPS)
┌───────────────────────────▼─────────────────────────────────┐
│                   BACKEND API (FastAPI)                       │
│  - /predict endpoint: text -> risk score + explanation payload │
│  - Preprocessing (cleaning, tokenization, chunking)             │
│  - Model inference (loads fine-tuned transformer)               │
│  - Explainability computation (attention extraction + SHAP)     │
│  - Response caching for repeated notes (optional)                │
└───────────────────────────┬─────────────────────────────────┘
                             │
┌───────────────────────────▼─────────────────────────────────┐
│              MODEL LAYER (PyTorch / HuggingFace)               │
│  - Fine-tuned ClinicalBERT / BioBERT / PubMedBERT               │
│  - Classification head (binary or multi-label)                   │
│  - Attention weight extraction (output_attentions=True)          │
│  - SHAP explainer wrapping the model                              │
└─────────────────────────────────────────────────────────────┘
```

### Frontend tech stack (proper interactive app, not a notebook demo)
- **Framework**: React with Next.js (or plain React + Vite if you don't need server-side rendering)
- **Styling/UI**: Tailwind CSS + a component library (e.g., shadcn/ui) for a polished, professional clinical-dashboard look
- **Visualization**: Recharts or D3.js for token-importance bar charts; custom highlighted-text component for inline word-level coloring
- **State management**: React hooks (useState/useReducer) or Zustand for larger state needs
- **API communication**: `fetch`/`axios` calls to the FastAPI backend

### Backend tech stack
- **Framework**: FastAPI (Python) — fast, async, auto-generates OpenAPI docs, integrates naturally with PyTorch/HuggingFace
- **Model serving**: Load the fine-tuned model once at startup; serve inference via a `/predict` endpoint
- **Explainability service**: Compute attention weights and SHAP values server-side, return them as structured JSON (token + score pairs) for the frontend to render
- **Deployment**: Containerize with Docker; can be deployed on a single GPU/CPU instance (Render, Railway, or a university lab server) for demo purposes

### Why this matters for your paper
A working full-stack system (not just an offline notebook) strengthens the "system contribution" angle of your paper — many accepted clinical NLP papers include a system/demo component, which reviewers view favorably as evidence of real-world applicability.

---

## 7. Implementation Phases

### Phase 0 — Setup & Scoping (Week 1)
- Finalize the exact prediction target (e.g., in-hospital mortality, 30-day readmission) — pick ONE as the primary result; others can be secondary experiments.
- Apply for PhysioNet/MIMIC-III access immediately (longest lead time item).
- Set up environment: Python, PyTorch, HuggingFace `transformers`, `shap`, `scikit-learn`, `pandas`; Node.js + React/Next.js for frontend; FastAPI for backend.
- Initialize a Git repository with a clear structure (`/backend`, `/frontend`, `/notebooks`, `/data`, `/results`) — reviewers and future readers of your paper's code release will expect this.
- While waiting for data access, build the pipeline skeleton and frontend shell using a small public/synthetic dataset.

### Phase 1 — Data Preparation (Weeks 2–3)
- Load and clean the clinical notes dataset.
- Handle missing values, de-identification artifacts, and inconsistent formatting.
- Split data into train/validation/test sets (stratified by outcome label), and hold out the test set strictly for final reporting only.
- Address class imbalance (mortality/readmission are rare events) via class weighting, oversampling, or focal loss.
- Tokenize using the ClinicalBERT/BioBERT tokenizer; decide on max sequence length and a chunking or sliding-window strategy for long notes.
- Document every preprocessing decision — this becomes your paper's "Data & Preprocessing" section.

### Phase 2 — Baselines (Week 4)
For a defensible publication, use **multiple baselines**, not just one:
- TF-IDF + Logistic Regression
- TF-IDF + traditional ML (e.g., Random Forest or XGBoost)
- A non-domain-adapted transformer (plain BERT-base) — to isolate the benefit of domain adaptation specifically
- (If feasible) a prior published result on the same task/dataset, cited for comparison

### Phase 3 — Transformer Model Fine-Tuning (Weeks 5–6)
- Fine-tune multiple domain-adapted transformer variants for comparison: **ClinicalBERT**, **BioBERT**, and **PubMedBERT** (or Bio_ClinicalBERT) — comparing which domain-pretraining choice performs best is itself a useful, citable finding.
- Use proper hyperparameter search (learning rate, batch size, epochs, max sequence length) via a validation set — grid search or a lightweight tool like Optuna.
- Run each configuration across multiple random seeds (e.g., 3–5 seeds) and report mean ± standard deviation — single-run results are a common rejection reason at good venues.
- Apply early stopping based on validation AUROC to prevent overfitting.

### Phase 4 — Explainability Integration (Weeks 7–8)
- **Attention visualization**: Extract attention weights from the fine-tuned model; aggregate across heads/layers using an established method (e.g., attention rollout) rather than a single raw layer.
- **SHAP**: Apply SHAP's `Explainer` for transformers to compute per-token contribution scores.
- Build the explanation payload format (token + score pairs) that the backend will serve to the frontend.
- Implement the frontend's highlighted-text rendering and importance-chart components, wired to live backend predictions.

### Phase 5 — Evaluation (Weeks 9–10)
**Predictive performance:**
- AUROC, AUPRC (important given class imbalance), F1-score, precision, recall, calibration (e.g., Brier score or a reliability diagram).
- Statistical significance testing between your best model and each baseline (e.g., bootstrapped confidence intervals or a paired significance test) — reviewers expect this, not just point-estimate comparisons.

**Explanation quality (this is where a real research contribution can live):**
- **Faithfulness metrics** (borrowed from the ERASER benchmark and XAI literature):
  - *Comprehensiveness*: How much does removing the top-k important tokens drop the model's confidence in its prediction? (Larger drop = more faithful explanation.)
  - *Sufficiency*: How well does the model perform using *only* the top-k important tokens? (Higher = more faithful.)
  - *Deletion/insertion curves*: Plot prediction confidence as top tokens are progressively removed/added.
- **Stability**: Do explanations stay consistent for near-duplicate or paraphrased notes?
- **Plausibility (human-facing)**: A small qualitative or structured review — ideally with a clinician or clinically-informed rater — scoring whether highlighted phrases make clinical sense. Even 2–3 raters with a simple Likert-scale rubric strengthens this section considerably.
- Compare attention-based vs. SHAP-based explanations directly on these metrics — this comparison is a natural "contribution" angle for the paper, since most existing clinical XAI work uses only one method in isolation.

### Phase 6 — Full-Stack Integration & Polish (Weeks 10–11)
- Connect the frontend to the trained model via the FastAPI backend end-to-end.
- Add error handling (e.g., empty input, extremely long notes, non-clinical text).
- Add a loading/inference state in the UI (transformer inference isn't instant).
- Polish the UI: consistent color scale for importance highlighting, tooltips explaining what SHAP/attention mean, a toggle between explanation methods, and a clean, professional clinical-dashboard aesthetic.
- Optional but strong for a demo: a short case-study mode with a few pre-loaded example notes (synthetic/de-identified) so evaluators/reviewers can try it without needing their own clinical text.

### Phase 7 — Final Results, Figures & Repository Cleanup (Week 12)
- Generate all final figures: ROC/PR curves, attention heatmaps, SHAP summary plots, deletion/insertion curves, model comparison tables.
- Clean up the codebase, write a clear `README.md` (setup instructions, how to reproduce results, how to run the frontend/backend).
- Ensure results are reproducible: fix random seeds, log package versions (`requirements.txt` / `package.json`), and save trained model checkpoints or a script to regenerate them.

---

## 8. Evaluation Plan Summary

| Aspect | Metric/Method |
|---|---|
| Predictive accuracy | AUROC, AUPRC, F1-score, precision/recall, calibration |
| Baseline comparison | TF-IDF+LR, TF-IDF+XGBoost, plain BERT, ClinicalBERT, BioBERT, PubMedBERT |
| Statistical rigor | Multiple seeds (mean ± std), significance testing / bootstrapped CIs |
| Explanation faithfulness | Comprehensiveness, sufficiency, deletion/insertion curves |
| Explanation stability | Consistency across paraphrased/near-duplicate notes |
| Explanation plausibility | Small structured human/clinician review (Likert-scale rubric) |
| Class imbalance handling | Class weighting, oversampling, or focal loss |

---

## 9. Tools & Technologies

**Modeling**
- Python, PyTorch
- HuggingFace `transformers` (ClinicalBERT, BioBERT, PubMedBERT)
- `shap` for explainability
- `scikit-learn` for classical baselines and metrics
- `Optuna` (optional) for hyperparameter search

**Frontend**
- React + Next.js
- Tailwind CSS + shadcn/ui
- Recharts or D3.js for importance visualizations

**Backend**
- FastAPI
- Docker for containerized deployment
- Uvicorn/Gunicorn for serving

**Experiment tracking & reproducibility**
- Weights & Biases or MLflow (recommended for tracking multi-seed, multi-model runs cleanly — very useful once you're comparing several models × several seeds × two explanation methods)
- Git + GitHub for version control and eventual code release alongside the paper

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MIMIC-III access delayed | Start with synthetic/public data; build full pipeline and frontend in parallel |
| Class imbalance skews results | Class weighting/oversampling/focal loss; report AUROC & AUPRC, not just accuracy |
| Long clinical notes exceed model's max token length | Chunking/sliding-window strategy, or a long-document model variant (e.g., Longformer) |
| Explanations look reasonable but aren't rigorously validated | Use faithfulness metrics (comprehensiveness/sufficiency), not just visual inspection — this is exactly what elevates the work to publication quality |
| Frontend scope creep delays the model work | Build a minimal working frontend first (input → prediction → basic highlight), then layer on polish/charts once the model pipeline is stable |
| Reviewers flag single-run results | Always report multi-seed mean ± std for your main results table |
| Scope too large for timeline | If time runs short, cut to one primary model (best-performing of ClinicalBERT/BioBERT/PubMedBERT) with full evaluation, keep others as a smaller comparison table |

---

## 11. Suggested Timeline Overview

| Phase | Weeks |
|---|---|
| Setup & scoping | 1 |
| Data preparation | 2–3 |
| Baselines | 4 |
| Transformer fine-tuning (multi-model, multi-seed) | 5–6 |
| Explainability integration | 7–8 |
| Evaluation (performance + faithfulness) | 9–10 |
| Full-stack integration & polish | 10–11 |
| Final results, figures, repo cleanup | 12 |

---

## 12. Deliverables Checklist

- [ ] PhysioNet/MIMIC-III access approved
- [ ] Cleaned, split, tokenized dataset with documented preprocessing
- [ ] Multiple baselines trained and evaluated
- [ ] Multiple transformer variants fine-tuned (multi-seed) and compared
- [ ] Attention visualization implemented (with rollout aggregation)
- [ ] SHAP explanations implemented
- [ ] Faithfulness metrics computed (comprehensiveness, sufficiency, deletion/insertion curves)
- [ ] Explanation stability check across paraphrased notes
- [ ] Small structured plausibility review completed
- [ ] React/Next.js frontend built and connected to FastAPI backend
- [ ] End-to-end demo working (paste note → risk score → highlighted explanation)
- [ ] All figures/tables generated for final results
- [ ] Clean, documented, reproducible GitHub repository
