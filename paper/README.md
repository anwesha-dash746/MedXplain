# Paper assets

Everything in this directory is **generated**, not hand-written. `scripts/07_figures_tables.py`
reads `results/metrics/*.json` and writes the figures and tables; it never
retrains anything, so a figure cannot drift from the numbers it claims to plot.

```bash
python scripts/07_figures_tables.py
```

## Where things land

| Asset | Path | Produced by |
|---|---|---|
| ROC + PR curves | `results/figures/fig1_roc_pr.png` | `03_evaluate` → `07` |
| Model comparison bars (multi-seed) | `results/figures/fig2_model_comparison.png` | `03` → `07` |
| Reliability diagram | `results/figures/fig3_calibration.png` | `03` → `07` |
| Deletion / insertion curves | `results/figures/fig4_deletion_insertion.png` | `04` → `07` |
| Faithfulness bars vs. random control | `results/figures/fig5_faithfulness_bars.png` | `04` → `07` |
| Faithfulness vs. compute cost | `results/figures/fig6_cost_vs_faithfulness.png` | `04` → `07` |
| Stability bars | `results/figures/fig7_stability.png` | `05` → `07` |
| Token attribution heatmap | `results/figures/fig8_attention_heatmap.png` | `07` |
| Main results table | `results/tables/main_results.tex` | `03` → `07` |
| Faithfulness table | `results/tables/faithfulness.tex` | `04` → `07` |
| Stability table | `results/tables/stability.tex` | `05` → `07` |
| Plausibility table | `results/tables/plausibility.tex` | `06` → `07` |

Each `.tex` has a `.md` twin for pasting into the README or a slide.

## Suggested paper structure

1. **Introduction** — the trust gap: accurate clinical risk models are opaque,
   and an unexplained score does not change a clinician's decision.
2. **Related work** — clinical BERT variants; attention-as-explanation and its
   critics (Jain & Wallace 2019; Wiegreffe & Pinter 2019); ERASER faithfulness.
3. **Data** — cohort definition, outcome definitions, exclusions, the
   patient-disjoint split, and the leakage masking. Source:
   `medxplain/data/mimic.py` docstring + `results/metrics/dataset_report.json`.
4. **Method** — fine-tuning protocol, long-note handling (head+tail truncation
   and sliding-window pooling), class-imbalance treatment, and the six
   explanation methods behind one payload contract.
5. **Experiments** — baselines, multi-seed transformer comparison, significance
   testing. Table: `main_results.tex`. Figures 1–3.
6. **Explanation evaluation** — the contribution most likely to carry the paper:
   faithfulness, stability and plausibility compared *across* methods on one
   model, including the cost/faithfulness trade-off. Tables:
   `faithfulness.tex`, `stability.tex`, `plausibility.tex`. Figures 4–7.
7. **System** — the FastAPI + Next.js application, with a screenshot of the
   highlighted-note view. Reviewers at applied venues reward a working demo.
8. **Discussion & limitations** — what the explanations do and do not license a
   clinician to conclude; the unsigned-attention caveat; the compute-constrained
   pilot vs. the full protocol.

## Claims to be careful about

- Do not report the synthetic-corpus numbers as clinical performance. They
  characterise the pipeline and the *relative* behaviour of explanation methods.
- Do not describe attention rollout as showing "which words increased the risk".
  It is unsigned and class-agnostic; say "which words the classifier's
  representation was built from".
- Report the number of seeds actually run, and say whether the encoder was fully
  fine-tuned or partially frozen — `results/metrics/transformers.json` records
  `n_trainable` per run, so the claim is checkable.
