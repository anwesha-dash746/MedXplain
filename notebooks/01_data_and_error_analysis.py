"""Exploratory data and error analysis.

Written as a `# %%` script rather than a .ipynb so it diffs cleanly in git and
runs headless in CI. Open it in VS Code or Jupyter (`jupytext --to notebook`) to
step through it interactively, or just run it:

    python notebooks/01_data_and_error_analysis.py

It answers the questions that shape the paper's discussion section:
  1. What does the corpus actually look like (length, prevalence, factors)?
  2. Where is the signal, and does truncation throw it away?
  3. Which cases does the best model get wrong, and do they share anything?
  4. Do the explanations for the wrong cases reveal the failure mode?
"""

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medxplain.config import load_config
from medxplain.utils import read_json

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

cfg = load_config(REPO / "configs" / "default.yaml")
RESULTS = REPO / Path(cfg.get_path("experiment.results_dir", "results"))
LABEL = cfg.get_path("experiment.primary_outcome", "mortality")

df = pd.read_parquet(REPO / cfg.get_path("data.processed_path"))
df["gold_spans"] = df["gold_spans"].map(json.loads)
df["factors"] = df["factors"].map(json.loads)
print(f"{len(df)} notes | outcome={LABEL} | prevalence={df[LABEL].mean():.3f}")
print(df.groupby("split")[LABEL].agg(["size", "mean"]))

# %% [markdown]
# ## 1. Corpus shape
#
# Two things matter for the modelling decisions: how long the notes are relative
# to the encoder's context window, and how skewed the outcome is.

# %%
df["n_words"] = df["text"].str.split().str.len()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

axes[0].hist(df["n_words"], bins=40, color="#265f8c", alpha=0.85)
axes[0].set(xlabel="words per note", ylabel="notes", title="Note length")

rates = df.groupby(pd.cut(df["age"], bins=[18, 45, 60, 70, 80, 95]), observed=True)[LABEL].mean()
axes[1].bar([str(i) for i in rates.index], rates.values, color="#c53030", alpha=0.85)
axes[1].set(xlabel="age band", ylabel=f"{LABEL} rate", title="Outcome by age")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Which risk factors carry the signal?
#
# On the synthetic corpus the generative weights are known, so this is a check
# that the realised corpus behaves the way the generator intended - and a sanity
# baseline for what a *plausible* explanation should highlight.

# %%
from medxplain.data.clinical_lexicon import FACTORS_BY_KEY

rows = []
for key, factor in FACTORS_BY_KEY.items():
    present = df["factors"].map(lambda f, k=key: k in f)
    if present.sum() < 20:
        continue
    rows.append(
        {
            "factor": key,
            "prevalence": present.mean(),
            "rate_present": df.loc[present, LABEL].mean(),
            "rate_absent": df.loc[~present, LABEL].mean(),
            "true_weight": factor.mortality_weight,
        }
    )
factor_df = pd.DataFrame(rows)
factor_df["empirical_lift"] = factor_df["rate_present"] - factor_df["rate_absent"]
factor_df = factor_df.sort_values("true_weight", ascending=False)
print(factor_df.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

corr = factor_df["true_weight"].corr(factor_df["empirical_lift"], method="spearman")
print(f"\nSpearman(true weight, empirical lift) = {corr:.3f}")

# %% [markdown]
# ## 3. Does truncation cost us signal?
#
# The pilot reads only the first `max_length` tokens under a head+tail strategy.
# This measures how much of the risk-factor content survives that window.

# %%
from transformers import AutoTokenizer

from medxplain.models.transformer import MODEL_REGISTRY

tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY["clinicalbert"])
max_length = int(cfg.get_path("model.max_length", 128))

sample = df.sample(n=min(300, len(df)), random_state=0)
covered, total = 0, 0
for _, note in sample.iterrows():
    enc = tokenizer(note["text"], add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    body = max_length - 2
    head_n = int(body * float(cfg.get_path("model.head_fraction", 0.35)))
    kept = offsets[:head_n] + offsets[-(body - head_n):] if len(offsets) > body else offsets
    kept_chars = {c for s, e in kept for c in range(s, e)}
    for span in note["gold_spans"]:
        total += 1
        span_chars = set(range(span["start"], span["end"]))
        if len(span_chars & kept_chars) / max(len(span_chars), 1) > 0.5:
            covered += 1

print(f"risk-factor sentences at least half inside the {max_length}-token window: "
      f"{covered}/{total} = {covered / max(total, 1):.1%}")
print("Raising model.max_length is the single most effective change on a GPU.")

# %% [markdown]
# ## 4. Error analysis
#
# Needs `scripts/03_evaluate.py` to have run.

# %%
evaluation_path = RESULTS / "metrics" / "evaluation.json"
if not evaluation_path.exists():
    print("run scripts/03_evaluate.py first - skipping the error analysis")
else:
    evaluation = read_json(evaluation_path)
    best = evaluation["best_model"]["backbone"]
    print(f"best model (selected on validation): {best}")

    preds = sorted((RESULTS / "predictions").glob(f"{best}_seed*.npz"))
    stacked = np.vstack([np.load(p)["test_probabilities"] for p in preds])
    probs = stacked.mean(axis=0)

    test = df[df["split"] == "test"].copy().reset_index(drop=True)
    test["probability"] = probs
    threshold = float(np.load(preds[0])["threshold"])
    test["predicted"] = (test["probability"] >= threshold).astype(int)
    test["error"] = np.where(
        (test["predicted"] == 1) & (test[LABEL] == 0), "false positive",
        np.where((test["predicted"] == 0) & (test[LABEL] == 1), "false negative", "correct"),
    )
    print(test["error"].value_counts(), "\n")

    # Do errors concentrate in particular clinical presentations?
    exploded = test.explode("factors")
    rate = (
        exploded.groupby("factors")["error"]
        .apply(lambda s: (s != "correct").mean())
        .sort_values(ascending=False)
    )
    counts = exploded["factors"].value_counts()
    summary = pd.DataFrame({"error_rate": rate, "n": counts}).query("n >= 20")
    print("Error rate by risk factor present:")
    print(summary.sort_values("error_rate", ascending=False).head(12).to_string())

    # The most confident mistakes are the ones worth reading by hand.
    worst_fn = test[test["error"] == "false negative"].nsmallest(3, "probability")
    worst_fp = test[test["error"] == "false positive"].nlargest(3, "probability")
    print("\nMost confident false negatives (model said low risk, patient died):")
    for _, row in worst_fn.iterrows():
        print(f"  p={row['probability']:.3f} factors={row['factors']}")
    print("\nMost confident false positives:")
    for _, row in worst_fp.iterrows():
        print(f"  p={row['probability']:.3f} factors={row['factors']}")

# %% [markdown]
# ## 5. Explanations on the failures
#
# The point of an explainable system: when it is wrong, can you see *why*?

# %%
checkpoint = RESULTS / "checkpoints" / (evaluation["best_model"]["backbone"] if evaluation_path.exists() else "clinicalbert")
if not (checkpoint / "medxplain_config.json").exists():
    print("no checkpoint - run scripts/02_train_transformers.py first")
else:
    import torch

    from medxplain.explain import build_explainer
    from medxplain.models.transformer import ClinicalRiskModel

    model, tok = ClinicalRiskModel.load(checkpoint, device="cpu")
    explainer = build_explainer("occlusion", model, tok, device="cpu",
                                max_length=model.cfg.max_length, max_words=120)

    target = worst_fn.iloc[0] if "worst_fn" in dir() and len(worst_fn) else test.iloc[0]
    explanation = explainer.explain(target["text"])
    print(f"note {target['note_id']} | p={explanation.probability:.3f} | true={target[LABEL]}")
    print("\nwords pushing risk UP:")
    for w in explanation.top_words(8, signed=True):
        print(f"  {w.score:+.3f}  {w.text}")
    print("\nwords pushing risk DOWN:")
    for w in explanation.top_negative_words(8):
        print(f"  {w.score:+.3f}  {w.text}")

    gold = {s["factor"] for s in target["gold_spans"]}
    print(f"\ntrue risk factors in this note: {sorted(gold)}")
    print(
        "If the model's top words do not overlap the true factors, the failure is a "
        "representation problem, not a threshold problem."
    )
