"""Training-set subsampling for compute-constrained runs.

Fine-tuning four BERT backbones across several seeds is hours of work on a CPU.
Rather than quietly training different models on different amounts of data — or
letting one stage silently use more than another — the subsample is an explicit
config value applied by *every* training script through this one function.

Two invariants make a subsampled run still comparable:

* the subsample is **stratified** on the outcome, so the positive rate the model
  sees is the cohort's positive rate, not a lucky draw;
* it is **deterministic given the split seed**, so the transformer and the TF-IDF
  baseline are fitted on exactly the same notes and the paired significance
  tests remain valid.

Validation and test are never subsampled: shrinking them would widen every
confidence interval reported in the paper for no compute saving worth having.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.data.subsample")


def subsample_train(
    train: pd.DataFrame,
    max_notes: int | None,
    label_col: str = "mortality",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a stratified subsample of ``train`` with at most ``max_notes`` rows."""
    if not max_notes or max_notes >= len(train):
        return train

    rng = np.random.default_rng(seed)
    positives = train.index[train[label_col] == 1].to_numpy()
    negatives = train.index[train[label_col] == 0].to_numpy()

    # preserve the observed prevalence rather than rebalancing: class imbalance is
    # part of the problem being studied and is handled in the loss, not the sample
    rate = len(positives) / max(len(train), 1)
    n_pos = min(len(positives), max(1, int(round(max_notes * rate))))
    n_neg = min(len(negatives), max_notes - n_pos)

    chosen = np.concatenate(
        [rng.choice(positives, n_pos, replace=False), rng.choice(negatives, n_neg, replace=False)]
    )
    rng.shuffle(chosen)
    out = train.loc[chosen]

    LOGGER.info(
        "training subsample: %d of %d notes (positive rate %.3f -> %.3f)",
        len(out), len(train), train[label_col].mean(), out[label_col].mean(),
    )
    return out
