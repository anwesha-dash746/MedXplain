"""Train/validation/test splitting.

The test set is created once, written to disk, and never touched again except
for the single final evaluation pass. Splits are stratified on the primary
outcome so the rare-event base rate is preserved in every fold, and grouped by
patient when a patient identifier is available — a patient with two admissions
must not appear on both sides of the split, or the reported AUROC is inflated
by identity leakage rather than clinical signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15  # fraction of the *whole* dataset
    seed: int = 42
    label_col: str = "mortality"
    group_col: str | None = None  # e.g. 'subject_id' for MIMIC-III


def stratified_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> pd.DataFrame:
    """Return ``df`` with an added ``split`` column in {train, val, test}."""
    cfg = cfg or SplitConfig()
    y = df[cfg.label_col].to_numpy()

    if cfg.group_col and cfg.group_col in df.columns:
        split = _grouped_split(df, y, cfg)
    else:
        idx = np.arange(len(df))
        train_val_idx, test_idx = train_test_split(
            idx, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
        )
        val_share = cfg.val_size / (1.0 - cfg.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share,
            stratify=y[train_val_idx],
            random_state=cfg.seed,
        )
        split = np.empty(len(df), dtype=object)
        split[train_idx] = "train"
        split[val_idx] = "val"
        split[test_idx] = "test"

    out = df.copy()
    out["split"] = split
    return out


def _grouped_split(df: pd.DataFrame, y: np.ndarray, cfg: SplitConfig) -> np.ndarray:
    """Patient-disjoint split, still stratified on the outcome."""
    groups = df[cfg.group_col].to_numpy()
    n_test_folds = max(2, int(round(1 / cfg.test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=cfg.seed)
    train_val_idx, test_idx = next(sgkf.split(df, y, groups))

    inner_share = cfg.val_size / (1.0 - cfg.test_size)
    n_val_folds = max(2, int(round(1 / inner_share)))
    inner = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=cfg.seed)
    sub = df.iloc[train_val_idx]
    tr_rel, val_rel = next(inner.split(sub, y[train_val_idx], groups[train_val_idx]))

    split = np.empty(len(df), dtype=object)
    split[test_idx] = "test"
    split[train_val_idx[tr_rel]] = "train"
    split[train_val_idx[val_rel]] = "val"
    return split


def split_report(df: pd.DataFrame, label_cols: tuple[str, ...] = ("mortality",)) -> dict:
    """Per-split size and positive rate, written to ``results/metrics``."""
    report: dict = {}
    for name, group in df.groupby("split"):
        entry = {"n": int(len(group)), "chars_mean": float(group["text"].str.len().mean())}
        for col in label_cols:
            if col in group.columns:
                entry[f"{col}_positive"] = int(group[col].sum())
                entry[f"{col}_rate"] = float(group[col].mean())
        report[str(name)] = entry
    return report


def check_leakage(df: pd.DataFrame, group_col: str | None = None) -> dict:
    """Assert the split is honest: no exact-duplicate text and no shared groups.

    Duplicate discharge summaries do occur in real extracts (amended notes), and
    a duplicate straddling train/test is the single most common way a clinical
    NLP result turns out to be inflated.
    """
    issues: dict = {"duplicate_text_across_splits": 0, "shared_groups": 0}
    dup = df[df.duplicated("text", keep=False)]
    if len(dup):
        per_text = dup.groupby("text")["split"].nunique()
        issues["duplicate_text_across_splits"] = int((per_text > 1).sum())
    if group_col and group_col in df.columns:
        per_group = df.groupby(group_col)["split"].nunique()
        issues["shared_groups"] = int((per_group > 1).sum())
    issues["ok"] = issues["duplicate_text_across_splits"] == 0 and issues["shared_groups"] == 0
    return issues
