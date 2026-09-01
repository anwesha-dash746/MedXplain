"""Phase 5a - predictive evaluation: metrics, calibration and significance tests.

Reads the test predictions written by scripts 01 and 02, then produces the
paper's main results table and the statistical comparison of the best
transformer against every baseline.

Nothing is retrained here. The script only consumes saved predictions, so it is
cheap to rerun and cannot accidentally change a reported number by retraining
with a different seed.

Run:
    python scripts/03_evaluate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from medxplain.config import load_config, parse_overrides
from medxplain.eval.metrics import (
    aggregate_seeds,
    bootstrap_ci,
    classification_metrics,
    curve_points,
    reliability_curve,
)
from medxplain.eval.significance import compare_against_baselines, seed_level_test
from medxplain.utils import get_logger, read_json, write_json

LOGGER = get_logger("medxplain.scripts.evaluate")


def load_transformer_predictions(results_dir: Path) -> dict[str, dict[int, dict]]:
    """{backbone: {seed: {test_probabilities, y_test, ...}}} from results/predictions."""
    out: dict[str, dict[int, dict]] = {}
    pred_dir = results_dir / "predictions"
    if not pred_dir.exists():
        return out
    for path in sorted(pred_dir.glob("*_seed*.npz")):
        stem = path.stem
        backbone, _, seed_part = stem.rpartition("_seed")
        blob = np.load(path, allow_pickle=True)
        out.setdefault(backbone, {})[int(seed_part)] = {
            key: blob[key] for key in blob.files
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    metric = cfg.get_path("evaluation.significance_metric", "auprc")
    n_bootstrap = int(cfg.get_path("evaluation.n_bootstrap", 1000))

    transformer_preds = load_transformer_predictions(results_dir)
    if not transformer_preds:
        raise SystemExit(
            "no transformer predictions found - run scripts/02_train_transformers.py first"
        )

    baseline_path = results_dir / "metrics" / "baseline_predictions.npz"
    baseline_blob = np.load(baseline_path, allow_pickle=True) if baseline_path.exists() else None
    if baseline_blob is None:
        LOGGER.warning("no baseline predictions; significance testing will be skipped")

    # y_true is identical across every prediction file by construction, but assert
    # it rather than assume: a silent split mismatch would invalidate every test.
    any_backbone = next(iter(transformer_preds.values()))
    y_true = np.asarray(next(iter(any_backbone.values()))["y_test"]).astype(int)
    for backbone, seeds in transformer_preds.items():
        for seed, blob in seeds.items():
            if not np.array_equal(np.asarray(blob["y_test"]).astype(int), y_true):
                raise SystemExit(f"{backbone} seed {seed} was evaluated on a different test split")
    if baseline_blob is not None and not np.array_equal(
        np.asarray(baseline_blob["y_true"]).astype(int), y_true
    ):
        raise SystemExit("baseline predictions use a different test split than the transformers")

    LOGGER.info("test set: n=%d, positives=%d (%.1f%%)", y_true.size, y_true.sum(), 100 * y_true.mean())

    evaluation: dict = {
        "outcome": cfg.get_path("experiment.primary_outcome", "mortality"),
        "n_test": int(y_true.size),
        "prevalence": float(y_true.mean()),
        "metric": metric,
        "models": {},
    }

    # ---- per-model metrics, per seed and aggregated ---------------------------------
    per_model_metric_values: dict[str, list[float]] = {}
    mean_probs: dict[str, np.ndarray] = {}

    for backbone, seeds in sorted(transformer_preds.items()):
        rows = []
        for seed in sorted(seeds):
            blob = seeds[seed]
            probs = np.asarray(blob["test_probabilities"], dtype=float)
            threshold = float(np.asarray(blob["threshold"]))
            row = classification_metrics(y_true, probs, threshold=threshold)
            row["seed"] = seed
            rows.append(row)
        aggregate = aggregate_seeds(rows)
        # The ensemble over seeds is reported separately: averaging seed
        # probabilities is a legitimate system, but it is NOT the same thing as
        # the mean of the per-seed scores and must not be conflated with it.
        stacked = np.vstack([np.asarray(seeds[s]["test_probabilities"], float) for s in sorted(seeds)])
        ensemble = stacked.mean(axis=0)
        mean_probs[backbone] = ensemble

        evaluation["models"][backbone] = {
            "family": "transformer",
            "per_seed": rows,
            "aggregate": aggregate,
            "seed_ensemble": classification_metrics(y_true, ensemble),
            "calibration": reliability_curve(y_true, ensemble),
            "curves": curve_points(y_true, ensemble),
        }
        per_model_metric_values[backbone] = [r[metric] for r in rows]
        LOGGER.info(
            "%-13s AUROC %.4f+/-%.4f  AUPRC %.4f+/-%.4f  (%d seeds)",
            backbone,
            aggregate["auroc"]["mean"], aggregate["auroc"]["std"],
            aggregate["auprc"]["mean"], aggregate["auprc"]["std"],
            aggregate["auroc"]["n_seeds"],
        )

    baseline_probs: dict[str, np.ndarray] = {}
    if baseline_blob is not None:
        for key in baseline_blob.files:
            if key in ("y_true", "note_ids"):
                continue
            probs = np.asarray(baseline_blob[key], dtype=float)
            baseline_probs[key] = probs
            evaluation["models"][key] = {
                "family": "baseline",
                "test": classification_metrics(y_true, probs, n_bootstrap=n_bootstrap),
                "calibration": reliability_curve(y_true, probs),
                "curves": curve_points(y_true, probs),
            }
            LOGGER.info("%-13s AUROC %.4f  AUPRC %.4f", key,
                        evaluation["models"][key]["test"]["auroc"],
                        evaluation["models"][key]["test"]["auprc"])

    # ---- pick the best model on VALIDATION, not test --------------------------------
    # Choosing the headline model by test performance is a textbook way to
    # over-report; the winner is selected on validation and then evaluated once.
    best_backbone, best_val = None, -np.inf
    for backbone, seeds in transformer_preds.items():
        from sklearn.metrics import average_precision_score

        val_scores = []
        for blob in seeds.values():
            if "val_probabilities" in blob and "y_val" in blob:
                val_scores.append(
                    average_precision_score(np.asarray(blob["y_val"]).astype(int),
                                            np.asarray(blob["val_probabilities"], float))
                )
        score = float(np.mean(val_scores)) if val_scores else -np.inf
        evaluation["models"][backbone]["val_auprc_mean"] = score
        if score > best_val:
            best_backbone, best_val = backbone, score

    evaluation["best_model"] = {
        "backbone": best_backbone,
        "selected_on": "mean validation AUPRC across seeds",
        "val_auprc": best_val,
    }
    LOGGER.info("best model by validation AUPRC: %s (%.4f)", best_backbone, best_val)

    # ---- significance ---------------------------------------------------------------
    if best_backbone and baseline_probs:
        proposed = mean_probs[best_backbone]
        evaluation["significance"] = compare_against_baselines(
            y_true, proposed, baseline_probs, metric=metric, n_bootstrap=n_bootstrap
        )
        LOGGER.info("--- %s vs. baselines (%s, Holm-adjusted) ---", best_backbone, metric)
        for name, block in evaluation["significance"]["holm"]["results"].items():
            LOGGER.info(
                "  vs %-14s p=%.4f (adj %.4f) %s",
                name, block["p_raw"], block["p_adjusted"],
                "significant" if block["significant"] else "n.s.",
            )

    # domain adaptation: each clinical model against the plain-BERT control
    if "bert" in per_model_metric_values:
        control = per_model_metric_values["bert"]
        evaluation["domain_adaptation"] = {}
        for backbone, values in per_model_metric_values.items():
            if backbone == "bert":
                continue
            evaluation["domain_adaptation"][backbone] = seed_level_test(values, control)
        LOGGER.info("--- domain adaptation vs. BERT-base (seed-level, %s) ---", metric)
        for backbone, test in evaluation["domain_adaptation"].items():
            LOGGER.info(
                "  %-13s delta=%+.4f  p=%.4f  d=%.2f",
                backbone, test["difference"], test.get("p_value", float("nan")),
                test.get("cohens_d", float("nan")),
            )

    # bootstrap CI for the headline model
    if best_backbone:
        evaluation["best_model"]["ci"] = bootstrap_ci(
            y_true, mean_probs[best_backbone], n_bootstrap=n_bootstrap
        )

    write_json(results_dir / "metrics" / "evaluation.json", evaluation)
    LOGGER.info("wrote %s", results_dir / "metrics" / "evaluation.json")


if __name__ == "__main__":
    main()
