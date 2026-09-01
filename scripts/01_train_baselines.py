"""Phase 2 - classical baselines (TF-IDF + LR / RandomForest / XGBoost).

Each baseline is tuned on the *same* validation split the transformers use for
early stopping, so model selection is not a confound in the comparison. Test
predictions are written to disk so the significance tests in Phase 5 can pair
them against the transformer predictions patient-by-patient.

Run:
    python scripts/01_train_baselines.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from medxplain.config import load_config, parse_overrides
from medxplain.data.subsample import subsample_train
from medxplain.eval.metrics import classification_metrics, curve_points
from medxplain.models.baselines import BaselineConfig, TfidfBaseline
from medxplain.utils import get_logger, set_seed, timer, write_json

LOGGER = get_logger("medxplain.scripts.baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    label = cfg.get_path("experiment.primary_outcome", "mortality")
    seeds = list(cfg.get_path("experiment.seeds", [42]))

    df = pd.read_parquet(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    # Same subsample as the transformers: the comparison is only meaningful if
    # every model family is fitted on exactly the same notes.
    train = subsample_train(
        df[df["split"] == "train"],
        cfg.get_path("train.max_train_notes"),
        label_col=label,
        seed=int(cfg.get_path("split.seed", 42)),
    )
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    LOGGER.info("train=%d val=%d test=%d | outcome=%s", len(train), len(val), len(test), label)

    kinds = list(cfg.get_path("baselines.kinds", ["logreg"]))
    tune = bool(cfg.get_path("baselines.tune", True))
    calibrate = bool(cfg.get_path("baselines.calibrate", True))

    all_results: dict = {"outcome": label, "n_train": int(len(train)), "models": {}}
    predictions: dict = {}

    for kind in kinds:
        per_seed: list[dict] = []
        for seed in seeds:
            set_seed(seed)
            started = time.perf_counter()
            base_cfg = BaselineConfig(
                kind=kind,
                seed=seed,
                max_features=int(cfg.get_path("baselines.max_features", 60000)),
            )
            model = TfidfBaseline(base_cfg)

            if tune:
                LOGGER.info("tuning %s (seed %d) on the validation split", kind, seed)
                best = model.tune(
                    train["text"].tolist(), train[label].tolist(),
                    val["text"].tolist(), val[label].tolist(),
                )
                LOGGER.info("  best params %s (val AUROC %.4f)", best["params"], best["auroc"])
            else:
                model.fit(train["text"].tolist(), train[label].tolist())

            # Threshold is selected on validation, then applied unchanged to test.
            # Choosing it on test would inflate F1/precision/recall.
            val_probs = model.predict_proba(val["text"].tolist())
            from medxplain.eval.metrics import best_f1_threshold

            threshold = best_f1_threshold(val[label].to_numpy(), val_probs)

            if calibrate:
                model.calibrate(val["text"].tolist(), val[label].tolist())
                val_probs = model.predict_proba(val["text"].tolist())
                threshold = best_f1_threshold(val[label].to_numpy(), val_probs)

            test_probs = model.predict_proba(test["text"].tolist())
            metrics = classification_metrics(
                test[label].to_numpy(), test_probs, threshold=threshold
            )
            metrics["seed"] = seed
            metrics["train_seconds"] = time.perf_counter() - started
            metrics["params"] = model.cfg.grid
            metrics["calibrated"] = model.calibrated
            per_seed.append(metrics)

            LOGGER.info(
                "%s seed=%d | test AUROC %.4f AUPRC %.4f F1 %.4f (%.0fs)",
                kind, seed, metrics["auroc"], metrics["auprc"], metrics["f1"],
                metrics["train_seconds"],
            )

            # keep the first seed's artefacts: model file, predictions, top features
            if seed == seeds[0]:
                model.save(results_dir / "checkpoints" / f"baseline_{kind}.joblib")
                predictions[kind] = test_probs
                all_results.setdefault("top_features", {})[kind] = model.top_features(20)
                all_results.setdefault("curves", {})[kind] = curve_points(
                    test[label].to_numpy(), test_probs
                )

        from medxplain.eval.metrics import aggregate_seeds

        all_results["models"][kind] = {
            "per_seed": per_seed,
            "aggregate": aggregate_seeds(per_seed),
        }

    write_json(results_dir / "metrics" / "baselines.json", all_results)
    np.savez(
        results_dir / "metrics" / "baseline_predictions.npz",
        y_true=test[label].to_numpy(),
        note_ids=test["note_id"].to_numpy(),
        **predictions,
    )

    LOGGER.info("--- baseline summary (test, mean +/- std over %d seeds) ---", len(seeds))
    for kind, block in all_results["models"].items():
        agg = block["aggregate"]
        LOGGER.info(
            "  %-14s AUROC %.4f+/-%.4f  AUPRC %.4f+/-%.4f  F1 %.4f",
            kind, agg["auroc"]["mean"], agg["auroc"]["std"],
            agg["auprc"]["mean"], agg["auprc"]["std"], agg["f1"]["mean"],
        )


if __name__ == "__main__":
    main()
