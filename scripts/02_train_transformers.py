"""Phase 3 - fine-tune the domain-adapted transformers, multi-seed.

Trains every backbone in ``model.backbones`` across every seed in
``experiment.seeds`` and writes, per run: the training history, validation and
test predictions, and test metrics at a threshold selected on validation. The
best seed per backbone (by validation AUPRC) is checkpointed for serving; the
others are discarded to keep the repository small, but their predictions are
kept so the multi-seed table and the significance tests are fully reproducible.

Run:
    python scripts/02_train_transformers.py
    python scripts/02_train_transformers.py --set model.backbones='[clinicalbert]'
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from medxplain.config import load_config, parse_overrides
from medxplain.data.subsample import subsample_train
from medxplain.eval.metrics import aggregate_seeds, best_f1_threshold, classification_metrics, curve_points
from medxplain.models.transformer import ModelConfig
from medxplain.train.trainer import TrainConfig, Trainer
from medxplain.utils import env_report, get_logger, read_json, set_seed, write_json

LOGGER = get_logger("medxplain.scripts.transformers")


def build_configs(cfg, backbone: str, seed: int) -> tuple[ModelConfig, TrainConfig]:
    model_cfg = ModelConfig(
        name=backbone,
        max_length=int(cfg.get_path("model.max_length", 256)),
        pooling=cfg.get_path("model.pooling", "cls"),
        dropout=float(cfg.get_path("model.dropout", 0.1)),
        truncation_strategy=cfg.get_path("model.truncation_strategy", "head_tail"),
        head_fraction=float(cfg.get_path("model.head_fraction", 0.35)),
        freeze_layers=int(cfg.get_path("model.freeze_layers", 0)),
    )
    train_cfg = TrainConfig(
        learning_rate=float(cfg.get_path("train.learning_rate", 2e-5)),
        head_learning_rate=cfg.get_path("train.head_learning_rate"),
        weight_decay=float(cfg.get_path("train.weight_decay", 0.01)),
        batch_size=int(cfg.get_path("train.batch_size", 16)),
        eval_batch_size=int(cfg.get_path("train.eval_batch_size", 32)),
        epochs=int(cfg.get_path("train.epochs", 3)),
        warmup_ratio=float(cfg.get_path("train.warmup_ratio", 0.1)),
        loss=cfg.get_path("train.loss", "weighted_ce"),
        class_weight_scheme=cfg.get_path("train.class_weight_scheme", "balanced"),
        focal_gamma=float(cfg.get_path("train.focal_gamma", 2.0)),
        metric_for_best=cfg.get_path("train.metric_for_best", "auprc"),
        patience=int(cfg.get_path("train.patience", 1)),
        log_every=int(cfg.get_path("train.log_every", 25)),
        amp=bool(cfg.get_path("train.amp", False)),
        device=cfg.get_path("experiment.device", "auto"),
        seed=seed,
        max_train_steps=cfg.get_path("train.max_train_steps"),
    )
    return model_cfg, train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume", action="store_true",
                        help="skip (backbone, seed) runs already present in the results file")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    metrics_path = results_dir / "metrics" / "transformers.json"
    label = cfg.get_path("experiment.primary_outcome", "mortality")
    seeds = list(cfg.get_path("experiment.seeds", [42]))
    backbones = list(cfg.get_path("model.backbones", ["clinicalbert"]))

    df = pd.read_parquet(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    train = subsample_train(
        df[df["split"] == "train"],
        cfg.get_path("train.max_train_notes"),
        label_col=label,
        seed=int(cfg.get_path("split.seed", 42)),
    )
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    LOGGER.info(
        "outcome=%s | train=%d val=%d test=%d | backbones=%s | seeds=%s",
        label, len(train), len(val), len(test), backbones, seeds,
    )

    results: dict = {
        "outcome": label,
        "environment": env_report(),
        "n_train": int(len(train)),
        "max_train_notes": cfg.get_path("train.max_train_notes"),
        "models": {},
    }
    if args.resume and metrics_path.exists():
        results = read_json(metrics_path)
        LOGGER.info("resuming: %d backbones already recorded", len(results.get("models", {})))

    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Seed is the OUTER loop deliberately: one full pass covers every backbone, so
    # an interrupted run still yields a complete (if single-seed) comparison table
    # rather than three seeds of one model and nothing for the others.
    for seed in seeds:
        for backbone in backbones:
            block = results["models"].setdefault(backbone, {"per_seed": []})
            done = {r["seed"] for r in block["per_seed"]}
            best_val = max((r["val_auprc"] for r in block["per_seed"]), default=-np.inf)

            if args.resume and seed in done:
                LOGGER.info("skipping %s seed=%d (already done)", backbone, seed)
                continue

            LOGGER.info("=" * 78)
            LOGGER.info("training %s | seed %d", backbone, seed)
            LOGGER.info("=" * 78)
            set_seed(seed)
            model_cfg, train_cfg = build_configs(cfg, backbone, seed)
            trainer = Trainer(model_cfg, train_cfg)

            started = time.perf_counter()
            result = trainer.train(
                train["text"].tolist(), train[label].tolist(),
                val["text"].tolist(), val[label].tolist(),
            )

            val_eval = trainer.evaluate(val["text"].tolist(), val[label].tolist())
            threshold = best_f1_threshold(val[label].to_numpy(), val_eval["probabilities"])
            test_eval = trainer.evaluate(test["text"].tolist(), test[label].tolist())
            test_probs = test_eval["probabilities"]

            metrics = classification_metrics(test[label].to_numpy(), test_probs, threshold=threshold)
            metrics.update(
                {
                    "seed": seed,
                    "backbone": backbone,
                    "hf_id": model_cfg.hf_id,
                    "val_auprc": float(val_eval["auprc"]),
                    "val_auroc": float(val_eval["auroc"]),
                    "best_epoch": result.best_epoch,
                    "train_seconds": time.perf_counter() - started,
                    "history": result.history,
                    "n_parameters": sum(p.numel() for p in trainer.model.parameters()),
                    "n_trainable": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                }
            )
            block["per_seed"].append(metrics)

            np.savez(
                pred_dir / f"{backbone}_seed{seed}.npz",
                test_probabilities=test_probs,
                val_probabilities=val_eval["probabilities"],
                y_test=test[label].to_numpy(),
                y_val=val[label].to_numpy(),
                note_ids=test["note_id"].to_numpy(),
                threshold=threshold,
            )

            LOGGER.info(
                "%s seed=%d | val AUPRC %.4f | test AUROC %.4f AUPRC %.4f F1 %.4f | %.0fs",
                backbone, seed, metrics["val_auprc"], metrics["auroc"], metrics["auprc"],
                metrics["f1"], metrics["train_seconds"],
            )

            # Checkpoint only the best seed per backbone: the API serves one model
            # per backbone, and keeping every seed would add gigabytes to the repo.
            if metrics["val_auprc"] > best_val:
                best_val = metrics["val_auprc"]
                ckpt = results_dir / "checkpoints" / backbone
                trainer.model.save(ckpt)
                trainer.tokenizer.save_pretrained(ckpt)
                write_json(
                    ckpt / "run_info.json",
                    {
                        "backbone": backbone,
                        "seed": seed,
                        "val_auprc": metrics["val_auprc"],
                        "test_auroc": metrics["auroc"],
                        "test_auprc": metrics["auprc"],
                        "threshold": float(threshold),
                        "outcome": label,
                        "max_length": model_cfg.max_length,
                        "display_name": model_cfg.display_name,
                    },
                )
                block["best_seed"] = seed
                block["checkpoint"] = str(ckpt)
                LOGGER.info("  checkpointed as the best seed for %s", backbone)
                block["curves"] = curve_points(test[label].to_numpy(), test_probs)

            del trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            block["aggregate"] = aggregate_seeds(block["per_seed"])
            write_json(metrics_path, results)  # checkpoint results after every run

    LOGGER.info("--- transformer summary (test, mean +/- std) ---")
    for backbone, block in results["models"].items():
        agg = block.get("aggregate", {})
        if "auroc" in agg:
            LOGGER.info(
                "  %-13s AUROC %.4f+/-%.4f  AUPRC %.4f+/-%.4f  (%d seeds)",
                backbone, agg["auroc"]["mean"], agg["auroc"]["std"],
                agg["auprc"]["mean"], agg["auprc"]["std"], agg["auroc"]["n_seeds"],
            )


if __name__ == "__main__":
    main()
