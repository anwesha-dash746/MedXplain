"""Phase 5b - explanation faithfulness (ERASER comprehensiveness / sufficiency).

Runs every explanation method over a random subsample of the test set and
measures how much the model's own confidence depends on the words each method
highlighted, together with deletion/insertion curves and a random-explanation
control.

This is the expensive script. Perturbation methods need hundreds of forward
passes per note, and the faithfulness metrics themselves add roughly
``2 * len(k_fractions) * (1 + n_random_controls)`` more, so the subsample size is
a config value and the per-method cost is recorded in the output.

Run:
    python scripts/04_faithfulness.py
    python scripts/04_faithfulness.py --set evaluation.faithfulness_n_notes=10
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
from medxplain.eval.faithfulness import (
    FaithfulnessConfig,
    FaithfulnessEvaluator,
    summarise_curves,
    summarise_faithfulness,
)
from medxplain.explain import build_explainer
from medxplain.models.transformer import ClinicalRiskModel
from medxplain.utils import get_logger, read_json, set_seed, write_json

LOGGER = get_logger("medxplain.scripts.faithfulness")


def pick_model(cfg, results_dir: Path) -> str:
    """Use the model chosen on validation by script 03, else the config default."""
    evaluation = results_dir / "metrics" / "evaluation.json"
    if evaluation.exists():
        try:
            best = read_json(evaluation).get("best_model", {}).get("backbone")
            if best and (results_dir / "checkpoints" / best).exists():
                return best
        except Exception:
            pass
    return cfg.get_path("serve.default_model", "clinicalbert")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--model", default=None, help="backbone key to explain")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    device = "cuda" if torch.cuda.is_available() and cfg.get_path("experiment.device") != "cpu" else "cpu"
    seed = int(cfg.get_path("split.seed", 42))
    set_seed(seed)

    backbone = args.model or pick_model(cfg, results_dir)
    checkpoint = results_dir / "checkpoints" / backbone
    if not (checkpoint / "medxplain_config.json").exists():
        raise SystemExit(f"no checkpoint at {checkpoint} - run scripts/02_train_transformers.py")

    LOGGER.info("loading %s from %s", backbone, checkpoint)
    model, tokenizer = ClinicalRiskModel.load(checkpoint, device=device)
    max_length = model.cfg.max_length

    df = pd.read_parquet(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    test = df[df["split"] == "test"]
    n_notes = int(cfg.get_path("evaluation.faithfulness_n_notes", 40))
    sample = test.sample(n=min(n_notes, len(test)), random_state=seed)
    texts = sample["text"].tolist()
    LOGGER.info("evaluating faithfulness on %d test notes", len(texts))

    faith_cfg = FaithfulnessConfig(
        k_fractions=tuple(cfg.get_path("evaluation.faithfulness_k_fractions", [0.01, 0.05, 0.1, 0.2, 0.5])),
        removal=cfg.get_path("evaluation.faithfulness_removal", "mask"),
        max_length=max_length,
        seed=seed,
    )
    evaluator = FaithfulnessEvaluator(model, tokenizer, faith_cfg, device=device)

    methods = list(cfg.get_path("explain.methods", ["attention_rollout", "shap"]))
    output: dict = {
        "model": backbone,
        "checkpoint": str(checkpoint),
        "n_notes": len(texts),
        "config": faith_cfg.__dict__,
        "methods": {},
        "curves": {},
        "cost": {},
        "per_note": {},
    }

    for method in methods:
        LOGGER.info("--- %s ---", method)
        kwargs: dict = {"max_length": max_length}
        if method == "shap":
            kwargs["max_evals"] = int(cfg.get_path("explain.shap_max_evals", 200))
        elif method == "integrated_gradients":
            kwargs["n_steps"] = int(cfg.get_path("explain.ig_steps", 16))
        elif method == "occlusion":
            kwargs["max_words"] = int(cfg.get_path("explain.occlusion_max_words", 200))

        explainer = build_explainer(method, model, tokenizer, device=device, **kwargs)

        rows, curves, explain_seconds, evaluations = [], [], [], []
        for i, text in enumerate(texts):
            try:
                started = time.perf_counter()
                needs_grad = method in ("integrated_gradients", "attention_grad")
                with torch.set_grad_enabled(needs_grad):
                    explanation = explainer.explain(text)
                explain_seconds.append(time.perf_counter() - started)
                evaluations.append(int(explanation.meta.get("n_model_evaluations", 1)))

                rows.append(evaluator.evaluate(text, explanation))
                curves.append(evaluator.deletion_insertion_curves(text, explanation))
            except Exception as exc:
                LOGGER.warning("  note %d failed: %s", i, exc)
            if (i + 1) % 5 == 0:
                LOGGER.info("  %d/%d notes", i + 1, len(texts))

        summary = summarise_faithfulness(rows)
        output["methods"][method] = summary
        output["curves"][method] = summarise_curves(curves)
        output["cost"][method] = {
            "mean_explain_seconds": float(np.mean(explain_seconds)) if explain_seconds else None,
            "mean_model_evaluations": float(np.mean(evaluations)) if evaluations else None,
        }
        output["methods"][method]["n_model_evaluations"] = (
            int(np.mean(evaluations)) if evaluations else None
        )
        output["per_note"][method] = rows

        if "comprehensiveness" in summary:
            LOGGER.info(
                "  comprehensiveness %.4f+/-%.4f | sufficiency %.4f+/-%.4f | "
                "over-random %+.4f | %.1fs/note",
                summary["comprehensiveness"]["mean"], summary["comprehensiveness"]["std"],
                summary["sufficiency"]["mean"], summary["sufficiency"]["std"],
                summary["comprehensiveness_over_random"]["mean"],
                output["cost"][method]["mean_explain_seconds"] or 0.0,
            )
        write_json(results_dir / "metrics" / "faithfulness.json", output)

    LOGGER.info("--- faithfulness summary (%s, %d notes) ---", backbone, len(texts))
    ranked = sorted(
        output["methods"].items(),
        key=lambda kv: kv[1].get("comprehensiveness", {}).get("mean", -np.inf),
        reverse=True,
    )
    for method, block in ranked:
        if "comprehensiveness" in block:
            LOGGER.info(
                "  %-22s comp %.4f  suff %.4f  vs-random %+.4f",
                method, block["comprehensiveness"]["mean"], block["sufficiency"]["mean"],
                block["comprehensiveness_over_random"]["mean"],
            )


if __name__ == "__main__":
    main()
