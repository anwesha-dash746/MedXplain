"""Phase 5c - explanation stability under meaning-preserving edits.

For each method, the same note is explained three times: once as written, once
after abbreviation/synonym paraphrasing, and once with an irrelevant sentence
inserted. Agreement between the original and perturbed explanations is the
stability score.

Run:
    python scripts/05_stability.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from medxplain.config import load_config, parse_overrides
from medxplain.eval.stability import StabilityConfig, StabilityEvaluator, summarise_stability
from medxplain.explain import build_explainer
from medxplain.models.transformer import ClinicalRiskModel
from medxplain.utils import get_logger, read_json, set_seed, write_json

LOGGER = get_logger("medxplain.scripts.stability")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    device = "cuda" if torch.cuda.is_available() and cfg.get_path("experiment.device") != "cpu" else "cpu"
    seed = int(cfg.get_path("split.seed", 42))
    set_seed(seed)

    backbone = args.model
    if backbone is None:
        evaluation = results_dir / "metrics" / "evaluation.json"
        backbone = (
            read_json(evaluation).get("best_model", {}).get("backbone")
            if evaluation.exists()
            else None
        ) or cfg.get_path("serve.default_model", "clinicalbert")

    checkpoint = results_dir / "checkpoints" / backbone
    if not (checkpoint / "medxplain_config.json").exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    model, tokenizer = ClinicalRiskModel.load(checkpoint, device=device)
    max_length = model.cfg.max_length

    df = pd.read_parquet(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    test = df[df["split"] == "test"]
    n_notes = int(cfg.get_path("evaluation.stability_n_notes", 30))
    texts = test.sample(n=min(n_notes, len(test)), random_state=seed)["text"].tolist()
    LOGGER.info("stability on %d notes with model %s", len(texts), backbone)

    stability_cfg = StabilityConfig(
        top_k=10, perturbations=("near_duplicate", "paraphrase", "distractor"), seed=seed
    )

    output: dict = {
        "model": backbone,
        "n_notes": len(texts),
        "perturbations": list(stability_cfg.perturbations),
        "methods": {},
        "per_note": {},
    }

    for method in cfg.get_path("explain.methods", ["attention_rollout", "shap"]):
        LOGGER.info("--- %s ---", method)
        kwargs: dict = {"max_length": max_length}
        if method == "shap":
            kwargs["max_evals"] = int(cfg.get_path("explain.shap_max_evals", 200))
        elif method == "integrated_gradients":
            kwargs["n_steps"] = int(cfg.get_path("explain.ig_steps", 16))
        elif method == "occlusion":
            kwargs["max_words"] = int(cfg.get_path("explain.occlusion_max_words", 200))

        explainer = build_explainer(method, model, tokenizer, device=device, **kwargs)
        needs_grad = method in ("integrated_gradients", "attention_grad")

        def explain_fn(text: str, _explainer=explainer, _grad=needs_grad):
            with torch.set_grad_enabled(_grad):
                return _explainer.explain(text)

        evaluator = StabilityEvaluator(explain_fn, stability_cfg)
        rows = []
        for i, text in enumerate(texts):
            try:
                rows.append(evaluator.evaluate_note(text, i))
            except Exception as exc:
                LOGGER.warning("  note %d failed: %s", i, exc)
            if (i + 1) % 5 == 0:
                LOGGER.info("  %d/%d notes", i + 1, len(texts))

        summary = summarise_stability(rows, stability_cfg.perturbations)
        output["methods"][method] = summary
        output["per_note"][method] = rows

        para = summary.get("paraphrase", {})
        if "spearman" in para:
            LOGGER.info(
                "  paraphrase: spearman %.3f | top-10 Jaccard %.3f | distractor mass %.3f",
                para["spearman"]["mean"],
                para.get("top_k_jaccard", {}).get("mean", float("nan")),
                summary.get("distractor", {}).get("distractor_mass", {}).get("mean", float("nan")),
            )
        write_json(results_dir / "metrics" / "stability.json", output)

    LOGGER.info("wrote %s", results_dir / "metrics" / "stability.json")


if __name__ == "__main__":
    main()
