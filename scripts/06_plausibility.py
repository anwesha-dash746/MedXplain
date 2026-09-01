"""Phase 5d - explanation plausibility.

Two outputs:

1.  **Automatic** (synthetic corpus only): explanations are scored against the
    gold rationale spans the generator recorded, giving token-F1, rationale
    AUPRC, correlation with the true log-odds weights, and sign agreement.
2.  **Human review packet**: a blinded, shuffled form for a small clinician
    study, written to ``results/human_review/``. Raters fill in the CSV; running
    this script with ``--score`` aggregates their responses, including
    Krippendorff's alpha.

Run:
    python scripts/06_plausibility.py
    python scripts/06_plausibility.py --score results/human_review/responses.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from medxplain.config import load_config, parse_overrides
from medxplain.eval.plausibility import (
    build_review_packet,
    rationale_metrics,
    score_review,
    summarise_rationale_metrics,
)
from medxplain.explain import build_explainer
from medxplain.models.transformer import ClinicalRiskModel
from medxplain.utils import get_logger, read_json, set_seed, write_json

LOGGER = get_logger("medxplain.scripts.plausibility")


def score_responses(path: Path, results_dir: Path) -> None:
    """Aggregate a completed human review."""
    packet_path = results_dir / "human_review" / "packet.json"
    if not packet_path.exists():
        raise SystemExit(f"no review packet at {packet_path}; run this script without --score first")
    packet = read_json(packet_path)

    with path.open(newline="", encoding="utf-8") as fh:
        responses = [
            {
                k: (int(v) if k.startswith("q") and v.strip().isdigit() else v.strip())
                for k, v in row.items()
                if v is not None
            }
            for row in csv.DictReader(fh)
        ]
    responses = [r for r in responses if r.get("note_id") and r.get("label")]
    if not responses:
        raise SystemExit(f"{path} contained no completed ratings")

    summary: dict = {"n_responses": len(responses)}
    for question in ("q1_sensibility", "q2_completeness"):
        if any(question in r for r in responses):
            summary[question] = score_review(responses, packet["answer_key"], question=question)

    write_json(results_dir / "metrics" / "human_review.json", summary)
    LOGGER.info("--- human review (%d ratings) ---", len(responses))
    for question, block in summary.items():
        if not isinstance(block, dict) or "methods" not in block:
            continue
        LOGGER.info("%s:", question)
        for method, stats in sorted(block["methods"].items(), key=lambda kv: -kv[1]["mean"]):
            alpha = block["inter_rater"]["krippendorff_alpha_by_method"].get(method)
            LOGGER.info(
                "  %-22s mean %.2f +/- %.2f (n=%d, alpha=%.2f)",
                method, stats["mean"], stats["std"], stats["n_ratings"],
                alpha if alpha == alpha else float("nan"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--model", default=None)
    parser.add_argument("--score", default=None, help="path to a completed responses.csv")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))

    if args.score:
        score_responses(Path(args.score), results_dir)
        return

    device = "cuda" if torch.cuda.is_available() and cfg.get_path("experiment.device") != "cpu" else "cpu"
    seed = int(cfg.get_path("split.seed", 42))
    set_seed(seed)

    backbone = args.model
    if backbone is None:
        evaluation = results_dir / "metrics" / "evaluation.json"
        backbone = (
            read_json(evaluation).get("best_model", {}).get("backbone") if evaluation.exists() else None
        ) or cfg.get_path("serve.default_model", "clinicalbert")

    checkpoint = results_dir / "checkpoints" / backbone
    if not (checkpoint / "medxplain_config.json").exists():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    model, tokenizer = ClinicalRiskModel.load(checkpoint, device=device)
    max_length = model.cfg.max_length

    df = pd.read_parquet(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    test = df[df["split"] == "test"].copy()
    test["gold_spans"] = test["gold_spans"].map(json.loads)

    outcome = cfg.get_path("experiment.primary_outcome", "mortality")
    weight_key = "mortality_weight" if outcome == "mortality" else "readmission_weight"

    n_notes = int(cfg.get_path("evaluation.plausibility_n_notes", 40))
    sample = test[test["gold_spans"].map(len) > 0].sample(
        n=min(n_notes, int((test["gold_spans"].map(len) > 0).sum())), random_state=seed
    )
    if sample.empty:
        LOGGER.warning(
            "no gold rationale spans in this dataset (expected for MIMIC-III); "
            "only the human review packet will be produced"
        )

    methods = list(cfg.get_path("explain.methods", ["attention_rollout", "shap"]))
    explanations_by_method: dict[str, list] = {}
    output: dict = {
        "model": backbone,
        "outcome": outcome,
        "n_notes": len(sample),
        "methods": {},
        "note": (
            "Gold rationales come from the synthetic generator's recorded risk-factor "
            "spans. On a real cohort no such ground truth exists and plausibility is "
            "human-rated instead."
        ),
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
        needs_grad = method in ("integrated_gradients", "attention_grad")

        rows, explanations = [], []
        for i, (_, note) in enumerate(sample.iterrows()):
            try:
                with torch.set_grad_enabled(needs_grad):
                    explanation = explainer.explain(note["text"])
                explanations.append(explanation)
                rows.append(
                    rationale_metrics(explanation, note["text"], note["gold_spans"], weight_key)
                )
            except Exception as exc:
                LOGGER.warning("  note %d failed: %s", i, exc)
            if (i + 1) % 10 == 0:
                LOGGER.info("  %d/%d notes", i + 1, len(sample))

        explanations_by_method[method] = explanations
        summary = summarise_rationale_metrics(rows)
        output["methods"][method] = summary
        if "token_f1" in summary:
            LOGGER.info(
                "  token F1 %.3f | rationale AUPRC %.3f | weight rho %.3f | sign agree %.3f",
                summary["token_f1"]["mean"],
                summary.get("rationale_auprc", {}).get("mean", float("nan")),
                summary.get("weight_spearman", {}).get("mean", float("nan")),
                summary.get("sign_agreement", {}).get("mean", float("nan")),
            )
        write_json(results_dir / "metrics" / "plausibility.json", output)

    # ---- human review packet --------------------------------------------------------
    review_n = int(cfg.get_path("evaluation.human_review_n_notes", 20))
    usable = {m: e for m, e in explanations_by_method.items() if len(e) >= 1}
    if usable:
        count = min(review_n, min(len(e) for e in usable.values()), len(sample))
        packet = build_review_packet(
            sample.head(count).to_dict("records"),
            {m: e[:count] for m, e in usable.items()},
            top_k=8,
            seed=seed,
        )
        review_dir = results_dir / "human_review"
        write_json(review_dir / "packet.json", packet)
        _write_review_csv(review_dir, packet)
        _write_review_html(review_dir, packet)
        LOGGER.info("wrote review packet for %d notes to %s", count, review_dir)


def _write_review_csv(directory: Path, packet: dict) -> None:
    """A blank rating sheet, one row per (note, blinded method)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "responses_template.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rater", "note_id", "label", "q1_sensibility", "q2_completeness", "q3_would_show", "comment"])
        for item in packet["items"]:
            for label in sorted(item["candidates"]):
                writer.writerow(["", item["note_id"], label, "", "", "", ""])


def _write_review_html(directory: Path, packet: dict) -> None:
    """Self-contained review form a rater can open in a browser."""
    rubric = "".join(
        f"<li><b>{score}</b> - {text}</li>" for score, text in sorted(packet["rubric"].items(), reverse=True)
    )
    items = []
    for item in packet["items"]:
        candidates = "".join(
            "<div class='cand'><h4>Set {label}</h4><p>{phrases}</p></div>".format(
                label=label,
                phrases=" &nbsp;·&nbsp; ".join(
                    f"<span class='ph'>{_escape(p)}</span>" for p in item["candidates"][label]
                ),
            )
            for label in sorted(item["candidates"])
        )
        items.append(
            f"""<section>
  <h3>{item['note_id']} &mdash; predicted risk {item['probability'] * 100:.1f}%</h3>
  <pre>{_escape(item['text'])}</pre>
  {candidates}
</section>"""
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MedXplain explanation review</title>
<style>
 body{{font:14px/1.6 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1e293b}}
 pre{{background:#f8fafc;border:1px solid #e2e8f0;padding:1rem;white-space:pre-wrap;font-size:12px;
      max-height:340px;overflow:auto;border-radius:8px}}
 .cand{{border:1px solid #e2e8f0;border-radius:8px;padding:.75rem 1rem;margin:.5rem 0;background:#fff}}
 .cand h4{{margin:0 0 .4rem}} .ph{{background:#fee2e2;padding:1px 6px;border-radius:4px;font-family:ui-monospace,monospace;font-size:12px}}
 section{{border-top:2px solid #e2e8f0;padding-top:1.5rem;margin-top:2rem}}
 ol.rubric li{{margin:.25rem 0}}
</style></head><body>
<h1>MedXplain &mdash; explanation review</h1>
<pre style="max-height:none">{_escape(packet['instructions'])}</pre>
<h2>Rubric</h2><ol class="rubric" reversed>{rubric}</ol>
<p>Record your scores in <code>responses_template.csv</code> (copy it to
<code>responses.csv</code>), then run
<code>python scripts/06_plausibility.py --score results/human_review/responses.csv</code>.</p>
{''.join(items)}
</body></html>"""
    (directory / "review_form.html").write_text(html, encoding="utf-8")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
