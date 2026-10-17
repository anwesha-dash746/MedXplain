"""Re-select a served model's alert level (decision threshold) for a target recall.

    python scripts/set_alert_level.py clinicalbert_512 --target-recall 0.8

The threshold is chosen on the VALIDATION predictions only - the highest
threshold that still catches ``target_recall`` of validation deaths - and the
test-set metrics at that threshold are then recomputed and reported, so the
test numbers remain an honest check rather than something the choice was fitted
to. Updates the checkpoint's run_info.json (what the API serves) and the
model's entry in results/metrics/transformers.json, keeping the previous
threshold and metrics under ``previous_operating_point``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from medxplain.eval.metrics import aggregate_seeds, classification_metrics, threshold_at_recall  # noqa: E402
from medxplain.utils import read_json, write_json  # noqa: E402

KEEP = ("threshold", "recall", "precision", "f1", "specificity", "balanced_accuracy", "confusion")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_key")
    parser.add_argument("--target-recall", type=float, required=True)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    ckpt = args.results / "checkpoints" / args.model_key
    info = read_json(ckpt / "run_info.json")
    seed = info["seed"]
    preds = np.load(args.results / "predictions" / f"{args.model_key}_seed{seed}.npz")
    y_val, p_val = preds["y_val"], preds["val_probabilities"]
    y_test, p_test = preds["y_test"], preds["test_probabilities"]

    threshold = threshold_at_recall(y_val, p_val, args.target_recall)
    val = classification_metrics(y_val, p_val, threshold=threshold)
    test = classification_metrics(y_test, p_test, threshold=threshold)

    metrics_path = args.results / "metrics" / "transformers.json"
    published = read_json(metrics_path)
    run = next(r for r in published["models"][args.model_key]["per_seed"] if r["seed"] == seed)
    previous = {k: run[k] for k in KEEP if k in run}
    run.setdefault("previous_operating_point", {**previous, "policy": run.get("threshold_policy", "best F1 on validation")})
    for k in KEEP:
        if k in test:
            run[k] = test[k]
    run["threshold"] = float(threshold)
    run["threshold_policy"] = f"recall >= {args.target_recall:.2f} on validation"
    block = published["models"][args.model_key]
    block["aggregate"] = aggregate_seeds(block["per_seed"])  # F1 depends on the threshold
    write_json(metrics_path, published)

    info["threshold"] = float(threshold)
    info["threshold_policy"] = run["threshold_policy"]
    write_json(ckpt / "run_info.json", info)

    print(f"{args.model_key}: alert level {previous.get('threshold', float('nan')):.3f} -> {threshold:.3f}")
    print(f"  validation: recall {val['recall']:.2f}  precision {val['precision']:.2f}  alert rate {np.mean(p_val >= threshold):.2f}")
    print(f"  test:       recall {test['recall']:.2f}  precision {test['precision']:.2f}  alert rate {np.mean(p_test >= threshold):.2f}"
          f"  F1 {test['f1']:.3f}  (was recall {previous.get('recall', float('nan')):.2f}, precision {previous.get('precision', float('nan')):.2f})")
    c = test["confusion"]
    print(f"  test counts: {c['tp']} of {c['tp'] + c['fn']} deaths flagged; {c['fp']} false alarms among {c['fp'] + c['tn']} survivors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
