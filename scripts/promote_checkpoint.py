"""Install a separately trained checkpoint as an extra servable model.

    python scripts/promote_checkpoint.py results/clinicalbert_512 clinicalbert clinicalbert_512 \
        --display-name "Bio_ClinicalBERT (512 tokens)"

Copies ``<run_dir>/checkpoints/<backbone>`` to ``results/checkpoints/<new_key>``
and merges that run's metrics block into ``results/metrics/transformers.json``
under ``<new_key>``, leaving every other model's published results untouched.
The API then lists it as a model, reads its calibration evidence, and the
Technical Details section shows its scores next to the others.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from medxplain.utils import read_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="results directory of the separate training run")
    parser.add_argument("backbone", help="backbone key inside that run, e.g. clinicalbert")
    parser.add_argument("new_key", help="model key to serve it under, e.g. clinicalbert_512")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    src = args.run_dir / "checkpoints" / args.backbone
    run_metrics = read_json(args.run_dir / "metrics" / "transformers.json")
    block = run_metrics["models"][args.backbone]
    if not (src / "medxplain_config.json").exists():
        raise SystemExit(f"no checkpoint at {src}")

    dst = args.results / "checkpoints" / args.new_key
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    info = read_json(dst / "run_info.json")
    info["backbone"] = args.new_key
    if args.display_name:
        info["display_name"] = args.display_name
    write_json(dst / "run_info.json", info)

    for seed_run in block["per_seed"]:
        seed_run["backbone"] = args.new_key
        npz = args.run_dir / "predictions" / f"{args.backbone}_seed{seed_run['seed']}.npz"
        if npz.exists():
            (args.results / "predictions").mkdir(parents=True, exist_ok=True)
            shutil.copy2(npz, args.results / "predictions" / f"{args.new_key}_seed{seed_run['seed']}.npz")
    block["checkpoint"] = str(dst.relative_to(REPO_ROOT)) if dst.is_relative_to(REPO_ROOT) else str(dst)

    metrics_path = args.results / "metrics" / "transformers.json"
    published = read_json(metrics_path)
    published.setdefault("models", {})[args.new_key] = block
    write_json(metrics_path, published)

    agg = block.get("aggregate", {})
    print(f"installed {args.new_key} -> {dst}")
    if "auroc" in agg:
        print(f"  test AUROC {agg['auroc']['mean']:.4f}  AUPRC {agg['auprc']['mean']:.4f}  "
              f"Brier {agg['brier']['mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
