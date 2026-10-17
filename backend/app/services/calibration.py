"""Calibration evidence for a served checkpoint, read from the evaluation outputs.

The risk score is only shown as a percentage chance when the evaluation files
show it is reasonably calibrated. The evidence is the per-seed test metrics in
``results/metrics/transformers.json`` for the exact seed the checkpoint came
from; nothing here is estimated or assumed.
"""

from __future__ import annotations

from pathlib import Path

from medxplain.utils import read_json

#: A score is presented as a probability only if both hold on the test split.
MAX_ECE = 0.05
MIN_BRIER_SKILL = 0.0


def calibration_status(results_root: Path, model_key: str, seed: int | None) -> dict:
    path = Path(results_root) / "metrics" / "transformers.json"
    base = {
        "evaluated": False,
        "well_calibrated": False,
        "ece": None,
        "brier_skill_score": None,
        "seed": seed,
        "criteria": f"ECE <= {MAX_ECE} and Brier skill score > {MIN_BRIER_SKILL} on the test split",
    }
    if not path.exists():
        return {**base, "summary": "Calibration has not been evaluated for this model."}
    try:
        block = read_json(path).get("models", {}).get(model_key, {})
    except Exception:
        return {**base, "summary": "Calibration results could not be read."}

    runs = block.get("per_seed", [])
    run = next((r for r in runs if seed is not None and r.get("seed") == seed), None)
    if run is None and len(runs) == 1:
        run = runs[0]
    if run is None or run.get("ece") is None:
        return {**base, "summary": "Calibration has not been evaluated for this checkpoint."}

    ece = float(run["ece"])
    bss = run.get("brier_skill_score")
    bss = float(bss) if bss is not None else None
    ok = ece <= MAX_ECE and (bss is not None and bss > MIN_BRIER_SKILL)
    if ok:
        summary = "On held-out test notes the model's scores matched observed death rates closely."
    else:
        summary = (
            "On held-out test notes the model's scores did not match observed death rates "
            f"(expected calibration error {ece:.2f}"
            + (f", Brier skill score {bss:.2f}" if bss is not None else "")
            + "). Treat the score as a ranking signal, not as a percentage chance."
        )
    return {
        **base,
        "evaluated": True,
        "well_calibrated": ok,
        "ece": ece,
        "brier_skill_score": bss,
        "summary": summary,
    }
