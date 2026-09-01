"""Phase 1 - build, clean, split and document the dataset.

Produces:
    data/processed/corpus.parquet     notes + labels + split assignment
    data/processed/examples.json      case-study notes for the frontend demo
    results/metrics/dataset_report.json   the paper's Data & Preprocessing table

Run:
    python scripts/00_build_dataset.py
    python scripts/00_build_dataset.py --set data.source=mimic --set data.mimic_root=/path/to/mimic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from medxplain.config import load_config, parse_overrides, save_config
from medxplain.data.preprocess import PreprocessConfig, clean_note, note_length_stats
from medxplain.data.splits import SplitConfig, check_leakage, split_report, stratified_split
from medxplain.data.synthetic import SyntheticCorpusConfig, generate_corpus, iter_examples
from medxplain.utils import env_report, get_logger, set_seed, timer, write_json

LOGGER = get_logger("medxplain.scripts.dataset")


def build_synthetic(cfg) -> tuple[pd.DataFrame, list[dict]]:
    data = cfg.section("data")
    corpus_cfg = SyntheticCorpusConfig(
        n_notes=int(data.get("n_notes", 3000)),
        seed=int(cfg.get_path("split.seed", 42)),
        target_positive_rate=float(data.get("target_positive_rate", 0.16)),
        target_readmission_rate=float(data.get("target_readmission_rate", 0.20)),
        label_noise=float(data.get("label_noise", 0.05)),
    )
    LOGGER.info("generating %d synthetic discharge summaries", corpus_cfg.n_notes)
    rows = generate_corpus(corpus_cfg)
    df = pd.DataFrame(rows)
    # synthetic notes are one-per-patient by construction
    df["subject_id"] = range(len(df))
    return df, rows


def build_mimic(cfg) -> tuple[pd.DataFrame, list[dict]]:
    from medxplain.data.mimic import MimicConfig, build_cohort

    data = cfg.section("data")
    root = data.get("mimic_root")
    if not root:
        raise SystemExit(
            "data.mimic_root must point at the directory containing ADMISSIONS.csv, "
            "PATIENTS.csv and NOTEEVENTS.csv"
        )
    df = build_cohort(
        MimicConfig(
            root=Path(root),
            mask_leakage=bool(data.get("mimic_mask_leakage", True)),
            limit=data.get("mimic_limit"),
        )
    )
    # no gold rationales exist for real notes; plausibility there is human-rated
    df["gold_spans"] = [[] for _ in range(len(df))]
    df["factors"] = [[] for _ in range(len(df))]
    return df, df.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="override a config value, e.g. --set data.n_notes=500")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    set_seed(int(cfg.get_path("split.seed", 42)))

    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    source = cfg.get_path("data.source", "synthetic")

    with timer("dataset build"):
        if source == "synthetic":
            df, rows = build_synthetic(cfg)
        elif source == "mimic":
            df, rows = build_mimic(cfg)
        else:
            raise SystemExit(f"unknown data.source: {source}")

        # ---- cleaning ---------------------------------------------------------------
        pre = cfg.section("preprocess")
        pre_cfg = PreprocessConfig(
            replace_deid=bool(pre.get("replace_deid", True)),
            collapse_whitespace=bool(pre.get("collapse_whitespace", True)),
            strip_rule_lines=bool(pre.get("strip_rule_lines", True)),
            lowercase=bool(pre.get("lowercase", False)),
            expand_abbreviations=bool(pre.get("expand_abbreviations", False)),
            max_chars=pre.get("max_chars"),
        )
        # The raw note is kept verbatim: gold rationale offsets are defined against
        # it, and the API explains whatever text it is given, cleaned the same way.
        df["text_raw"] = df["text"]
        df["text"] = df["text_raw"].map(lambda t: clean_note(t, pre_cfg))

        before = len(df)
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
        if before != len(df):
            LOGGER.info("dropped %d exact-duplicate notes", before - len(df))

        # ---- splitting --------------------------------------------------------------
        label = cfg.get_path("experiment.primary_outcome", "mortality")
        split_cfg = SplitConfig(
            test_size=float(cfg.get_path("split.test_size", 0.15)),
            val_size=float(cfg.get_path("split.val_size", 0.15)),
            seed=int(cfg.get_path("split.seed", 42)),
            label_col=label,
            group_col=cfg.get_path("split.group_col"),
        )
        df = stratified_split(df, split_cfg)

        leakage = check_leakage(df, split_cfg.group_col)
        if not leakage["ok"]:
            LOGGER.warning("split leakage check failed: %s", leakage)

    # ---- persistence ----------------------------------------------------------------
    processed = Path(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    processed.parent.mkdir(parents=True, exist_ok=True)
    to_save = df.copy()
    # parquet cannot store the nested gold-span dicts directly; JSON-encode them
    import json

    to_save["gold_spans"] = to_save["gold_spans"].map(json.dumps)
    to_save["factors"] = to_save["factors"].map(json.dumps)
    to_save.to_parquet(processed, index=False)
    LOGGER.info("wrote %s (%d notes)", processed, len(to_save))

    # case-study examples for the frontend
    test_rows = df[df["split"] == "test"].to_dict("records")
    examples = iter_examples(test_rows, n=6) if source == "synthetic" else test_rows[:6]
    example_payload = [
        {
            "note_id": r["note_id"],
            "title": _example_title(r, label),
            "text": r["text"],
            "true_label": int(r[label]),
            "outcome": label,
        }
        for r in examples
    ]
    write_json(Path("data/processed/examples.json"), example_payload)

    # ---- report ---------------------------------------------------------------------
    report: dict = {
        "source": source,
        "n_notes": int(len(df)),
        "primary_outcome": label,
        "splits": split_report(df, (label, cfg.get_path("experiment.secondary_outcome", ""))),
        "leakage_check": leakage,
        "preprocess": pre_cfg.__dict__,
        "environment": env_report(),
        "char_length": {
            "mean": float(df["text"].str.len().mean()),
            "median": float(df["text"].str.len().median()),
            "max": int(df["text"].str.len().max()),
        },
    }
    if source == "synthetic":
        report["gold_rationales"] = {
            "mean_spans_per_note": float(df["gold_spans"].map(len).mean()),
            "notes_with_spans": int((df["gold_spans"].map(len) > 0).sum()),
        }

    try:
        from transformers import AutoTokenizer

        from medxplain.models.transformer import MODEL_REGISTRY

        backbone = cfg.get_path("model.backbones", ["clinicalbert"])[0]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY.get(backbone, backbone))
        report["token_length"] = note_length_stats(df["text"].tolist(), tokenizer)
        report["token_length"]["tokenizer"] = MODEL_REGISTRY.get(backbone, backbone)
    except Exception as exc:  # tokenizer download unavailable
        LOGGER.warning("skipping token-length stats: %s", exc)

    write_json(results_dir / "metrics" / "dataset_report.json", report)
    save_config(cfg, results_dir / "config_used.yaml")

    LOGGER.info("dataset report:")
    for name, block in report["splits"].items():
        LOGGER.info(
            "  %-5s n=%-5d %s rate=%.3f",
            name, block["n"], label, block.get(f"{label}_rate", float("nan")),
        )


def _example_title(row: dict, label: str) -> str:
    """A short, honest label for the case-study picker in the UI."""
    factors = row.get("factors") or []
    if isinstance(factors, str):
        import json

        factors = json.loads(factors)
    pretty = {
        "septic_shock": "septic shock",
        "cardiac_arrest": "cardiac arrest",
        "comfort_measures": "comfort measures",
        "hospice": "hospice transition",
        "metastatic_cancer": "metastatic cancer",
        "stable_course": "uncomplicated course",
        "mech_vent": "mechanical ventilation",
        "chf": "heart failure",
    }
    highlights = [pretty[f] for f in factors if f in pretty][:2]
    age = row.get("age", "")
    if highlights:
        return f"{age}y - {', '.join(highlights)}"
    return f"{age}y admission"


if __name__ == "__main__":
    main()
