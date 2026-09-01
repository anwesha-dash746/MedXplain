"""Phase 7 - generate every figure and LaTeX table from the saved results.

Reads only ``results/metrics/*.json``, so it never retrains anything and can be
rerun freely while drafting. Missing inputs are skipped with a warning rather
than raising, so the script is useful part-way through the pipeline.

Produces in ``results/figures``:
    fig1_roc_pr.png              ROC and precision-recall curves for all models
    fig2_model_comparison.png    AUROC/AUPRC bars with multi-seed error bars
    fig3_calibration.png         reliability diagram for the best model
    fig4_deletion_insertion.png  faithfulness deletion/insertion curves
    fig5_faithfulness_bars.png   comprehensiveness/sufficiency by method
    fig6_cost_vs_faithfulness.png  the deployability trade-off
    fig7_stability.png           agreement under paraphrase, by method
    fig8_attention_heatmap.png   token-importance heatmap for one example note

and in ``results/tables``: main_results.tex, faithfulness.tex, stability.tex,
plausibility.tex, plus .md copies for the README.

Run:
    python scripts/07_figures_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medxplain.config import load_config, parse_overrides
from medxplain.utils import get_logger, read_json

LOGGER = get_logger("medxplain.scripts.figures")

PALETTE = ["#265f8c", "#c53030", "#0f9d76", "#7c3aed", "#dd6b20", "#0891b2", "#64748b"]

PRETTY_MODEL = {
    "clinicalbert": "Bio_ClinicalBERT",
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
    "bert": "BERT-base",
    "logreg": "TF-IDF + LR",
    "random_forest": "TF-IDF + RF",
    "xgboost": "TF-IDF + XGB",
}
PRETTY_METHOD = {
    "attention_rollout": "Attention rollout",
    "attention_raw": "Raw attention",
    "attention_grad": "Grad-weighted attn.",
    "shap": "SHAP",
    "occlusion": "Occlusion",
    "integrated_gradients": "Integrated grads",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def load(path: Path) -> dict | None:
    if not path.exists():
        LOGGER.warning("missing %s - skipping the figures that need it", path.name)
        return None
    try:
        return read_json(path)
    except Exception as exc:
        LOGGER.warning("could not read %s: %s", path.name, exc)
        return None


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_roc_pr(evaluation: dict, out: Path) -> None:
    models = evaluation.get("models", {})
    curves = [(k, v["curves"]) for k, v in models.items() if "curves" in v]
    if not curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for i, (name, curve) in enumerate(curves):
        color = PALETTE[i % len(PALETTE)]
        label = PRETTY_MODEL.get(name, name)
        roc, pr = curve["roc"], curve["pr"]
        axes[0].plot(roc["fpr"], roc["tpr"], color=color, lw=1.6,
                     label=f"{label} ({roc['auc']:.3f})")
        axes[1].plot(pr["recall"], pr["precision"], color=color, lw=1.6,
                     label=f"{label} ({pr['ap']:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    prevalence = evaluation.get("prevalence")
    if prevalence:
        axes[1].axhline(prevalence, ls="--", color="#cbd5e1", lw=1)
        axes[1].text(0.02, prevalence + 0.015, f"no-skill ({prevalence:.2f})",
                     fontsize=7, color="#64748b")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower right" if ax is axes[0] else "upper right")

    fig.suptitle("Held-out test performance (seed-ensembled predictions)", fontsize=11, y=1.02)
    fig.savefig(out / "fig1_roc_pr.png")
    plt.close(fig)
    LOGGER.info("wrote fig1_roc_pr.png")


def fig_model_comparison(evaluation: dict, out: Path) -> None:
    rows = []
    for name, block in evaluation.get("models", {}).items():
        if "aggregate" in block:
            agg = block["aggregate"]
            rows.append((name, agg["auroc"]["mean"], agg["auroc"]["std"],
                         agg["auprc"]["mean"], agg["auprc"]["std"], block.get("family", "")))
        elif "test" in block:
            t = block["test"]
            rows.append((name, t["auroc"], 0.0, t["auprc"], 0.0, block.get("family", "")))
    if not rows:
        return
    rows.sort(key=lambda r: r[3])

    labels = [PRETTY_MODEL.get(r[0], r[0]) for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.42 * len(rows) + 1.8), sharey=True)

    for ax, (mean_i, std_i, title) in zip(axes, [(1, 2, "AUROC"), (3, 4, "AUPRC")]):
        values = [r[mean_i] for r in rows]
        errors = [r[std_i] for r in rows]
        colors = ["#265f8c" if r[5] == "transformer" else "#94a3b8" for r in rows]
        ax.barh(y, values, xerr=errors, color=colors, height=0.62,
                error_kw={"ecolor": "#334155", "lw": 1, "capsize": 3})
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1.0)
        for yi, (v, e) in enumerate(zip(values, errors)):
            ax.text(v + e + 0.012, yi, f"{v:.3f}", va="center", fontsize=7, color="#334155")

    prevalence = evaluation.get("prevalence")
    if prevalence:
        axes[1].axvline(prevalence, ls="--", color="#cbd5e1", lw=1)

    n_seeds = max(
        (b["aggregate"]["auroc"].get("n_seeds", 1) for b in evaluation["models"].values()
         if "aggregate" in b),
        default=1,
    )
    fig.suptitle(
        f"Model comparison (error bars: ±1 SD over {n_seeds} seeds; baselines single-run)",
        fontsize=10, y=1.0,
    )
    fig.savefig(out / "fig2_model_comparison.png")
    plt.close(fig)
    LOGGER.info("wrote fig2_model_comparison.png")


def fig_calibration(evaluation: dict, out: Path) -> None:
    best = evaluation.get("best_model", {}).get("backbone")
    models = evaluation.get("models", {})
    targets = [k for k in (best, "logreg") if k in models and "calibration" in models[k]]
    if not targets:
        return

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1, label="perfect calibration")
    for i, name in enumerate(targets):
        cal = models[name]["calibration"]
        bins = cal.get("bins", [])
        if not bins:
            continue
        x = [b["mean_predicted"] for b in bins]
        y = [b["observed_rate"] for b in bins]
        sizes = np.array([b["count"] for b in bins], dtype=float)
        ax.plot(x, y, "-o", color=PALETTE[i], lw=1.5, ms=4,
                label=f"{PRETTY_MODEL.get(name, name)} (ECE {cal['ece']:.3f})")
        ax.scatter(x, y, s=8 + 90 * sizes / sizes.max(), color=PALETTE[i], alpha=0.25)

    ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate",
           title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig3_calibration.png")
    plt.close(fig)
    LOGGER.info("wrote fig3_calibration.png")


def fig_deletion_insertion(faithfulness: dict, out: Path) -> None:
    curves = faithfulness.get("curves", {})
    usable = {k: v for k, v in curves.items() if v.get("fractions")}
    if not usable:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for i, (method, curve) in enumerate(usable.items()):
        color = PALETTE[i % len(PALETTE)]
        x = curve["fractions"]
        label = PRETTY_METHOD.get(method, method)
        for ax, key, auc_key in ((axes[0], "deletion_mean", "deletion_auc"),
                                 (axes[1], "insertion_mean", "insertion_auc")):
            y = np.asarray(curve[key])
            sd = np.asarray(curve[key.replace("_mean", "_std")])
            ax.plot(x, y, color=color, lw=1.7,
                    label=f"{label} (AUC {curve[auc_key]['mean']:.3f})")
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, lw=0)

    axes[0].set(xlabel="Fraction of top tokens masked", ylabel="Confidence in predicted class",
                title="Deletion (steeper drop = more faithful)")
    axes[1].set(xlabel="Fraction of top tokens retained",
                title="Insertion (faster rise = more faithful)")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Faithfulness curves - {PRETTY_MODEL.get(faithfulness.get('model'), faithfulness.get('model'))}, "
        f"{faithfulness.get('n_notes')} test notes",
        fontsize=10, y=1.02,
    )
    fig.savefig(out / "fig4_deletion_insertion.png")
    plt.close(fig)
    LOGGER.info("wrote fig4_deletion_insertion.png")


def fig_faithfulness_bars(faithfulness: dict, out: Path) -> None:
    methods = {
        k: v for k, v in faithfulness.get("methods", {}).items() if "comprehensiveness" in v
    }
    if not methods:
        return
    order = sorted(methods, key=lambda m: methods[m]["comprehensiveness"]["mean"])
    y = np.arange(len(order))
    labels = [PRETTY_METHOD.get(m, m) for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.45 * len(order) + 1.8), sharey=True)
    comp = [methods[m]["comprehensiveness"]["mean"] for m in order]
    comp_err = [methods[m]["comprehensiveness"].get("sem", 0) for m in order]
    rand = [methods[m]["comprehensiveness_random"]["mean"] for m in order]
    suff = [methods[m]["sufficiency"]["mean"] for m in order]
    suff_err = [methods[m]["sufficiency"].get("sem", 0) for m in order]

    axes[0].barh(y, comp, xerr=comp_err, color="#265f8c", height=0.6,
                 error_kw={"ecolor": "#334155", "lw": 1, "capsize": 3}, label="explanation")
    axes[0].barh(y, rand, height=0.6, color="none", edgecolor="#c53030", lw=1.2,
                 ls="--", label="random control")
    axes[0].set(xlabel="Comprehensiveness (higher is better)", title="Comprehensiveness")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right")

    axes[1].barh(y, suff, xerr=suff_err, color="#0f9d76", height=0.6,
                 error_kw={"ecolor": "#334155", "lw": 1, "capsize": 3})
    axes[1].set(xlabel="Sufficiency (lower is better)", title="Sufficiency")

    fig.suptitle("Explanation faithfulness by method", fontsize=10, y=1.0)
    fig.savefig(out / "fig5_faithfulness_bars.png")
    plt.close(fig)
    LOGGER.info("wrote fig5_faithfulness_bars.png")


def fig_cost_vs_faithfulness(faithfulness: dict, out: Path) -> None:
    """The trade-off that decides what is actually deployable at the bedside."""
    methods = faithfulness.get("methods", {})
    cost = faithfulness.get("cost", {})
    points = [
        (m, cost.get(m, {}).get("mean_explain_seconds"), methods[m]["comprehensiveness"]["mean"])
        for m in methods
        if "comprehensiveness" in methods[m] and cost.get(m, {}).get("mean_explain_seconds")
    ]
    if len(points) < 2:
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, (method, seconds, comp) in enumerate(points):
        ax.scatter(seconds, comp, s=90, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(PRETTY_METHOD.get(method, method), (seconds, comp),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="Explanation time per note (s, log scale)",
           ylabel="Comprehensiveness",
           title="Faithfulness vs. compute cost")
    fig.savefig(out / "fig6_cost_vs_faithfulness.png")
    plt.close(fig)
    LOGGER.info("wrote fig6_cost_vs_faithfulness.png")


def fig_stability(stability: dict, out: Path) -> None:
    methods = stability.get("methods", {})
    rows = [
        (
            m,
            block.get("paraphrase", {}).get("spearman", {}).get("mean"),
            block.get("paraphrase", {}).get("top_k_jaccard", {}).get("mean"),
            block.get("distractor", {}).get("distractor_mass", {}).get("mean"),
        )
        for m, block in methods.items()
    ]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return
    rows.sort(key=lambda r: r[1])

    y = np.arange(len(rows))
    labels = [PRETTY_METHOD.get(r[0], r[0]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.45 * len(rows) + 1.8), sharey=True)

    for ax, idx, title, color in (
        (axes[0], 1, "Spearman ρ (paraphrase)", "#265f8c"),
        (axes[1], 2, "Top-10 Jaccard", "#0f9d76"),
        (axes[2], 3, "Distractor mass (lower better)", "#c53030"),
    ):
        values = [r[idx] if r[idx] is not None else 0.0 for r in rows]
        ax.barh(y, values, color=color, height=0.6)
        ax.set(xlabel=title)
        for yi, v in enumerate(values):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7, color="#334155")
    axes[0].set_yticks(y, labels, fontsize=8)

    fig.suptitle("Explanation stability under meaning-preserving edits", fontsize=10, y=1.0)
    fig.savefig(out / "fig7_stability.png")
    plt.close(fig)
    LOGGER.info("wrote fig7_stability.png")


def fig_attention_heatmap(cfg, results_dir: Path, out: Path) -> None:
    """Token-importance heatmap over one example note, per method."""
    import pandas as pd
    import torch

    from medxplain.explain import build_explainer
    from medxplain.models.transformer import ClinicalRiskModel

    evaluation = load(results_dir / "metrics" / "evaluation.json")
    backbone = (evaluation or {}).get("best_model", {}).get("backbone") or cfg.get_path(
        "serve.default_model", "clinicalbert"
    )
    checkpoint = results_dir / "checkpoints" / backbone
    if not (checkpoint / "medxplain_config.json").exists():
        LOGGER.warning("no checkpoint for the heatmap figure")
        return

    df = pd.read_parquet(cfg.get_path("data.processed_path", "data/processed/corpus.parquet"))
    test = df[df["split"] == "test"]
    label = cfg.get_path("experiment.primary_outcome", "mortality")
    positives = test[test[label] == 1]
    if positives.empty:
        return
    text = positives.iloc[0]["text"]

    model, tokenizer = ClinicalRiskModel.load(checkpoint, device="cpu")
    methods = [m for m in cfg.get_path("explain.methods", []) if m != "occlusion"][:4]
    if not methods:
        return

    grids, labels_out, words = [], [], None
    for method in methods:
        kwargs: dict = {"max_length": model.cfg.max_length}
        if method == "shap":
            kwargs["max_evals"] = int(cfg.get_path("explain.shap_max_evals", 200))
        elif method == "integrated_gradients":
            kwargs["n_steps"] = int(cfg.get_path("explain.ig_steps", 16))
        try:
            explainer = build_explainer(method, model, tokenizer, device="cpu", **kwargs)
            with torch.set_grad_enabled(method in ("integrated_gradients", "attention_grad")):
                explanation = explainer.explain(text)
        except Exception as exc:
            LOGGER.warning("heatmap: %s failed (%s)", method, exc)
            continue
        scored = [w for w in explanation.words if w.token_indices][:60]
        if words is None:
            words = [w.text for w in scored]
        values = np.array([w.score for w in scored], dtype=float)
        peak = np.abs(values).max() or 1.0
        grids.append(values / peak)
        labels_out.append(PRETTY_METHOD.get(method, method))

    if not grids or words is None:
        return
    width = min(len(words), min(len(g) for g in grids))
    matrix = np.vstack([g[:width] for g in grids])

    fig, ax = plt.subplots(figsize=(max(8.0, width * 0.22), 0.5 * len(grids) + 1.8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(labels_out)), labels_out, fontsize=8)
    ax.set_xticks(range(width), words[:width], rotation=90, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="normalised attribution")
    ax.set_title(
        f"Token attributions on one high-risk test note ({PRETTY_MODEL.get(backbone, backbone)})",
        fontsize=10,
    )
    fig.savefig(out / "fig8_attention_heatmap.png")
    plt.close(fig)
    LOGGER.info("wrote fig8_attention_heatmap.png")


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def _latex(caption: str, label: str, header: list[str], rows: list[list[str]], align: str) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _markdown(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def _pm(block: dict | None, digits: int = 3) -> str:
    if not block:
        return "--"
    if isinstance(block, (int, float)):
        return f"{block:.{digits}f}"
    mean, std = block.get("mean"), block.get("std", 0.0)
    if mean is None or mean != mean:
        return "--"
    return f"{mean:.{digits}f}" + (f" $\\pm$ {std:.{digits}f}" if std else "")


def table_main(evaluation: dict, out: Path) -> None:
    rows, md_rows = [], []
    entries = []
    for name, block in evaluation.get("models", {}).items():
        if "aggregate" in block:
            agg = block["aggregate"]
            entries.append((name, block.get("family", ""), agg, agg["auprc"]["mean"]))
        elif "test" in block:
            t = block["test"]
            wrapped = {k: {"mean": t[k], "std": 0.0} for k in ("auroc", "auprc", "f1", "brier")}
            entries.append((name, block.get("family", ""), wrapped, t["auprc"]))
    entries.sort(key=lambda e: e[3], reverse=True)

    sig = evaluation.get("significance", {}).get("holm", {}).get("results", {})
    for name, family, agg, _ in entries:
        p = sig.get(name, {}).get("p_adjusted")
        p_text = "--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
        rows.append([
            PRETTY_MODEL.get(name, name), _pm(agg.get("auroc")), _pm(agg.get("auprc")),
            _pm(agg.get("f1")), _pm(agg.get("brier")), p_text,
        ])
        md_rows.append([
            PRETTY_MODEL.get(name, name),
            _pm(agg.get("auroc")).replace("$\\pm$", "±"),
            _pm(agg.get("auprc")).replace("$\\pm$", "±"),
            _pm(agg.get("f1")).replace("$\\pm$", "±"),
            _pm(agg.get("brier")).replace("$\\pm$", "±"),
            p_text.replace("$<$", "<"),
        ])

    header = ["Model", "AUROC", "AUPRC", "F1", "Brier", "$p$ (Holm)"]
    caption = (
        f"Held-out test performance on {evaluation.get('outcome', 'the primary outcome')} "
        f"($n={evaluation.get('n_test')}$, prevalence "
        f"{evaluation.get('prevalence', 0):.3f}). Transformer rows are mean $\\pm$ SD over "
        "random seeds; $p$-values are paired-bootstrap comparisons against the best model, "
        "Holm-adjusted."
    )
    (out / "main_results.tex").write_text(
        _latex(caption, "tab:main", header, rows, "l" + "r" * (len(header) - 1)), encoding="utf-8"
    )
    (out / "main_results.md").write_text(
        _markdown([h.replace("$p$", "p").replace("(Holm)", "(Holm)") for h in header], md_rows),
        encoding="utf-8",
    )
    LOGGER.info("wrote main_results.tex / .md")


def table_faithfulness(faithfulness: dict, out: Path) -> None:
    methods = faithfulness.get("methods", {})
    curves = faithfulness.get("curves", {})
    cost = faithfulness.get("cost", {})
    if not methods:
        return
    order = sorted(
        methods, key=lambda m: methods[m].get("comprehensiveness", {}).get("mean", -np.inf), reverse=True
    )
    rows, md_rows = [], []
    for method in order:
        block = methods[method]
        if "comprehensiveness" not in block:
            continue
        seconds = cost.get(method, {}).get("mean_explain_seconds")
        row = [
            PRETTY_METHOD.get(method, method),
            _pm(block["comprehensiveness"]),
            _pm(block["sufficiency"]),
            _pm(block.get("comprehensiveness_over_random")),
            f"{curves.get(method, {}).get('deletion_auc', {}).get('mean', float('nan')):.3f}",
            f"{seconds:.1f}" if seconds else "--",
        ]
        rows.append(row)
        md_rows.append([c.replace("$\\pm$", "±") for c in row])

    header = ["Method", "Compreh. $\\uparrow$", "Suffic. $\\downarrow$",
              "vs. random $\\uparrow$", "Del. AUC $\\downarrow$", "s/note"]
    caption = (
        f"Explanation faithfulness on {faithfulness.get('n_notes')} held-out notes with "
        f"{PRETTY_MODEL.get(faithfulness.get('model'), faithfulness.get('model'))}. "
        "Comprehensiveness and sufficiency are averaged over $k \\in \\{1,5,10,20,50\\}\\%$ of "
        "tokens; `vs. random' is comprehensiveness minus that of a random explanation of the "
        "same size."
    )
    (out / "faithfulness.tex").write_text(
        _latex(caption, "tab:faithfulness", header, rows, "l" + "r" * (len(header) - 1)),
        encoding="utf-8",
    )
    (out / "faithfulness.md").write_text(
        _markdown(["Method", "Compreh. ↑", "Suffic. ↓", "vs. random ↑", "Del. AUC ↓", "s/note"], md_rows),
        encoding="utf-8",
    )
    LOGGER.info("wrote faithfulness.tex / .md")


def table_stability(stability: dict, out: Path) -> None:
    methods = stability.get("methods", {})
    if not methods:
        return
    rows, md_rows = [], []
    for method, block in sorted(
        methods.items(),
        key=lambda kv: kv[1].get("paraphrase", {}).get("spearman", {}).get("mean", -np.inf),
        reverse=True,
    ):
        row = [
            PRETTY_METHOD.get(method, method),
            _pm(block.get("paraphrase", {}).get("spearman")),
            _pm(block.get("paraphrase", {}).get("top_k_jaccard")),
            _pm(block.get("near_duplicate", {}).get("spearman")),
            _pm(block.get("distractor", {}).get("distractor_mass")),
        ]
        rows.append(row)
        md_rows.append([c.replace("$\\pm$", "±") for c in row])

    header = ["Method", "Paraphr. $\\rho$", "Top-10 Jaccard", "Near-dup. $\\rho$",
              "Distractor mass $\\downarrow$"]
    caption = (
        f"Explanation stability over {stability.get('n_notes')} notes. Higher agreement under "
        "meaning-preserving edits is better; distractor mass is the share of importance placed "
        "on an inserted clinically irrelevant sentence."
    )
    (out / "stability.tex").write_text(
        _latex(caption, "tab:stability", header, rows, "l" + "r" * (len(header) - 1)), encoding="utf-8"
    )
    (out / "stability.md").write_text(
        _markdown(["Method", "Paraphrase ρ", "Top-10 Jaccard", "Near-dup ρ", "Distractor mass ↓"], md_rows),
        encoding="utf-8",
    )
    LOGGER.info("wrote stability.tex / .md")


def table_plausibility(plausibility: dict, out: Path) -> None:
    methods = plausibility.get("methods", {})
    if not methods:
        return
    rows, md_rows = [], []
    for method, block in sorted(
        methods.items(),
        key=lambda kv: kv[1].get("token_f1", {}).get("mean", -np.inf),
        reverse=True,
    ):
        if "token_f1" not in block:
            continue
        row = [
            PRETTY_METHOD.get(method, method),
            _pm(block.get("token_f1")),
            _pm(block.get("rationale_auprc")),
            _pm(block.get("weight_spearman")),
            _pm(block.get("sign_agreement")),
            _pm(block.get("off_rationale_mass")),
        ]
        rows.append(row)
        md_rows.append([c.replace("$\\pm$", "±") for c in row])

    header = ["Method", "Token F1", "Rationale AUPRC", "Weight $\\rho$",
              "Sign agree.", "Off-rationale $\\downarrow$"]
    caption = (
        "Explanation plausibility against the gold rationale spans recorded by the corpus "
        "generator. Sign agreement is undefined for unsigned (attention) methods."
    )
    (out / "plausibility.tex").write_text(
        _latex(caption, "tab:plausibility", header, rows, "l" + "r" * (len(header) - 1)),
        encoding="utf-8",
    )
    (out / "plausibility.md").write_text(
        _markdown(["Method", "Token F1", "Rationale AUPRC", "Weight ρ", "Sign agree.", "Off-rationale ↓"], md_rows),
        encoding="utf-8",
    )
    LOGGER.info("wrote plausibility.tex / .md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--skip-heatmap", action="store_true",
                        help="skip fig8, which needs to load the model and run explainers")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    results_dir = Path(cfg.get_path("experiment.results_dir", "results"))
    metrics_dir = results_dir / "metrics"
    fig_dir = results_dir / "figures"
    table_dir = results_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    style()

    evaluation = load(metrics_dir / "evaluation.json")
    faithfulness = load(metrics_dir / "faithfulness.json")
    stability = load(metrics_dir / "stability.json")
    plausibility = load(metrics_dir / "plausibility.json")

    if evaluation:
        fig_roc_pr(evaluation, fig_dir)
        fig_model_comparison(evaluation, fig_dir)
        fig_calibration(evaluation, fig_dir)
        table_main(evaluation, table_dir)
    if faithfulness:
        fig_deletion_insertion(faithfulness, fig_dir)
        fig_faithfulness_bars(faithfulness, fig_dir)
        fig_cost_vs_faithfulness(faithfulness, fig_dir)
        table_faithfulness(faithfulness, table_dir)
    if stability:
        fig_stability(stability, fig_dir)
        table_stability(stability, table_dir)
    if plausibility:
        table_plausibility(plausibility, table_dir)
    if not args.skip_heatmap:
        try:
            fig_attention_heatmap(cfg, results_dir, fig_dir)
        except Exception as exc:
            LOGGER.warning("heatmap figure failed: %s", exc)

    LOGGER.info("figures -> %s", fig_dir)
    LOGGER.info("tables  -> %s", table_dir)


if __name__ == "__main__":
    main()
