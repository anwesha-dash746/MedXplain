"""Explanation methods, all returning the same :class:`Explanation` payload."""

from __future__ import annotations

from medxplain.explain.attention import AttentionExplainer, attention_rollout
from medxplain.explain.gradients import IntegratedGradientsExplainer
from medxplain.explain.payload import (
    Explanation,
    TokenAttribution,
    WordAttribution,
    aggregate_to_words,
    merge_explanations,
    normalise,
)
from medxplain.explain.shap_explainer import OcclusionExplainer, ShapExplainer

#: Methods exposed to the API, the frontend toggle, and the evaluation scripts.
EXPLAINER_REGISTRY: dict[str, type] = {
    "attention_raw": AttentionExplainer,
    "attention_rollout": AttentionExplainer,
    "attention_grad": AttentionExplainer,
    "shap": ShapExplainer,
    "occlusion": OcclusionExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
}

#: Short descriptions surfaced as tooltips in the UI.
METHOD_DESCRIPTIONS: dict[str, str] = {
    "attention_raw": (
        "Mean attention from the [CLS] position in the final layer. Fast, but a single "
        "layer's attention is a weak guide to what the model actually used."
    ),
    "attention_rollout": (
        "Attention rolled out across all layers with residual connections accounted for "
        "(Abnar & Zuidema, 2020). Shows how much of each word reaches the classifier's "
        "summary representation. Class-agnostic: it shows what was used, not which way."
    ),
    "attention_grad": (
        "Attention weighted by the gradient of the predicted class, so the map is specific "
        "to the risk being predicted rather than shared across classes."
    ),
    "shap": (
        "Shapley values via partition SHAP. Signed, additive contributions in log-odds: "
        "a word's value is how much it moved the risk relative to a masked baseline. "
        "The most expensive method here."
    ),
    "occlusion": (
        "Leave-one-word-out: the drop in predicted risk when each word is masked. "
        "Directly measures the model's dependence on each word, one forward pass per word."
    ),
    "integrated_gradients": (
        "Integrated Gradients (Sundararajan et al., 2017). Signed attributions that sum to "
        "the change in prediction from a masked baseline, with a built-in completeness check."
    ),
}


def build_explainer(method: str, model, tokenizer, device: str = "cpu", **kwargs):
    """Instantiate an explainer by registry name."""
    if method not in EXPLAINER_REGISTRY:
        raise ValueError(
            f"unknown explanation method {method!r}; expected one of {sorted(EXPLAINER_REGISTRY)}"
        )
    cls = EXPLAINER_REGISTRY[method]
    if cls is AttentionExplainer:
        kwargs.setdefault("method", method)
    return cls(model, tokenizer, device=device, **kwargs)


__all__ = [
    "AttentionExplainer",
    "ShapExplainer",
    "OcclusionExplainer",
    "IntegratedGradientsExplainer",
    "Explanation",
    "TokenAttribution",
    "WordAttribution",
    "EXPLAINER_REGISTRY",
    "METHOD_DESCRIPTIONS",
    "build_explainer",
    "attention_rollout",
    "aggregate_to_words",
    "merge_explanations",
    "normalise",
]
