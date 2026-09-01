"""Loss functions for the rare-event setting.

Mortality and 30-day readmission are both minority-class problems (roughly
10-20% positive), so plain cross-entropy trained to convergence tends to produce
a model that is well-calibrated on the majority and useless at the decision
threshold a clinician cares about. Three standard remedies are implemented and
selected by config, so the choice is an experimental variable rather than a
hidden default:

* ``ce``            - unweighted cross-entropy (the control condition)
* ``weighted_ce``   - inverse-frequency class weights
* ``focal``         - focal loss, which down-weights easy negatives
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) for binary/multi-class logits.

    ``gamma`` controls how aggressively well-classified examples are discounted;
    ``alpha`` is an optional per-class weight applied on top, which lets focal
    loss and class weighting be combined rather than treated as alternatives.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, target)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    kind: str = "weighted_ce",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Instantiate the configured loss."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if kind == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f"unknown loss: {kind!r} (expected ce, weighted_ce or focal)")
