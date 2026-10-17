"""Attention-based explanations.

Raw last-layer attention is the method most clinical XAI papers reach for and
the one most criticised in the literature ("Attention is not Explanation",
Jain & Wallace 2019). Rather than pick a side, this module implements three
variants so the paper can report the difference empirically on the same model
and the same faithfulness metrics:

* ``attention_raw``     - mean over heads of the last layer's [CLS] row. The
  weak baseline; included precisely because it is what most systems ship.
* ``attention_rollout`` - Abnar & Zuidema (2020). Attention is averaged over
  heads, mixed with the identity to account for residual connections, and
  multiplied across layers, so the score reflects how much each input token
  actually reaches the [CLS] representation rather than one layer's local view.
* ``attention_grad``    - gradient-weighted attention (Chefer et al., 2021,
  simplified): heads are weighted by the gradient of the target logit before
  rollout, which makes the explanation class-specific. Plain rollout is
  class-agnostic — it produces the same map for the positive and negative class,
  which is a real limitation worth stating.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from medxplain.explain.payload import (
    Explanation,
    aggregate_to_words,
    build_token_attributions,
    encode_with_offsets,
    head_fraction_of,
    normalise,
)

AttentionMethod = Literal["attention_raw", "attention_rollout", "attention_grad"]


def _head_fusion(attn: torch.Tensor, fusion: str = "mean") -> torch.Tensor:
    """Collapse the head dimension of a (heads, seq, seq) attention tensor."""
    if fusion == "mean":
        return attn.mean(dim=0)
    if fusion == "max":
        return attn.max(dim=0).values
    if fusion == "min":
        return attn.min(dim=0).values
    raise ValueError(f"unknown head fusion: {fusion}")


def attention_rollout(
    attentions: tuple[torch.Tensor, ...],
    head_fusion: str = "mean",
    add_residual: bool = True,
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Attention rollout over all layers; returns the [CLS] row (seq,).

    ``attentions`` is the HuggingFace tuple of ``(batch, heads, seq, seq)``
    tensors, one per layer. Only the first item of the batch is used.

    ``discard_ratio`` optionally zeroes the weakest fraction of connections at
    each layer before multiplying, which Abnar & Zuidema note sharpens the map;
    it defaults to 0 so the plain published method is the default.
    """
    result: torch.Tensor | None = None
    for layer_attn in attentions:
        attn = layer_attn[0].detach()  # (heads, seq, seq)
        fused = _head_fusion(attn, head_fusion).to(torch.float64)

        if discard_ratio > 0:
            flat = fused.flatten()
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                cutoff = torch.kthvalue(flat, k).values
                fused = torch.where(fused < cutoff, torch.zeros_like(fused), fused)

        if add_residual:
            eye = torch.eye(fused.size(-1), dtype=fused.dtype, device=fused.device)
            fused = 0.5 * fused + 0.5 * eye

        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        result = fused if result is None else fused @ result

    assert result is not None, "no attention layers provided"
    return result[0].cpu().numpy()  # [CLS] row


def gradient_weighted_rollout(
    attentions: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    add_residual: bool = True,
) -> np.ndarray:
    """Class-specific rollout: weight each head by its gradient before fusing.

    Negative contributions are clipped at zero, following Chefer et al. — a
    head that pushes *away* from the target class should not be shown as
    evidence *for* it.
    """
    result: torch.Tensor | None = None
    for attn_layer, grad_layer in zip(attentions, gradients):
        attn = attn_layer[0].detach().to(torch.float64)
        grad = grad_layer[0].detach().to(torch.float64)
        weighted = (grad * attn).clamp(min=0).mean(dim=0)

        if add_residual:
            eye = torch.eye(weighted.size(-1), dtype=weighted.dtype, device=weighted.device)
            weighted = 0.5 * weighted + 0.5 * eye

        weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        result = weighted if result is None else weighted @ result

    assert result is not None, "no attention layers provided"
    return result[0].cpu().numpy()


class AttentionExplainer:
    """Produces :class:`Explanation` objects from a model's attention maps."""

    def __init__(
        self,
        model,
        tokenizer,
        method: AttentionMethod = "attention_rollout",
        max_length: int = 512,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        device: str = "cpu",
        word_reduction: str = "sum",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.max_length = max_length
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.device = device
        self.word_reduction = word_reduction

    def explain(self, text: str, target: int | None = None) -> Explanation:
        enc = encode_with_offsets(self.tokenizer, text, self.max_length, head_fraction_of(self.model))
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        if self.method == "attention_grad":
            scores, probability, predicted = self._grad_scores(input_ids, attention_mask, target)
        else:
            with torch.no_grad():
                out = self.model(input_ids, attention_mask, output_attentions=True)
            probs = torch.softmax(out["logits"], dim=-1)[0]
            predicted = int(probs.argmax())
            probability = float(probs[1])
            if self.method == "attention_raw":
                last = out["attentions"][-1][0]  # (heads, seq, seq)
                scores = _head_fusion(last, self.head_fusion)[0].cpu().numpy()
            else:
                scores = attention_rollout(
                    out["attentions"],
                    head_fusion=self.head_fusion,
                    discard_ratio=self.discard_ratio,
                )

        # attention mass on padding/special positions is not evidence about the note
        mask = attention_mask[0].cpu().numpy().astype(bool)
        scores = np.where(mask, scores, 0.0)

        tokens = build_token_attributions(
            enc["tokens"], normalise(scores, "max_abs"), enc["offsets"], drop_special=True
        )
        words = aggregate_to_words(text, tokens, reduction=self.word_reduction)

        return Explanation(
            method=self.method,
            tokens=tokens,
            words=words,
            probability=probability,
            predicted_label=predicted,
            meta={
                "unread_span": enc["unread_span"],
                "head_fusion": self.head_fusion,
                "discard_ratio": self.discard_ratio,
                "truncated": enc["truncated"],
                "n_model_evaluations": 1,
                "signed": self.method == "attention_grad",
                "notes": (
                    "Attention rollout is class-agnostic: it shows what the [CLS] "
                    "representation is built from, not which class the evidence supports."
                    if self.method == "attention_rollout"
                    else ""
                ),
            },
        )

    # -- gradient-weighted variant -----------------------------------------------------

    def _grad_scores(self, input_ids, attention_mask, target: int | None):
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        out = self.model(input_ids, attention_mask, output_attentions=True)
        attentions = out["attentions"]
        for attn in attentions:
            attn.retain_grad()

        probs = torch.softmax(out["logits"], dim=-1)[0]
        predicted = int(probs.argmax())
        target = predicted if target is None else target
        positive_prob = float(probs[1].detach())
        out["logits"][0, target].backward()

        gradients = tuple(a.grad.detach() for a in attentions)
        scores = gradient_weighted_rollout(
            tuple(a.detach() for a in attentions), gradients
        )
        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()
        return scores, positive_prob, predicted
