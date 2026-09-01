"""Integrated Gradients for the clinical risk classifier.

Included as a third explanation family because attention and SHAP occupy the two
ends of the cost spectrum and neither is gradient-based. IG (Sundararajan et
al., 2017) sits in between: it needs ``n_steps`` forward/backward passes
regardless of note length, it is signed and class-specific, and it satisfies
completeness — the attributions sum to the difference between the prediction at
the input and at the baseline, which gives a built-in correctness check
(``convergence_delta``) that attention has no analogue for.

The baseline is the all-``[MASK]`` sequence with the special tokens kept in
place, rather than the all-zeros embedding used in vision. A zero embedding is
not a point in the model's input distribution for text, and using it makes
attributions dominated by how far each token's embedding happens to sit from the
origin.
"""

from __future__ import annotations

import numpy as np
import torch

from medxplain.explain.payload import (
    Explanation,
    aggregate_to_words,
    build_token_attributions,
    encode_with_offsets,
    normalise,
)


class IntegratedGradientsExplainer:
    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 512,
        n_steps: int = 32,
        internal_batch_size: int = 8,
        device: str = "cpu",
        baseline: str = "mask",  # mask | pad | zero
        word_reduction: str = "sum",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.device = device
        self.baseline = baseline
        self.word_reduction = word_reduction

    def _baseline_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer
        if self.baseline == "pad":
            fill = tok.pad_token_id
        elif self.baseline == "zero":
            return torch.zeros_like(input_ids)
        else:
            fill = tok.mask_token_id if tok.mask_token_id is not None else tok.pad_token_id
        baseline = torch.full_like(input_ids, fill)
        # keep [CLS]/[SEP] so the baseline is a well-formed sequence
        special = set(tok.all_special_ids)
        for pos, token_id in enumerate(input_ids[0].tolist()):
            if token_id in special:
                baseline[0, pos] = token_id
        return baseline

    def explain(self, text: str, target: int | None = None) -> Explanation:
        enc = encode_with_offsets(self.tokenizer, text, self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            input_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(self._baseline_ids(input_ids))
            probs = self.model.predict_proba(input_ids, attention_mask)[0]
        predicted = int(probs.argmax())
        target = predicted if target is None else target

        # Riemann midpoint rule over the straight-line path baseline -> input
        alphas = (torch.arange(self.n_steps, dtype=torch.float32, device=self.device) + 0.5) / self.n_steps
        delta = input_embeds - baseline_embeds
        total_grad = torch.zeros_like(input_embeds)

        self.model.eval()
        for start in range(0, self.n_steps, self.internal_batch_size):
            chunk = alphas[start : start + self.internal_batch_size]
            scaled = baseline_embeds + chunk.view(-1, 1, 1) * delta
            scaled = scaled.detach().requires_grad_(True)
            mask = attention_mask.expand(scaled.size(0), -1)

            logits = self.model(
                input_ids=None, attention_mask=mask, inputs_embeds=scaled
            )["logits"]
            selected = logits[:, target].sum()
            grads = torch.autograd.grad(selected, scaled)[0]
            total_grad += grads.sum(dim=0, keepdim=True)

        avg_grad = total_grad / self.n_steps
        attributions = (delta * avg_grad).sum(dim=-1)[0].detach().cpu().numpy()

        # completeness check: sum(attributions) should approximate f(x) - f(baseline)
        with torch.no_grad():
            f_input = float(self.model(input_ids, attention_mask)["logits"][0, target])
            base_logits = self.model(
                input_ids=None, attention_mask=attention_mask, inputs_embeds=baseline_embeds
            )["logits"]
            f_baseline = float(base_logits[0, target])
        convergence_delta = float(attributions.sum() - (f_input - f_baseline))

        mask_np = attention_mask[0].cpu().numpy().astype(bool)
        attributions = np.where(mask_np, attributions, 0.0)

        tokens = build_token_attributions(
            enc["tokens"], normalise(attributions, "max_abs"), enc["offsets"], drop_special=True
        )
        words = aggregate_to_words(text, tokens, reduction=self.word_reduction)

        return Explanation(
            method="integrated_gradients",
            tokens=tokens,
            words=words,
            probability=float(probs[1]),
            predicted_label=predicted,
            meta={
                "n_steps": self.n_steps,
                "baseline": self.baseline,
                "convergence_delta": convergence_delta,
                "relative_convergence_error": abs(convergence_delta) / (abs(f_input - f_baseline) + 1e-9),
                "n_model_evaluations": self.n_steps,
                "truncated": enc["truncated"],
                "signed": True,
            },
        )
