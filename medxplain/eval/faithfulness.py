"""Faithfulness metrics for explanations (ERASER-style).

An explanation is *faithful* if it reflects what the model actually did, as
opposed to merely looking sensible. The two headline measures come from DeYoung
et al. (2020):

**Comprehensiveness** — remove the top-k most important tokens and measure how
far the model's confidence in its own prediction falls. If the explanation
identified the evidence, removing it should hurt a lot. Higher is better.

    comp = p(y | x) - p(y | x \\ top-k)

**Sufficiency** — keep *only* the top-k tokens and measure how much confidence
survives. If the explanation captured the evidence, the retained fragment alone
should nearly reproduce the prediction. Reported as a *drop*, so lower is
better, and we also report ``1 - sufficiency`` where a higher-is-better framing
reads more naturally in a table.

    suff = p(y | x) - p(y | top-k only)

Both are averaged over several k (bins of 1/5/10/20/50% of tokens) exactly as
ERASER does, since a single k is easy to cherry-pick.

Three details that matter for the numbers being defensible
----------------------------------------------------------
1.  **Removal means masking, not deletion.** Deleting tokens shortens the
    sequence and shifts every position, so part of the measured confidence drop
    is a length artefact rather than evidence removal. We replace with the
    tokenizer's mask token and keep the length fixed.
2.  **A random-explanation control is computed alongside.** Masking any 20% of a
    note lowers confidence somewhat. The interesting quantity is how much more
    the *explanation's* tokens hurt than random tokens do, so every result
    carries the random baseline and the difference.
3.  **Ranking uses signed scores for the predicted class.** For a positive
    prediction the most important tokens are the most positive ones; ranking by
    absolute value would mix in the strongest evidence *against* the prediction
    and understate comprehensiveness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from medxplain.explain.payload import Explanation

#: ERASER's default bins, as fractions of the note's tokens.
DEFAULT_K_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50)


@dataclass
class FaithfulnessConfig:
    k_fractions: tuple[float, ...] = DEFAULT_K_FRACTIONS
    removal: str = "mask"  # mask | delete | unk
    rank_by: str = "signed"  # signed | absolute
    n_random_controls: int = 3
    batch_size: int = 16
    max_length: int = 512
    seed: int = 0
    #: number of points on the deletion/insertion curves
    curve_steps: int = 11
    extra: dict = field(default_factory=dict)


class FaithfulnessEvaluator:
    """Computes comprehensiveness, sufficiency and deletion/insertion curves."""

    def __init__(self, model, tokenizer, cfg: FaithfulnessConfig | None = None, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or FaithfulnessConfig()
        self.device = device
        self.mask_id = (
            tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
        )

    # -- model access ------------------------------------------------------------------

    @torch.no_grad()
    def _probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, input_ids.size(0), self.cfg.batch_size):
            logits = self.model(
                input_ids[i : i + self.cfg.batch_size].to(self.device),
                attention_mask[i : i + self.cfg.batch_size].to(self.device),
            )["logits"]
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    # -- perturbation ------------------------------------------------------------------

    def _apply_removal(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, positions: Sequence[int], keep_only: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask out (or keep only) ``positions``, preserving special tokens."""
        ids = input_ids.clone()
        mask = attention_mask.clone()
        special = set(self.tokenizer.all_special_ids)
        content = [
            i for i in range(ids.size(1))
            if int(ids[0, i]) not in special and int(mask[0, i]) == 1
        ]
        targets = (
            [i for i in content if i not in set(positions)] if keep_only else list(positions)
        )

        if self.cfg.removal == "delete":
            keep = [i for i in range(ids.size(1)) if i not in set(targets)]
            ids = ids[:, keep]
            mask = mask[:, keep]
        else:
            fill = self.mask_id if self.cfg.removal == "mask" else self.tokenizer.unk_token_id
            for pos in targets:
                ids[0, pos] = fill
        return ids, mask

    # -- ranking -----------------------------------------------------------------------

    def _ranked_positions(self, explanation: Explanation, predicted: int) -> list[int]:
        """Token positions ordered most- to least-important for the prediction."""
        tokens = explanation.tokens
        scores = np.asarray([t.score for t in tokens], dtype=float)
        if self.cfg.rank_by == "absolute":
            key = np.abs(scores)
        else:
            # for a negative prediction, evidence *for* it is the most negative score
            key = scores if predicted == 1 else -scores
        order = np.argsort(-key)
        return [tokens[i].index for i in order]

    # -- headline metrics --------------------------------------------------------------

    def evaluate(self, text: str, explanation: Explanation) -> dict:
        """Comprehensiveness/sufficiency for one note, with the random control."""
        from medxplain.explain.payload import encode_with_offsets

        enc = encode_with_offsets(self.tokenizer, text, self.cfg.max_length)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        base_probs = self._probs(input_ids, attention_mask)[0]
        predicted = int(base_probs.argmax())
        p_base = float(base_probs[predicted])

        ranked = self._ranked_positions(explanation, predicted)
        n_content = len(ranked)
        if n_content == 0:
            return {"error": "no content tokens", "n_tokens": 0}

        rng = np.random.default_rng(self.cfg.seed)
        comp_terms, suff_terms, comp_rand_terms, suff_rand_terms = [], [], [], []
        per_k = []

        for frac in self.cfg.k_fractions:
            k = max(1, int(round(frac * n_content)))
            top = ranked[:k]

            ids_c, mask_c = self._apply_removal(input_ids, attention_mask, top, keep_only=False)
            ids_s, mask_s = self._apply_removal(input_ids, attention_mask, top, keep_only=True)
            probs = self._probs(
                torch.cat([ids_c, ids_s], dim=0), torch.cat([mask_c, mask_s], dim=0)
            )
            p_comp, p_suff = float(probs[0, predicted]), float(probs[1, predicted])

            rand_comp, rand_suff = [], []
            for r in range(self.cfg.n_random_controls):
                choice = rng.choice(ranked, size=k, replace=False).tolist()
                ids_rc, mask_rc = self._apply_removal(input_ids, attention_mask, choice, keep_only=False)
                ids_rs, mask_rs = self._apply_removal(input_ids, attention_mask, choice, keep_only=True)
                rp = self._probs(
                    torch.cat([ids_rc, ids_rs], dim=0), torch.cat([mask_rc, mask_rs], dim=0)
                )
                rand_comp.append(float(rp[0, predicted]))
                rand_suff.append(float(rp[1, predicted]))

            comp = p_base - p_comp
            suff = p_base - p_suff
            comp_r = p_base - float(np.mean(rand_comp))
            suff_r = p_base - float(np.mean(rand_suff))

            comp_terms.append(comp)
            suff_terms.append(suff)
            comp_rand_terms.append(comp_r)
            suff_rand_terms.append(suff_r)
            per_k.append(
                {
                    "fraction": frac,
                    "k": k,
                    "p_after_removal": p_comp,
                    "p_top_k_only": p_suff,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "comprehensiveness_random": comp_r,
                    "sufficiency_random": suff_r,
                }
            )

        return {
            "method": explanation.method,
            "n_tokens": n_content,
            "p_base": p_base,
            "predicted_label": predicted,
            "comprehensiveness": float(np.mean(comp_terms)),
            "sufficiency": float(np.mean(suff_terms)),
            "comprehensiveness_random": float(np.mean(comp_rand_terms)),
            "sufficiency_random": float(np.mean(suff_rand_terms)),
            # the quantities that actually separate methods: how much better than chance
            "comprehensiveness_over_random": float(np.mean(comp_terms) - np.mean(comp_rand_terms)),
            "sufficiency_over_random": float(np.mean(suff_rand_terms) - np.mean(suff_terms)),
            "per_k": per_k,
        }

    # -- curves ------------------------------------------------------------------------

    def deletion_insertion_curves(self, text: str, explanation: Explanation) -> dict:
        """Confidence as tokens are progressively removed (deletion) or added (insertion).

        The summary statistic is the area under each curve: a faithful
        explanation gives a *low* deletion AUC (confidence collapses quickly as
        its top tokens go) and a *high* insertion AUC (confidence recovers
        quickly as they come back).
        """
        from medxplain.explain.payload import encode_with_offsets

        enc = encode_with_offsets(self.tokenizer, text, self.cfg.max_length)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        base_probs = self._probs(input_ids, attention_mask)[0]
        predicted = int(base_probs.argmax())

        ranked = self._ranked_positions(explanation, predicted)
        n = len(ranked)
        if n == 0:
            return {"error": "no content tokens"}

        fractions = np.linspace(0.0, 1.0, self.cfg.curve_steps)
        del_ids, del_masks, ins_ids, ins_masks = [], [], [], []
        for frac in fractions:
            k = int(round(frac * n))
            top = ranked[:k]
            a, ma = self._apply_removal(input_ids, attention_mask, top, keep_only=False)
            b, mb = self._apply_removal(input_ids, attention_mask, top, keep_only=True)
            del_ids.append(a)
            del_masks.append(ma)
            ins_ids.append(b)
            ins_masks.append(mb)

        width = max(t.size(1) for t in del_ids + ins_ids)

        def pad(tensors: list[torch.Tensor], value: int) -> torch.Tensor:
            out = torch.full((len(tensors), width), value, dtype=torch.long)
            for i, t in enumerate(tensors):
                out[i, : t.size(1)] = t[0]
            return out

        deletion = self._probs(pad(del_ids, self.tokenizer.pad_token_id), pad(del_masks, 0))[:, predicted]
        insertion = self._probs(pad(ins_ids, self.tokenizer.pad_token_id), pad(ins_masks, 0))[:, predicted]

        return {
            "method": explanation.method,
            "fractions": fractions.tolist(),
            "deletion": deletion.tolist(),
            "insertion": insertion.tolist(),
            "deletion_auc": float(np.trapezoid(deletion, fractions)),
            "insertion_auc": float(np.trapezoid(insertion, fractions)),
            "predicted_label": predicted,
        }


def summarise_faithfulness(results: list[dict]) -> dict:
    """Mean +/- std over notes, for one (model, explanation method) pair."""
    keys = (
        "comprehensiveness",
        "sufficiency",
        "comprehensiveness_random",
        "sufficiency_random",
        "comprehensiveness_over_random",
        "sufficiency_over_random",
    )
    usable = [r for r in results if "error" not in r]
    out: dict = {"n_notes": len(usable)}
    for key in keys:
        vals = np.asarray([r[key] for r in usable if key in r], dtype=float)
        if vals.size:
            out[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "sem": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
            }
    return out


def summarise_curves(curves: list[dict]) -> dict:
    """Average deletion/insertion curves across notes for plotting."""
    usable = [c for c in curves if "error" not in c]
    if not usable:
        return {}
    fractions = usable[0]["fractions"]
    deletion = np.vstack([c["deletion"] for c in usable])
    insertion = np.vstack([c["insertion"] for c in usable])
    return {
        "n_notes": len(usable),
        "fractions": fractions,
        "deletion_mean": deletion.mean(axis=0).tolist(),
        "deletion_std": deletion.std(axis=0).tolist(),
        "insertion_mean": insertion.mean(axis=0).tolist(),
        "insertion_std": insertion.std(axis=0).tolist(),
        "deletion_auc": {
            "mean": float(np.mean([c["deletion_auc"] for c in usable])),
            "std": float(np.std([c["deletion_auc"] for c in usable], ddof=1)) if len(usable) > 1 else 0.0,
        },
        "insertion_auc": {
            "mean": float(np.mean([c["insertion_auc"] for c in usable])),
            "std": float(np.std([c["insertion_auc"] for c in usable], ddof=1)) if len(usable) > 1 else 0.0,
        },
    }
