"""Transformer classifier for clinical risk, with attention exposed for explanation.

The model is a thin wrapper over a HuggingFace encoder plus a classification
head. It is deliberately *not* ``AutoModelForSequenceClassification``, for two
reasons:

1.  We need ``output_attentions`` plumbed through inference without paying for
    it during training, and we need the raw per-layer attention tensors for
    attention rollout (see :mod:`medxplain.explain.attention`).
2.  Long notes need window pooling at inference: a discharge summary that
    exceeds the encoder's 512-token limit is scored by sliding windows whose
    logits are pooled, rather than silently truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

#: Short names used throughout configs, scripts and the API, mapped to HF ids.
MODEL_REGISTRY: dict[str, str] = {
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bert": "bert-base-uncased",
}

#: Human-facing labels for the UI and the results tables.
MODEL_DISPLAY: dict[str, str] = {
    "clinicalbert": "Bio_ClinicalBERT",
    "biobert": "BioBERT v1.2",
    "pubmedbert": "PubMedBERT (BiomedBERT)",
    "bert": "BERT-base (no domain adaptation)",
}


@dataclass
class ModelConfig:
    """Everything needed to rebuild a trained model from disk."""

    name: str = "clinicalbert"
    max_length: int = 512
    num_labels: int = 2
    dropout: float = 0.1
    pooling: str = "cls"  # cls | mean | cls_mean
    truncation_strategy: str = "head_tail"
    head_fraction: float = 0.35
    #: freeze the bottom N encoder layers (0 = full fine-tuning)
    freeze_layers: int = 0
    label_names: tuple[str, str] = ("low risk", "high risk")
    extra: dict = field(default_factory=dict)

    @property
    def hf_id(self) -> str:
        return MODEL_REGISTRY.get(self.name, self.name)

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY.get(self.name, self.name)


def load_backbone(cfg: ModelConfig):
    """Load encoder + tokenizer for ``cfg`` (cached by HuggingFace on disk)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    hf_config = AutoConfig.from_pretrained(cfg.hf_id)
    encoder = AutoModel.from_pretrained(cfg.hf_id, config=hf_config)
    return encoder, tokenizer, hf_config


class ClinicalRiskModel(nn.Module):
    """Encoder + pooling + linear head, with optional attention outputs."""

    def __init__(self, cfg: ModelConfig, encoder: nn.Module | None = None, hidden_size: int | None = None):
        super().__init__()
        self.cfg = cfg
        if encoder is None:
            encoder, _, hf_config = load_backbone(cfg)
            hidden_size = hf_config.hidden_size
        if hidden_size is None:
            hidden_size = encoder.config.hidden_size
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        head_in = hidden_size * (2 if cfg.pooling == "cls_mean" else 1)
        self.classifier = nn.Linear(head_in, cfg.num_labels)
        self._apply_freezing()

    # -- construction helpers ---------------------------------------------------------

    def _apply_freezing(self) -> None:
        if self.cfg.freeze_layers <= 0:
            return
        embeddings = getattr(self.encoder, "embeddings", None)
        if embeddings is not None:
            for p in embeddings.parameters():
                p.requires_grad_(False)
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is not None:
            for layer in layers[: self.cfg.freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)

    # -- forward ----------------------------------------------------------------------

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = hidden[:, 0]
        if self.cfg.pooling == "cls":
            return cls
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.cfg.pooling == "mean":
            return mean
        return torch.cat([cls, mean], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict:
        kwargs = {
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "return_dict": True,
        }
        if inputs_embeds is not None:
            # gradient-based attribution paths (integrated gradients) feed embeddings
            outputs = self.encoder(inputs_embeds=inputs_embeds, **kwargs)
        else:
            outputs = self.encoder(input_ids=input_ids, **kwargs)

        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        result = {"logits": logits, "pooled": pooled}
        if output_attentions:
            # tuple of (batch, heads, seq, seq), one per layer
            result["attentions"] = outputs.attentions
        return result

    # -- convenience ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        logits = self.forward(input_ids, attention_mask)["logits"]
        return F.softmax(logits, dim=-1)

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    # -- persistence ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "pytorch_model.bin")
        from medxplain.utils import write_json

        write_json(
            directory / "medxplain_config.json",
            {
                "name": self.cfg.name,
                "hf_id": self.cfg.hf_id,
                "max_length": self.cfg.max_length,
                "num_labels": self.cfg.num_labels,
                "dropout": self.cfg.dropout,
                "pooling": self.cfg.pooling,
                "truncation_strategy": self.cfg.truncation_strategy,
                "head_fraction": self.cfg.head_fraction,
                "freeze_layers": self.cfg.freeze_layers,
                "label_names": list(self.cfg.label_names),
                "extra": self.cfg.extra,
            },
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> tuple["ClinicalRiskModel", object]:
        from medxplain.utils import read_json

        directory = Path(directory)
        raw = read_json(directory / "medxplain_config.json")
        cfg = ModelConfig(
            name=raw["name"],
            max_length=raw["max_length"],
            num_labels=raw["num_labels"],
            dropout=raw["dropout"],
            pooling=raw["pooling"],
            truncation_strategy=raw.get("truncation_strategy", "head_tail"),
            head_fraction=raw.get("head_fraction", 0.35),
            freeze_layers=raw.get("freeze_layers", 0),
            label_names=tuple(raw.get("label_names", ("low risk", "high risk"))),
            extra=raw.get("extra", {}),
        )
        encoder, tokenizer, hf_config = load_backbone(cfg)
        model = cls(cfg, encoder=encoder, hidden_size=hf_config.hidden_size)
        state = torch.load(directory / "pytorch_model.bin", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, tokenizer


# --------------------------------------------------------------------------------------
# Long-document inference
# --------------------------------------------------------------------------------------


@torch.no_grad()
def predict_long_note(
    model: ClinicalRiskModel,
    tokenizer,
    text: str,
    max_length: int = 512,
    stride: int = 128,
    pooling: str = "max",
    device: str = "cpu",
    max_chunks: int = 8,
) -> dict:
    """Score a note that may exceed the encoder's context window.

    The note is split into overlapping windows; each window is scored, and the
    per-window positive-class probabilities are pooled. ``max`` pooling matches
    the clinical semantics of the task — a single window describing a cardiac
    arrest should drive the risk up regardless of how much routine text
    surrounds it — while ``mean`` is available for comparison.
    """
    from medxplain.data.preprocess import chunk_text

    chunks = chunk_text(text, tokenizer, max_length=max_length, stride=stride, max_chunks=max_chunks)
    max_len = max(len(c["input_ids"]) for c in chunks)
    input_ids = torch.full((len(chunks), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(chunks), max_len), dtype=torch.long)
    for i, c in enumerate(chunks):
        n = len(c["input_ids"])
        input_ids[i, :n] = torch.tensor(c["input_ids"])
        attention[i, :n] = 1

    model.eval()
    probs = model.predict_proba(input_ids.to(device), attention.to(device))[:, 1].cpu()

    if pooling == "mean":
        score = float(probs.mean())
    elif pooling == "max":
        score = float(probs.max())
    elif pooling == "noisy_or":
        score = float(1.0 - torch.prod(1.0 - probs))
    else:
        raise ValueError(f"unknown chunk pooling: {pooling}")

    return {
        "probability": score,
        "n_chunks": len(chunks),
        "chunk_probabilities": [float(p) for p in probs],
        "pooling": pooling,
    }
