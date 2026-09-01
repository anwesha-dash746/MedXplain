"""Shared fixtures.

A real BERT checkpoint is far too slow to load in a unit test, so the tests use a
tiny randomly-initialised transformer with the same interface. That keeps the
suite honest about *shapes, offsets, signs and payload contracts* — which is
where the bugs actually live — without pretending to test model quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A 2-layer BERT with a real WordPiece tokenizer, saved as a MedXplain checkpoint."""
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from medxplain.models.transformer import ClinicalRiskModel, ModelConfig

    directory = tmp_path_factory.mktemp("tiny_model")

    vocab = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "the", "patient", "was", "in", "septic", "shock", "and", "required",
        "vasopressor", "support", "hospital", "course", "stable", "discharged",
        "home", "with", "follow", "up", "no", "complications", "lactate", "of",
        "intubated", "for", "respiratory", "failure", "comfort", "measures",
        "only", "hospice", "chf", "copd", "dementia", "##s", "##ed", "##ing",
        "12", "7", ".", ",", ":", "-",
    ]
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
    )
    encoder = BertModel(config)

    model_cfg = ModelConfig(name="tiny", max_length=64, freeze_layers=0)
    model = ClinicalRiskModel(model_cfg, encoder=encoder, hidden_size=config.hidden_size)

    # Bias the head so predictions are not exactly 0.5, which would hide sign bugs.
    with torch.no_grad():
        model.classifier.bias.copy_(torch.tensor([0.4, -0.4]))

    model.save(directory)
    tokenizer.save_pretrained(directory)

    # ClinicalRiskModel.load() resolves the backbone by hf_id; point it at this dir
    import json

    config_path = directory / "medxplain_config.json"
    payload = json.loads(config_path.read_text())
    payload["name"] = str(directory)
    config_path.write_text(json.dumps(payload))
    config.save_pretrained(directory)
    encoder.save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_model(tiny_model_dir):
    from medxplain.models.transformer import ClinicalRiskModel

    model, tokenizer = ClinicalRiskModel.load(tiny_model_dir, device="cpu")
    return model, tokenizer


@pytest.fixture
def sample_note() -> str:
    return (
        "Hospital Course:\n"
        "The patient was in septic shock and required vasopressor support. "
        "Lactate of 7. Intubated for respiratory failure.\n\n"
        "Discharge Disposition:\n"
        "Comfort measures only. Hospice.\n"
    )
