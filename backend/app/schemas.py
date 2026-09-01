"""Request/response schemas.

These are the contract between the FastAPI backend and the Next.js frontend, and
FastAPI turns them into the OpenAPI document at ``/docs``. Field descriptions are
written for a reader who has the paper open — several of them exist to stop a
consumer from misreading a number (for example, that attention scores are
unsigned and therefore cannot mean "protective").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ExplanationMethod = Literal[
    "attention_rollout",
    "attention_raw",
    "attention_grad",
    "shap",
    "occlusion",
    "integrated_gradients",
]


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class PredictRequest(BaseModel):
    text: str = Field(..., description="The clinical note to score, as free text.")
    model: str | None = Field(
        None, description="Backbone key, e.g. 'clinicalbert'. Defaults to the server's."
    )
    method: ExplanationMethod | None = Field(
        None, description="Explanation method. Defaults to attention rollout."
    )
    explain: bool = Field(True, description="Set false for a risk score with no explanation.")
    top_k: int = Field(15, ge=1, le=100, description="How many top words to return separately.")
    max_evals: int | None = Field(
        None, ge=32, le=800,
        description="SHAP budget. Higher is more accurate and proportionally slower.",
    )
    compare_methods: list[ExplanationMethod] | None = Field(
        None, description="Optionally compute several explanations of the same prediction."
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("note text is empty")
        return value


class BatchPredictRequest(BaseModel):
    notes: list[str] = Field(..., max_length=32)
    model: str | None = None
    explain: bool = False


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class TokenScore(BaseModel):
    token: str
    score: float
    start: int
    end: int


class WordScore(BaseModel):
    text: str
    score: float
    start: int
    end: int


class ExplanationPayload(BaseModel):
    method: str
    description: str = Field(..., description="Plain-language account of what the method measures.")
    signed: bool = Field(
        ...,
        description=(
            "True if negative scores mean 'pushes risk down'. Attention-based maps are "
            "unsigned: a high score means 'used', not 'increases risk'."
        ),
    )
    words: list[WordScore]
    top_positive: list[WordScore]
    top_negative: list[WordScore]
    n_model_evaluations: int
    compute_ms: float
    truncated: bool = Field(
        False, description="True if the note exceeded the model's context window."
    )
    meta: dict = {}


class RiskBand(BaseModel):
    label: str
    lower: float
    upper: float


class PredictResponse(BaseModel):
    # 'model' is a real field name here (the backbone key); pydantic reserves the
    # model_ prefix by default, so the namespace guard is disabled explicitly.
    model_config = ConfigDict(protected_namespaces=())

    note_id: str
    model: str
    model_display_name: str
    outcome: str = Field(..., description="Which risk this score refers to, e.g. in-hospital mortality.")
    probability: float = Field(..., ge=0.0, le=1.0)
    predicted_label: int
    decision_threshold: float = Field(
        ...,
        description=(
            "Operating threshold selected on the validation split, not on test. "
            "predicted_label is probability >= this value."
        ),
    )
    risk_band: str
    calibration_note: str
    explanations: list[ExplanationPayload]
    n_chars: int
    n_tokens: int
    truncated: bool
    inference_ms: float
    warnings: list[str] = []


class ModelInfo(BaseModel):
    key: str
    display_name: str
    hf_id: str
    available: bool
    outcome: str | None = None
    test_auroc: float | None = None
    test_auprc: float | None = None
    threshold: float | None = None
    max_length: int | None = None
    seed: int | None = None
    note: str | None = None


class MethodInfo(BaseModel):
    key: str
    display_name: str
    description: str
    signed: bool
    cost: Literal["fast", "moderate", "slow"]
    typical_ms: int | None = None


class ExampleNote(BaseModel):
    note_id: str
    title: str
    text: str
    true_label: int
    outcome: str


class HistoryEntry(BaseModel):
    note_id: str
    timestamp: float
    model: str
    method: str
    probability: float
    predicted_label: int
    preview: str
    n_chars: int


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: list[str]
    models_available: list[str]
    default_model: str
    default_method: str


class ErrorResponse(BaseModel):
    detail: str
    hint: str | None = None
