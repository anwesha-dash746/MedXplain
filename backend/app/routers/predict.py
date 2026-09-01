"""/predict - the endpoint the clinician-facing UI calls."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas import (
    BatchPredictRequest,
    ExplanationPayload,
    PredictRequest,
    PredictResponse,
    WordScore,
)
from backend.app.services.inference import InferenceService, risk_band
from medxplain.explain.payload import Explanation

router = APIRouter(tags=["prediction"])

CALIBRATION_NOTE = (
    "This probability is the model's calibrated estimate on held-out data, not a "
    "clinical determination. It is a decision-support signal and must be read "
    "alongside the note itself."
)


def _service(request: Request) -> InferenceService:
    return request.app.state.inference


def _explanation_payload(
    explanation: Explanation, compute_ms: float, top_k: int, truncated: bool
) -> ExplanationPayload:
    meta = InferenceService.method_metadata(explanation.method)
    signed = explanation.meta.get("signed", meta["signed"])

    words = [
        WordScore(text=w.text, score=w.score, start=w.start, end=w.end)
        for w in explanation.words
    ]
    if signed:
        top_positive = [
            WordScore(text=w.text, score=w.score, start=w.start, end=w.end)
            for w in explanation.top_words(top_k, signed=True) if w.score > 0
        ]
        top_negative = [
            WordScore(text=w.text, score=w.score, start=w.start, end=w.end)
            for w in explanation.top_negative_words(top_k) if w.score < 0
        ]
    else:
        # Unsigned methods have no "protective" direction to report; presenting the
        # least-attended words as protective would be an outright misreading.
        top_positive = [
            WordScore(text=w.text, score=w.score, start=w.start, end=w.end)
            for w in explanation.top_words(top_k, signed=False)
        ]
        top_negative = []

    return ExplanationPayload(
        method=explanation.method,
        description=meta["description"],
        signed=signed,
        words=words,
        top_positive=top_positive,
        top_negative=top_negative,
        n_model_evaluations=int(explanation.meta.get("n_model_evaluations", 1)),
        compute_ms=round(compute_ms, 1),
        truncated=truncated,
        meta={k: v for k, v in explanation.meta.items() if k not in ("signed",)},
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score a clinical note and explain the score",
)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    service = _service(request)

    methods = payload.compare_methods or ([payload.method] if payload.method else None)
    if methods is None and payload.explain:
        methods = [service.settings.default_method]

    try:
        outcome = service.predict(
            payload.text,
            model_key=payload.model,
            methods=methods,
            explain=payload.explain,
            max_evals=payload.max_evals,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    explanations = [
        _explanation_payload(e, ms, payload.top_k, outcome.truncated)
        for e, ms in outcome.explanations
    ]

    response = PredictResponse(
        note_id=outcome.note_id,
        model=outcome.model.key,
        model_display_name=outcome.model.display_name,
        outcome=_outcome_label(outcome.model.outcome),
        probability=outcome.probability,
        predicted_label=outcome.predicted_label,
        decision_threshold=outcome.threshold,
        risk_band=risk_band(outcome.probability, outcome.threshold),
        calibration_note=CALIBRATION_NOTE,
        explanations=explanations,
        n_chars=outcome.n_chars,
        n_tokens=outcome.n_tokens,
        truncated=outcome.truncated,
        inference_ms=round(outcome.inference_ms, 1),
        warnings=outcome.warnings,
    )

    request.app.state.history.add(
        {
            "note_id": response.note_id,
            "timestamp": time.time(),
            "model": response.model,
            "method": explanations[0].method if explanations else "none",
            "probability": response.probability,
            "predicted_label": response.predicted_label,
            "preview": payload.text.strip()[:140],
            "n_chars": response.n_chars,
        }
    )
    return response


@router.post("/predict/batch", summary="Score several notes without explanations")
def predict_batch(payload: BatchPredictRequest, request: Request) -> dict:
    service = _service(request)
    results, errors = [], []
    for i, note in enumerate(payload.notes):
        try:
            outcome = service.predict(note, model_key=payload.model, explain=payload.explain)
            results.append(
                {
                    "index": i,
                    "note_id": outcome.note_id,
                    "probability": outcome.probability,
                    "predicted_label": outcome.predicted_label,
                    "risk_band": risk_band(outcome.probability, outcome.threshold),
                }
            )
        except (ValueError, FileNotFoundError) as exc:
            errors.append({"index": i, "error": str(exc)})
    return {"results": results, "errors": errors, "n": len(payload.notes)}


def _outcome_label(key: str) -> str:
    return {
        "mortality": "in-hospital mortality",
        "readmission_30d": "30-day readmission",
    }.get(key, key)
