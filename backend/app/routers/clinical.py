"""/clinical - the doctor-facing workspace for one submitted note.

Everything returned is either quoted from the submitted note (with offsets),
computed from historical records in the case library, or explicitly labelled
as an interpretation. Nothing is persisted: the note is processed in memory
and discarded, the same policy as /predict's in-memory history.
"""

from __future__ import annotations

import hashlib
import time

from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas import ClinicalAnalysisRequest, ClinicalAnalysisResponse
from medxplain.clinical import analyse_note
from medxplain.clinical.suggestions import generate_suggestions

router = APIRouter(tags=["clinical"])

DISCLAIMER = (
    "Research prototype. Not a medical device and not clinically validated. Summaries are "
    "keyword-based extractions from the provided note; historical findings describe what was "
    "documented in the case library, not what is effective; AI suggestions are rule-generated "
    "prompts, not treatment advice. All content must be verified by a clinician."
)


@router.post(
    "/clinical/analyze",
    response_model=ClinicalAnalysisResponse,
    summary="Case summary, documentation review, follow-up and comparable cases for one note",
)
def analyze(payload: ClinicalAnalysisRequest, request: Request) -> ClinicalAnalysisResponse:
    settings = request.app.state.settings
    text = payload.text
    stripped = text.strip()
    if len(stripped) < settings.min_note_chars:
        raise HTTPException(
            status_code=422,
            detail=f"The note is too short to analyse ({len(stripped)} characters; "
            f"at least {settings.min_note_chars} are needed).",
        )
    if len(stripped) > settings.max_note_chars:
        raise HTTPException(
            status_code=422,
            detail=f"The note is too long ({len(stripped)} characters; the limit is "
            f"{settings.max_note_chars}).",
        )

    started = time.perf_counter()
    analysis = analyse_note(text)
    try:
        similar = request.app.state.case_library.query(text)
    except Exception as exc:  # retrieval failure must not take the rest of the workspace down
        similar = {
            "available": False,
            "reason": f"Comparable-case search failed: {exc}",
            "source": request.app.state.case_library.source(),
        }

    return ClinicalAnalysisResponse(
        case_id=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        analysed_at=time.time(),
        n_chars=len(text),
        sections_found=analysis["sections_found"],
        case_summary=analysis["case_summary"],
        documentation_review=analysis["documentation_review"],
        follow_up=analysis["follow_up"],
        similar_cases=similar,
        ai_suggestions=generate_suggestions(text, analysis["documentation_review"], similar),
        peer_consult={
            # built-in tab; nothing from this note is sent to it automatically
            "available": getattr(request.app.state, "peer", None) is not None,
            "url": None,
        },
        disclaimer=DISCLAIMER,
        compute_ms=round((time.perf_counter() - started) * 1000, 1),
    )
