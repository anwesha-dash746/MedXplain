"""Request/response schemas for the Peer Consult API (/api/peer).

Length limits are enforced here so oversized input is rejected before it
reaches the service; vocabularies (specialty, consultation type, priority) are
checked by the service against its single source of truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _strip(v):
    return v.strip() if isinstance(v, str) else v


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=10, max_length=200)
    display_name: str = Field(..., min_length=3, max_length=120,
                              description="Professional name as it appears on the register.")
    specialty: str = Field(..., max_length=60)
    subspecialty: str | None = Field(None, max_length=100)
    qualifications: str | None = Field(None, max_length=300)
    years_experience: int | None = Field(None, ge=0, le=70)
    registration_jurisdiction: str | None = Field(None, max_length=100)
    registration_number: str | None = Field(None, max_length=60)
    interests: list[str] = Field(default_factory=list, max_length=15)
    accepting_consults: bool = True

    _s = field_validator("display_name", "specialty", "subspecialty", "qualifications", mode="before")(_strip)

    @field_validator("interests")
    @classmethod
    def _interests(cls, v: list[str]) -> list[str]:
        cleaned = [i.strip() for i in v if i and i.strip()]
        if any(len(i) > 60 for i in cleaned):
            raise ValueError("each interest must be 60 characters or fewer")
        return cleaned


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=200)


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=3, max_length=120)
    specialty: str | None = Field(None, max_length=60)
    subspecialty: str | None = Field(None, max_length=100)
    qualifications: str | None = Field(None, max_length=300)
    years_experience: int | None = Field(None, ge=0, le=70)
    interests: list[str] | None = Field(None, max_length=15)
    accepting_consults: bool | None = None


class VerificationRequest(BaseModel):
    registration_jurisdiction: str = Field(..., min_length=2, max_length=100)
    registration_number: str = Field(..., min_length=2, max_length=60)


class ConsultationCreate(BaseModel):
    client_request_id: str = Field(
        ..., min_length=8, max_length=64, pattern=r"^[A-Za-z0-9-]+$",
        description="Generated once per form; resubmitting the same id returns the original request.",
    )
    title: str = Field(..., min_length=5, max_length=150)
    specialty: str = Field(..., max_length=60)
    subspecialty: str | None = Field(None, max_length=100)
    summary: str = Field(..., min_length=30, max_length=4000, description="De-identified case summary.")
    findings: str = Field(..., min_length=5, max_length=4000)
    diagnosis: str | None = Field(None, max_length=2000)
    question: str = Field(..., min_length=10, max_length=2000)
    consult_type: str
    priority: Literal["routine", "time_sensitive"]
    preferred_clinician_id: str | None = Field(None, max_length=64)
    matching_preferences: str | None = Field(None, max_length=500)
    deidentified_confirmed: bool
    authorized_confirmed: bool

    _s = field_validator(
        "title", "subspecialty", "summary", "findings", "diagnosis", "question", "matching_preferences",
        mode="before",
    )(_strip)

    @field_validator("subspecialty", "diagnosis", "matching_preferences", "preferred_clinician_id")
    @classmethod
    def _empty_to_none(cls, v):
        return v or None


class ActionRequest(BaseModel):
    action: Literal["accept", "decline", "cancel", "close"]
    reason: str | None = Field(None, max_length=1000)


class MessageCreate(BaseModel):
    kind: Literal["message", "opinion"] = "message"
    text: str | None = Field(None, max_length=4000)
    interpretation: str | None = Field(None, max_length=4000)
    considerations: str | None = Field(None, max_length=4000)
    questions: str | None = Field(None, max_length=2000)
    limitations: str | None = Field(None, max_length=2000)
    references: str | None = Field(None, max_length=2000)
