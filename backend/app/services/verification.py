"""Professional-registration verification for Peer Consult.

Medical licensure must be confirmed against the licensing authority of the
clinician's jurisdiction (for example a national medical council register). An
email address or a self-declared registration number proves nothing about
licensure, so neither is ever treated as verification here.

No authoritative provider is integrated yet. ``LicenceVerifier`` is the
interface a real integration implements; ``UnavailableVerifier`` is what runs
today and verifies nobody. The only other route to a usable account is the
development override in ``scripts/peer_consult_admin.py``, which is stored as
``dev_override`` (never ``verified``) and is honoured only when
``MEDXPLAIN_PEER_ALLOW_DEV_VERIFICATION=1``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

#: statuses stored on a clinician record
UNVERIFIED = "unverified"
PENDING = "pending"
VERIFIED = "verified"
REJECTED = "rejected"
DEV_OVERRIDE = "dev_override"

STATUS_LABELS = {
    UNVERIFIED: "Not verified",
    PENDING: "Verification requested - not completed",
    VERIFIED: "Registration verified",
    REJECTED: "Verification unsuccessful",
    DEV_OVERRIDE: "Development override - NOT checked against a licensing authority",
}


@dataclass
class VerificationResult:
    status: str  # VERIFIED | REJECTED | "unavailable"
    source: str
    reference: str | None
    checked_at: float
    detail: str


class LicenceVerifier(Protocol):
    name: str

    def describe(self) -> dict:
        """{'name', 'available', 'message'} for the status endpoint."""

    def verify(self, *, jurisdiction: str, registration_number: str, full_name: str) -> VerificationResult:
        """Look the registration up with the authoritative register."""


class UnavailableVerifier:
    """Placeholder until a jurisdiction's licensing register is integrated."""

    name = "none"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "available": False,
            "message": (
                "No licensing-authority verification service is connected. Registrations cannot be "
                "verified automatically yet, so new accounts stay unverified."
            ),
        }

    def verify(self, *, jurisdiction: str, registration_number: str, full_name: str) -> VerificationResult:
        return VerificationResult(
            status="unavailable",
            source=self.name,
            reference=None,
            checked_at=time.time(),
            detail="No licensing-authority verification service is connected.",
        )


def get_verifier(provider: str) -> LicenceVerifier:
    # Real providers register here, keyed by MEDXPLAIN_PEER_VERIFIER.
    return UnavailableVerifier()


def is_verified(status: str, allow_dev_override: bool) -> bool:
    """The single server-side rule for who may use Peer Consult."""
    return status == VERIFIED or (allow_dev_override and status == DEV_OVERRIDE)
