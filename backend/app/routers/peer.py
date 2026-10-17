"""/peer - Peer Consult: clinician accounts, verified directory and consultations.

Authentication is a bearer session token from ``POST /peer/auth/login``. Every
endpoint except ``/peer/status``, ``/peer/auth/register`` and
``/peer/auth/login`` requires it, and everything beyond the caller's own
profile additionally requires a verified registration - enforced in
:class:`PeerService`, not in the UI.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response

from backend.app.schemas_peer import (
    ActionRequest,
    ConsultationCreate,
    LoginRequest,
    MessageCreate,
    ProfileUpdate,
    RegisterRequest,
    VerificationRequest,
)
from backend.app.services.peer_service import PeerError, PeerService

router = APIRouter(prefix="/peer", tags=["peer consult"])

DISCLAIMER = (
    "Peer Consult is not an emergency service. For urgent or emergency patient care, use established "
    "clinical escalation channels. Colleagues' opinions are advisory; the treating clinician remains "
    "responsible for patient care."
)


def service(request: Request) -> PeerService:
    peer = getattr(request.app.state, "peer", None)
    if peer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Peer Consult storage is unavailable: {getattr(request.app.state, 'peer_error', 'not initialised')}",
        )
    return peer


def _token(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def current_clinician(
    authorization: str | None = Header(None), peer: PeerService = Depends(service)
) -> dict:
    try:
        return peer.authenticate(_token(authorization))
    except PeerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail,
                            headers={"WWW-Authenticate": "Bearer"}) from exc


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PeerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _page(page) -> dict:
    return {"items": page.items, "total": page.total, "page": page.page, "page_size": page.page_size}


# -- status & accounts --------------------------------------------------------------

@router.get("/status", summary="What parts of Peer Consult are available")
def status(request: Request) -> dict:
    peer = getattr(request.app.state, "peer", None)
    if peer is None:
        return {
            "database": {"available": False, "error": getattr(request.app.state, "peer_error", "not initialised")},
            "disclaimer": DISCLAIMER,
        }
    return {**peer.status(), "disclaimer": DISCLAIMER}


@router.post("/auth/register", status_code=201, summary="Create an (unverified) clinician account")
def register(payload: RegisterRequest, peer: PeerService = Depends(service)) -> dict:
    return _call(peer.register, payload.model_dump())


@router.post("/auth/login", summary="Sign in and receive a session token")
def login(payload: LoginRequest, peer: PeerService = Depends(service)) -> dict:
    token, expires_at, profile = _call(peer.login, payload.email, payload.password)
    return {"token": token, "expires_at": expires_at, "profile": profile}


@router.post("/auth/logout", status_code=204, summary="End this session")
def logout(authorization: str | None = Header(None), peer: PeerService = Depends(service)) -> Response:
    token = _token(authorization)
    if token:
        peer.logout(token)
    return Response(status_code=204)


@router.get("/me", summary="The signed-in clinician's own profile")
def me(clinician: dict = Depends(current_clinician), peer: PeerService = Depends(service)) -> dict:
    return _call(peer.get_self, clinician["id"])


@router.patch("/me", summary="Update the signed-in clinician's profile")
def update_me(payload: ProfileUpdate, clinician: dict = Depends(current_clinician),
              peer: PeerService = Depends(service)) -> dict:
    return _call(peer.update_profile, clinician, payload.model_dump(exclude_unset=True))


@router.post("/me/verification", summary="Submit registration details for licensing verification")
def request_verification(payload: VerificationRequest, clinician: dict = Depends(current_clinician),
                         peer: PeerService = Depends(service)) -> dict:
    return _call(peer.request_verification, clinician, payload.registration_jurisdiction,
                 payload.registration_number)


# -- directory ----------------------------------------------------------------------

@router.get("/clinicians", summary="Search the directory of verified clinicians")
def clinicians(
    q: str | None = Query(None, max_length=100),
    specialty: str | None = Query(None, max_length=60),
    subspecialty: str | None = Query(None, max_length=100),
    min_experience: int | None = Query(None, ge=0, le=70),
    interest: str | None = Query(None, max_length=60),
    available: bool | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    clinician: dict = Depends(current_clinician),
    peer: PeerService = Depends(service),
) -> dict:
    return _page(_call(
        peer.search_clinicians, clinician, q=q, specialty=specialty, subspecialty=subspecialty,
        min_experience=min_experience, interest=interest, available=available, page=page, page_size=page_size,
    ))


@router.get("/clinicians/{clinician_id}", summary="One verified clinician's profile")
def clinician_profile(clinician_id: str, clinician: dict = Depends(current_clinician),
                      peer: PeerService = Depends(service)) -> dict:
    return _call(peer.get_clinician, clinician, clinician_id)


# -- consultations ------------------------------------------------------------------

@router.get("/dashboard", summary="Counts for the Peer Consult dashboard")
def dashboard(clinician: dict = Depends(current_clinician), peer: PeerService = Depends(service)) -> dict:
    return _call(peer.dashboard, clinician)


@router.post("/consultations", status_code=201, summary="Create a consultation request")
def create_consultation(payload: ConsultationCreate, response: Response,
                        clinician: dict = Depends(current_clinician),
                        peer: PeerService = Depends(service)) -> dict:
    detail, created = _call(peer.create_consultation, clinician, payload.model_dump())
    if not created:
        response.status_code = 200  # duplicate submission: original returned, nothing created
    return {"consultation": detail, "created": created}


@router.get("/consultations", summary="List consultations in one box")
def list_consultations(
    box: str = Query(..., pattern="^(sent|received|active|completed)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    clinician: dict = Depends(current_clinician),
    peer: PeerService = Depends(service),
) -> dict:
    return _page(_call(peer.list_consultations, clinician, box, page, page_size))


@router.get("/consultations/{consult_id}", summary="One consultation (participants only)")
def get_consultation(consult_id: str, clinician: dict = Depends(current_clinician),
                     peer: PeerService = Depends(service)) -> dict:
    return _call(peer.get_consultation, clinician, consult_id)


@router.post("/consultations/{consult_id}/actions", summary="Accept, decline, cancel or close")
def act(consult_id: str, payload: ActionRequest, clinician: dict = Depends(current_clinician),
        peer: PeerService = Depends(service)) -> dict:
    return _call(peer.act, clinician, consult_id, payload.action, payload.reason)


@router.post("/consultations/{consult_id}/messages", status_code=201, summary="Post a message or an opinion")
def post_message(consult_id: str, payload: MessageCreate, clinician: dict = Depends(current_clinician),
                 peer: PeerService = Depends(service)) -> dict:
    content = payload.model_dump(exclude={"kind"})
    return _call(peer.post_message, clinician, consult_id, payload.kind, content)


@router.get("/consultations/{consult_id}/history", summary="Status history (participants only)")
def history(consult_id: str, clinician: dict = Depends(current_clinician),
            peer: PeerService = Depends(service)) -> list[dict]:
    return _call(peer.history, clinician, consult_id)
