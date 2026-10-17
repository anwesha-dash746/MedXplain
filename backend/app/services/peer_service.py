"""Peer Consult business logic: accounts, directory, consultations, messages.

All authorisation decisions live here, not in the UI:

* only *verified* clinicians (see :func:`verification.is_verified`) may search
  the directory, appear in it, create or receive requests, or post messages;
* a consultation is visible only to its requester, its assigned clinician, or -
  while it is an unassigned open request - verified clinicians of the requested
  specialty who are accepting consults;
* every state change follows ``TRANSITIONS`` and is written to the audit log.

Case text (title, summary, findings, diagnosis, question, preferences, messages,
decline reasons) is encrypted at rest. Audit rows and application logs carry
identifiers and state names only.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from backend.app.services import verification as vf
from backend.app.services.peer_db import connect, migrate, schema_version
from backend.app.services.peer_security import (
    FieldCipher,
    hash_password,
    new_token,
    token_hash,
    verify_password,
)
from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.peer")

SPECIALTIES: tuple[str, ...] = (
    "Cardiology", "Critical Care", "Emergency Medicine", "Endocrinology", "Gastroenterology",
    "General Surgery", "Geriatrics", "Haematology", "Hepatology", "Infectious Diseases",
    "Internal Medicine", "Nephrology", "Neurology", "Oncology", "Palliative Care", "Psychiatry",
    "Pulmonology", "Radiology", "Rheumatology", "Other",
)

CONSULT_TYPES: dict[str, str] = {
    "second_opinion": "General Second Opinion",
    "complex_case": "Complex Case Discussion",
    "diagnostic": "Diagnostic Considerations",
    "treatment_review": "Treatment Plan Review",
    "follow_up": "Follow-up and Care Coordination",
}
PRIORITIES: dict[str, str] = {"routine": "Routine", "time_sensitive": "Time-sensitive"}

# -- workflow -----------------------------------------------------------------------
REQUESTED = "requested"
ACCEPTED = "accepted"
IN_DISCUSSION = "in_discussion"
RESPONSE_RECEIVED = "response_received"
CLOSED = "closed"
DECLINED = "declined"
CANCELLED = "cancelled"

STATUS_LABELS = {
    REQUESTED: "Requested",
    ACCEPTED: "Accepted",
    IN_DISCUSSION: "In Discussion",
    RESPONSE_RECEIVED: "Response Received",
    CLOSED: "Closed",
    DECLINED: "Declined",
    CANCELLED: "Cancelled",
}
ACTIVE = (ACCEPTED, IN_DISCUSSION, RESPONSE_RECEIVED)
FINISHED = (CLOSED, DECLINED, CANCELLED)

#: action -> (statuses it may start from, resulting status, who may perform it)
TRANSITIONS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "accept": ((REQUESTED,), ACCEPTED, "consultant"),
    "decline": ((REQUESTED,), DECLINED, "consultant"),
    "cancel": ((REQUESTED, ACCEPTED, IN_DISCUSSION, RESPONSE_RECEIVED), CANCELLED, "requester"),
    "close": (ACTIVE, CLOSED, "participant"),
}

MESSAGE_KINDS = ("message", "opinion")
OPINION_SECTIONS = ("interpretation", "considerations", "questions", "limitations", "references")

# -- de-identification screen -------------------------------------------------------
#: Obvious direct identifiers. A pass is not proof of de-identification; the
#: requester's explicit confirmation remains required.
IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("an email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("a phone number", re.compile(r"(?<!\d)(?:\+?\d[\d ().-]{8,}\d)(?!\d)")),
    ("a full date", re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")),
    ("a record number", re.compile(r"\b(MRN|medical record (number|no)|NHS (number|no)|SSN)\b", re.I)),
    ("a long identifier-like number", re.compile(r"(?<![\d.])\d{7,}(?![\d.])")),
    ("a de-identification placeholder from a source note", re.compile(r"\[\*\*.*?\*\*\]")),
)


class PeerError(Exception):
    status_code = 400

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class NotFound(PeerError):
    status_code = 404


class Forbidden(PeerError):
    status_code = 403


class Conflict(PeerError):
    status_code = 409


class Invalid(PeerError):
    status_code = 422


class Unauthorized(PeerError):
    status_code = 401


def find_identifiers(*texts: str | None) -> list[str]:
    found: list[str] = []
    for text in texts:
        if not text:
            continue
        for label, pattern in IDENTIFIER_PATTERNS:
            if label not in found and pattern.search(text):
                found.append(label)
    return found


@dataclass
class Page:
    items: list[dict]
    total: int
    page: int
    page_size: int


def build_peer_service(settings) -> "PeerService":
    return PeerService(
        settings.peer_db_path,
        FieldCipher.from_settings(settings.peer_key, settings.peer_key_path),
        vf.get_verifier(settings.peer_verifier),
        allow_dev_override=settings.peer_allow_dev_verification,
        session_hours=settings.peer_session_hours,
        password_iterations=settings.peer_password_iterations,
    )


class PeerService:
    def __init__(self, db_path: Path, cipher: FieldCipher, verifier: vf.LicenceVerifier, *,
                 allow_dev_override: bool, session_hours: float, password_iterations: int):
        self.db_path = Path(db_path)
        self.cipher = cipher
        self.verifier = verifier
        self.allow_dev_override = allow_dev_override
        self.session_seconds = session_hours * 3600
        self.password_iterations = password_iterations
        migrate(self.db_path)

    # -- plumbing --------------------------------------------------------------------
    @contextmanager
    def _db(self) -> Iterator:
        conn = connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _audit(self, conn, consultation_id: str | None, actor_id: str | None, action: str,
               from_status: str | None = None, to_status: str | None = None) -> None:
        conn.execute(
            "INSERT INTO audit_events (consultation_id, actor_id, action, from_status, to_status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (consultation_id, actor_id, action, from_status, to_status, time.time()),
        )

    def status(self) -> dict:
        return {
            "database": {"available": True, "engine": "sqlite", "schema_version": schema_version(self.db_path)},
            "verification": {
                **self.verifier.describe(),
                "dev_override_enabled": self.allow_dev_override,
            },
            "encryption": {
                "at_rest": True,
                "key_source": self.cipher.source,
                "note": (
                    "Case content and messages are encrypted in the database."
                    + (" The key is a development key stored on local disk." if self.cipher.source != "environment" else "")
                ),
            },
            "attachments": {"supported": False, "note": "File attachments are not supported in this version."},
            "specialties": list(SPECIALTIES),
            "consult_types": [{"key": k, "label": v} for k, v in CONSULT_TYPES.items()],
            "priorities": [{"key": k, "label": v} for k, v in PRIORITIES.items()],
        }

    # -- accounts --------------------------------------------------------------------
    def register(self, data: dict) -> dict:
        if data["specialty"] not in SPECIALTIES:
            raise Invalid("Choose a specialty from the list.")
        clinician_id = str(uuid.uuid4())
        with self._db() as conn:
            try:
                conn.execute(
                    "INSERT INTO clinicians (id, email, password_hash, display_name, specialty, subspecialty,"
                    " qualifications, years_experience, registration_jurisdiction, registration_number,"
                    " interests, accepting_consults, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        clinician_id, data["email"].strip().lower(),
                        hash_password(data["password"], self.password_iterations),
                        data["display_name"].strip(), data["specialty"], data.get("subspecialty"),
                        data.get("qualifications"), data.get("years_experience"),
                        data.get("registration_jurisdiction"), data.get("registration_number"),
                        json.dumps(data.get("interests") or []), 1 if data.get("accepting_consults", True) else 0,
                        time.time(),
                    ),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise Conflict("An account with this email already exists.") from exc
                raise
        LOGGER.info("clinician registered id=%s", clinician_id)
        return self.get_self(clinician_id)

    def login(self, email: str, password: str) -> tuple[str, float, dict]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM clinicians WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
            # compare against a dummy hash when the account is unknown, so timing does not reveal it
            stored = row["password_hash"] if row else hash_password("x", 1000)
            if not verify_password(password, stored) or row is None:
                raise Unauthorized("Email or password is incorrect.")
            token = new_token()
            now = time.time()
            expires = now + self.session_seconds
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            conn.execute(
                "INSERT INTO sessions (token_hash, clinician_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash(token), row["id"], now, expires),
            )
        return token, expires, self.get_self(row["id"])

    def logout(self, token: str) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))

    def authenticate(self, token: str | None) -> dict:
        if not token:
            raise Unauthorized("Sign in to use Peer Consult.")
        with self._db() as conn:
            row = conn.execute(
                "SELECT c.* FROM sessions s JOIN clinicians c ON c.id = s.clinician_id"
                " WHERE s.token_hash = ? AND s.expires_at > ?",
                (token_hash(token), time.time()),
            ).fetchone()
        if row is None:
            raise Unauthorized("Your session has expired. Sign in again.")
        return dict(row)

    def is_verified(self, clinician: dict) -> bool:
        return vf.is_verified(clinician["verification_status"], self.allow_dev_override)

    def require_verified(self, clinician: dict) -> None:
        if not self.is_verified(clinician):
            raise Forbidden(
                "Your professional registration has not been verified with a licensing authority, so "
                "Peer Consult is not available to this account yet."
            )

    def get_self(self, clinician_id: str) -> dict:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM clinicians WHERE id = ?", (clinician_id,)).fetchone()
        if row is None:
            raise NotFound("Account not found.")
        out = self._profile(dict(row))
        out["email"] = row["email"]
        out["registration_number"] = row["registration_number"]
        out["can_use_peer_consult"] = self.is_verified(dict(row))
        return out

    def update_profile(self, clinician: dict, data: dict) -> dict:
        allowed = ("display_name", "specialty", "subspecialty", "qualifications", "years_experience",
                   "interests", "accepting_consults")
        updates = {k: v for k, v in data.items() if k in allowed and v is not None}
        if "specialty" in updates and updates["specialty"] not in SPECIALTIES:
            raise Invalid("Choose a specialty from the list.")
        if "interests" in updates:
            updates["interests"] = json.dumps(updates["interests"])
        if "accepting_consults" in updates:
            updates["accepting_consults"] = 1 if updates["accepting_consults"] else 0
        if updates:
            cols = ", ".join(f"{k} = ?" for k in updates)
            with self._db() as conn:
                conn.execute(f"UPDATE clinicians SET {cols} WHERE id = ?", (*updates.values(), clinician["id"]))
        return self.get_self(clinician["id"])

    def request_verification(self, clinician: dict, jurisdiction: str, registration_number: str) -> dict:
        result = self.verifier.verify(
            jurisdiction=jurisdiction, registration_number=registration_number,
            full_name=clinician["display_name"],
        )
        if result.status == vf.VERIFIED:
            status, note = vf.VERIFIED, result.detail
        elif result.status == vf.REJECTED:
            status, note = vf.REJECTED, result.detail
        else:
            # keep a stronger status (e.g. an existing override) rather than downgrading it
            status = clinician["verification_status"] if clinician["verification_status"] != vf.UNVERIFIED else vf.PENDING
            note = result.detail
        with self._db() as conn:
            conn.execute(
                "UPDATE clinicians SET registration_jurisdiction = ?, registration_number = ?,"
                " verification_status = ?, verification_source = ?, verification_reference = ?,"
                " verification_checked_at = ?, verification_note = ? WHERE id = ?",
                (jurisdiction, registration_number, status, result.source, result.reference,
                 result.checked_at, note, clinician["id"]),
            )
        return {"result": result.status, "detail": result.detail, "profile": self.get_self(clinician["id"])}

    def set_dev_override(self, email: str) -> dict:
        """Development only (scripts/peer_consult_admin.py). Never sets VERIFIED."""
        with self._db() as conn:
            cur = conn.execute(
                "UPDATE clinicians SET verification_status = ?, verification_source = ?,"
                " verification_checked_at = ?, verification_note = ? WHERE email = ?",
                (vf.DEV_OVERRIDE, "development override", time.time(),
                 "Set manually for development. Not checked against any licensing authority.",
                 email.strip().lower()),
            )
            if cur.rowcount == 0:
                raise NotFound("No account with that email.")
            row = conn.execute("SELECT id FROM clinicians WHERE email = ?", (email.strip().lower(),)).fetchone()
            self._audit(conn, None, row["id"], "verification_dev_override")
        return self.get_self(row["id"])

    def _profile(self, row: dict) -> dict:
        status = row["verification_status"]
        return {
            "id": row["id"],
            "display_name": row["display_name"],
            "specialty": row["specialty"],
            "subspecialty": row["subspecialty"],
            "qualifications": row["qualifications"],
            "years_experience": row["years_experience"],
            "registration_jurisdiction": row["registration_jurisdiction"],
            "interests": json.loads(row["interests"] or "[]"),
            "accepting_consults": bool(row["accepting_consults"]),
            "verification": {
                "status": status,
                "label": vf.STATUS_LABELS.get(status, status),
                "source": row["verification_source"],
                "checked_at": row["verification_checked_at"],
                "note": row["verification_note"],
                "authoritative": status == vf.VERIFIED,
            },
        }

    # -- directory -------------------------------------------------------------------
    def _verified_sql(self) -> tuple[str, tuple]:
        statuses = [vf.VERIFIED] + ([vf.DEV_OVERRIDE] if self.allow_dev_override else [])
        return f"verification_status IN ({', '.join('?' * len(statuses))})", tuple(statuses)

    def search_clinicians(self, viewer: dict, *, q: str | None, specialty: str | None,
                          subspecialty: str | None, min_experience: int | None, interest: str | None,
                          available: bool | None, page: int, page_size: int) -> Page:
        self.require_verified(viewer)
        where_v, params_v = self._verified_sql()
        where = [where_v, "id != ?"]
        params: list = [*params_v, viewer["id"]]
        if q:
            where.append("(display_name LIKE ? OR interests LIKE ? OR subspecialty LIKE ? OR qualifications LIKE ?)")
            params += [f"%{q}%"] * 4
        if specialty:
            where.append("specialty = ?")
            params.append(specialty)
        if subspecialty:
            where.append("subspecialty LIKE ?")
            params.append(f"%{subspecialty}%")
        if min_experience is not None:
            where.append("years_experience >= ?")
            params.append(min_experience)
        if interest:
            where.append("interests LIKE ?")
            params.append(f"%{interest}%")
        if available is not None:
            where.append("accepting_consults = ?")
            params.append(1 if available else 0)
        clause = " AND ".join(where)
        with self._db() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM clinicians WHERE {clause}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM clinicians WHERE {clause} ORDER BY display_name LIMIT ? OFFSET ?",
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()
        return Page([self._profile(dict(r)) for r in rows], total, page, page_size)

    def get_clinician(self, viewer: dict, clinician_id: str) -> dict:
        self.require_verified(viewer)
        with self._db() as conn:
            row = conn.execute("SELECT * FROM clinicians WHERE id = ?", (clinician_id,)).fetchone()
        if row is None or not self.is_verified(dict(row)):
            raise NotFound("Clinician not found.")
        return self._profile(dict(row))

    # -- consultations ---------------------------------------------------------------
    def _box_sql(self, viewer: dict, box: str) -> tuple[str, list]:
        me = viewer["id"]
        if box == "sent":
            return "c.requester_id = ?", [me]
        if box == "received":
            return (
                "c.status = ? AND (c.assigned_id = ? OR (c.assigned_id IS NULL AND c.specialty = ?"
                " AND c.requester_id != ? AND ? = 1))",
                [REQUESTED, me, viewer["specialty"], me, 1 if viewer["accepting_consults"] else 0],
            )
        if box == "active":
            return (
                f"c.status IN ({', '.join('?' * len(ACTIVE))}) AND (c.requester_id = ? OR c.assigned_id = ?)",
                [*ACTIVE, me, me],
            )
        if box == "completed":
            return (
                f"c.status IN ({', '.join('?' * len(FINISHED))}) AND (c.requester_id = ? OR c.assigned_id = ?)",
                [*FINISHED, me, me],
            )
        raise Invalid("Unknown list. Use sent, received, active or completed.")

    def dashboard(self, viewer: dict) -> dict:
        self.require_verified(viewer)
        counts = {}
        with self._db() as conn:
            for box in ("sent", "received", "active", "completed"):
                where, params = self._box_sql(viewer, box)
                counts[box] = conn.execute(f"SELECT COUNT(*) FROM consultations c WHERE {where}", params).fetchone()[0]
        return {"counts": counts}

    def create_consultation(self, viewer: dict, data: dict) -> tuple[dict, bool]:
        self.require_verified(viewer)
        if data["specialty"] not in SPECIALTIES:
            raise Invalid("Choose a specialty from the list.")
        if data["consult_type"] not in CONSULT_TYPES:
            raise Invalid("Choose a consultation type from the list.")
        if data["priority"] not in PRIORITIES:
            raise Invalid("Choose a priority from the list.")
        if not (data.get("deidentified_confirmed") and data.get("authorized_confirmed")):
            raise Invalid("Confirm that the case is de-identified and that you are authorised to share it.")
        identifiers = find_identifiers(
            data["title"], data["summary"], data["findings"], data.get("diagnosis"),
            data["question"], data.get("matching_preferences"),
        )
        if identifiers:
            raise Invalid(
                "The request appears to contain " + ", ".join(identifiers)
                + ". Remove identifying details before sharing."
            )

        with self._db() as conn:
            existing = conn.execute(
                "SELECT id FROM consultations WHERE requester_id = ? AND client_request_id = ?",
                (viewer["id"], data["client_request_id"]),
            ).fetchone()
            if existing:
                # same submission received twice: return the original, create nothing
                return self._detail(conn, viewer, existing["id"]), False

            assigned = data.get("preferred_clinician_id")
            if assigned:
                if assigned == viewer["id"]:
                    raise Invalid("You cannot send a consultation request to yourself.")
                row = conn.execute("SELECT * FROM clinicians WHERE id = ?", (assigned,)).fetchone()
                if row is None or not self.is_verified(dict(row)):
                    raise Invalid("The selected clinician is not available in the verified directory.")
                if not row["accepting_consults"]:
                    raise Invalid("The selected clinician is not accepting consultation requests.")

            now = time.time()
            consult_id = str(uuid.uuid4())
            enc = self.cipher.encrypt
            conn.execute(
                "INSERT INTO consultations (id, requester_id, assigned_id, client_request_id, specialty,"
                " subspecialty, consult_type, priority, status, title_enc, summary_enc, findings_enc,"
                " diagnosis_enc, question_enc, matching_enc, deidentified_confirmed_at, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    consult_id, viewer["id"], assigned, data["client_request_id"], data["specialty"],
                    data.get("subspecialty"), data["consult_type"], data["priority"], REQUESTED,
                    enc(data["title"]), enc(data["summary"]), enc(data["findings"]),
                    enc(data.get("diagnosis")), enc(data["question"]), enc(data.get("matching_preferences")),
                    now, now, now,
                ),
            )
            self._audit(conn, consult_id, viewer["id"], "created", None, REQUESTED)
            return self._detail(conn, viewer, consult_id), True

    def _load(self, conn, consult_id: str) -> dict:
        row = conn.execute("SELECT * FROM consultations WHERE id = ?", (consult_id,)).fetchone()
        if row is None:
            raise NotFound("Consultation not found.")
        return dict(row)

    def _role(self, viewer: dict, c: dict) -> str | None:
        if c["requester_id"] == viewer["id"]:
            return "requester"
        if c["assigned_id"] == viewer["id"]:
            return "consultant"
        if (c["assigned_id"] is None and c["status"] == REQUESTED and c["specialty"] == viewer["specialty"]
                and viewer["accepting_consults"] and self.is_verified(viewer)):
            return "open_pool"
        return None

    def _require_role(self, viewer: dict, c: dict) -> str:
        role = self._role(viewer, c)
        if role is None:
            # 404, not 403: do not confirm that a consultation exists to non-participants
            raise NotFound("Consultation not found.")
        return role

    def _names(self, conn, *ids: str | None) -> dict[str, str]:
        wanted = [i for i in ids if i]
        if not wanted:
            return {}
        rows = conn.execute(
            f"SELECT id, display_name FROM clinicians WHERE id IN ({', '.join('?' * len(wanted))})", wanted
        ).fetchall()
        return {r["id"]: r["display_name"] for r in rows}

    def _allowed_actions(self, role: str, c: dict) -> list[str]:
        actions = []
        for action, (sources, _target, who) in TRANSITIONS.items():
            if c["status"] not in sources:
                continue
            if who == "requester" and role != "requester":
                continue
            if who == "consultant" and role not in ("consultant", "open_pool"):
                continue
            if action == "decline" and role == "open_pool":
                continue  # open requests are simply left for another colleague
            if who == "participant" and role not in ("requester", "consultant"):
                continue
            actions.append(action)
        if role in ("requester", "consultant") and c["status"] in ACTIVE:
            actions.append("message")
            if role == "consultant":
                actions.append("opinion")
        return actions

    def _summary_row(self, c: dict, names: dict, viewer: dict) -> dict:
        counterpart = c["assigned_id"] if c["requester_id"] == viewer["id"] else c["requester_id"]
        return {
            "id": c["id"],
            "title": self.cipher.decrypt(c["title_enc"]),
            "specialty": c["specialty"],
            "consult_type": c["consult_type"],
            "consult_type_label": CONSULT_TYPES.get(c["consult_type"], c["consult_type"]),
            "priority": c["priority"],
            "priority_label": PRIORITIES.get(c["priority"], c["priority"]),
            "status": c["status"],
            "status_label": STATUS_LABELS[c["status"]],
            "role": "requester" if c["requester_id"] == viewer["id"] else
                    "consultant" if c["assigned_id"] == viewer["id"] else "open_pool",
            "counterpart_name": names.get(counterpart) if counterpart else None,
            "open_request": c["assigned_id"] is None,
            "created_at": c["created_at"],
            "updated_at": c["updated_at"],
        }

    def list_consultations(self, viewer: dict, box: str, page: int, page_size: int) -> Page:
        self.require_verified(viewer)
        where, params = self._box_sql(viewer, box)
        with self._db() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM consultations c WHERE {where}", params).fetchone()[0]
            rows = [dict(r) for r in conn.execute(
                f"SELECT c.* FROM consultations c WHERE {where} ORDER BY c.updated_at DESC LIMIT ? OFFSET ?",
                (*params, page_size, (page - 1) * page_size),
            ).fetchall()]
            names = self._names(conn, *{i for r in rows for i in (r["requester_id"], r["assigned_id"])})
        return Page([self._summary_row(r, names, viewer) for r in rows], total, page, page_size)

    def _detail(self, conn, viewer: dict, consult_id: str) -> dict:
        c = self._load(conn, consult_id)
        role = self._require_role(viewer, c)
        names = self._names(conn, c["requester_id"], c["assigned_id"])
        dec = self.cipher.decrypt
        messages = [
            {
                "id": m["id"],
                "author_id": m["author_id"],
                "author_name": self._names(conn, m["author_id"]).get(m["author_id"]),
                "author_role": "requester" if m["author_id"] == c["requester_id"] else "consultant",
                "kind": m["kind"],
                "content": json.loads(dec(m["body_enc"]) or "{}"),
                "created_at": m["created_at"],
            }
            for m in conn.execute(
                "SELECT * FROM messages WHERE consultation_id = ? ORDER BY created_at", (consult_id,)
            ).fetchall()
        ]
        return {
            **self._summary_row(c, names, viewer),
            "role": role,
            "requester": {"id": c["requester_id"], "name": names.get(c["requester_id"])},
            "assigned": {"id": c["assigned_id"], "name": names.get(c["assigned_id"])} if c["assigned_id"] else None,
            "subspecialty": c["subspecialty"],
            "summary": dec(c["summary_enc"]),
            "findings": dec(c["findings_enc"]),
            "diagnosis": dec(c["diagnosis_enc"]),
            "question": dec(c["question_enc"]),
            "matching_preferences": dec(c["matching_enc"]),
            "decline_reason": dec(c["decline_reason_enc"]),
            "closed_at": c["closed_at"],
            "messages": messages,
            "allowed_actions": self._allowed_actions(role, c),
        }

    def get_consultation(self, viewer: dict, consult_id: str) -> dict:
        self.require_verified(viewer)
        with self._db() as conn:
            detail = self._detail(conn, viewer, consult_id)
            self._audit(conn, consult_id, viewer["id"], "viewed")
            return detail

    def act(self, viewer: dict, consult_id: str, action: str, reason: str | None = None) -> dict:
        self.require_verified(viewer)
        if action not in TRANSITIONS:
            raise Invalid("Unknown action.")
        with self._db() as conn:
            c = self._load(conn, consult_id)
            role = self._require_role(viewer, c)
            if action not in self._allowed_actions(role, c):
                raise Conflict(
                    f"Cannot {action} a consultation that is {STATUS_LABELS[c['status']].lower()}"
                    + ("" if role in ("requester", "consultant") or action == "accept" else " from your role")
                    + "."
                )
            _sources, target, _who = TRANSITIONS[action]
            now = time.time()
            if reason and find_identifiers(reason):
                raise Invalid("The reason appears to contain identifying details. Remove them first.")
            # the status condition makes accept/decline race-safe: two colleagues
            # accepting the same open request cannot both succeed
            cur = conn.execute(
                "UPDATE consultations SET status = ?, updated_at = ?,"
                " assigned_id = COALESCE(assigned_id, ?),"
                " closed_at = CASE WHEN ? IN ('closed', 'declined', 'cancelled') THEN ? ELSE closed_at END,"
                " decline_reason_enc = CASE WHEN ? = 'declined' THEN ? ELSE decline_reason_enc END"
                " WHERE id = ? AND status = ?",
                (target, now, viewer["id"] if action == "accept" else None, target, now,
                 target, self.cipher.encrypt(reason) if reason else None, consult_id, c["status"]),
            )
            if cur.rowcount == 0:
                raise Conflict("The consultation changed while you were acting on it. Reload and try again.")
            self._audit(conn, consult_id, viewer["id"], action, c["status"], target)
            return self._detail(conn, viewer, consult_id)

    def post_message(self, viewer: dict, consult_id: str, kind: str, content: dict) -> dict:
        self.require_verified(viewer)
        if kind not in MESSAGE_KINDS:
            raise Invalid("Unknown message type.")
        if kind == "message":
            text = (content.get("text") or "").strip()
            if not text:
                raise Invalid("Write a message first.")
            body = {"text": text}
        else:
            body = {k: (content.get(k) or "").strip() for k in OPINION_SECTIONS}
            if not body["interpretation"]:
                raise Invalid("An opinion needs at least your interpretation.")
        identifiers = find_identifiers(*body.values())
        if identifiers:
            raise Invalid("The message appears to contain " + ", ".join(identifiers) + ". Remove it first.")

        with self._db() as conn:
            c = self._load(conn, consult_id)
            role = self._require_role(viewer, c)
            allowed = self._allowed_actions(role, c)
            if kind not in allowed:
                if role == "open_pool":
                    raise Conflict("Accept the request before replying.")
                if kind == "opinion" and role == "requester":
                    raise Forbidden("Only the consulting clinician can post an opinion.")
                raise Conflict(f"Messages cannot be added to a consultation that is {STATUS_LABELS[c['status']].lower()}.")
            now = time.time()
            conn.execute(
                "INSERT INTO messages (id, consultation_id, author_id, kind, body_enc, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), consult_id, viewer["id"], kind, self.cipher.encrypt(json.dumps(body)), now),
            )
            new_status = c["status"]
            if kind == "opinion":
                new_status = RESPONSE_RECEIVED
            elif c["status"] == ACCEPTED:
                new_status = IN_DISCUSSION
            conn.execute(
                "UPDATE consultations SET status = ?, updated_at = ? WHERE id = ?", (new_status, now, consult_id)
            )
            self._audit(conn, consult_id, viewer["id"], "opinion_posted" if kind == "opinion" else "message_posted",
                        c["status"], new_status if new_status != c["status"] else None)
            return self._detail(conn, viewer, consult_id)

    def history(self, viewer: dict, consult_id: str) -> list[dict]:
        self.require_verified(viewer)
        with self._db() as conn:
            c = self._load(conn, consult_id)
            role = self._require_role(viewer, c)
            if role == "open_pool":
                raise NotFound("Consultation not found.")
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE consultation_id = ? ORDER BY id", (consult_id,)
            ).fetchall()
            names = self._names(conn, *{r["actor_id"] for r in rows})
        return [
            {
                "action": r["action"],
                "actor_name": names.get(r["actor_id"]),
                "from_status": r["from_status"],
                "to_status": r["to_status"],
                "created_at": r["created_at"],
            }
            # access ("viewed") events stay in the audit table for review but are not
            # part of the clinical history shown to participants
            for r in rows
            if r["action"] != "viewed"
        ]

    # -- retention -------------------------------------------------------------------
    def purge_finished(self, retention_days: int) -> int:
        """Delete finished consultations (and their messages/audit) past the retention window."""
        cutoff = time.time() - retention_days * 86400
        with self._db() as conn:
            cur = conn.execute(
                f"DELETE FROM consultations WHERE status IN ({', '.join('?' * len(FINISHED))}) AND closed_at < ?",
                (*FINISHED, cutoff),
            )
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        if cur.rowcount:
            LOGGER.info("retention: purged %d finished consultations", cur.rowcount)
        return cur.rowcount
