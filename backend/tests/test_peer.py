"""Peer Consult: authentication, verification gate, directory, consultation
workflow, message permissions and persistence.

The API tests run the real FastAPI app against a temporary SQLite database with
the development verification override *enabled*, because no licensing provider
exists yet; the tests in ``TestVerificationGate`` show the override is ignored
when the setting is off and that unverified accounts are blocked either way.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app.services import verification as vf
from backend.app.services.peer_security import FieldCipher
from backend.app.services.peer_service import (
    Forbidden,
    PeerService,
    find_identifiers,
)

PASSWORD = "correct-horse-battery"


def _settings(tmp_path, allow_dev: bool):
    from backend.app.config import Settings

    return Settings(
        checkpoint_root=tmp_path / "checkpoints",
        results_root=tmp_path / "results",
        case_library_path=tmp_path / "none.parquet",
        peer_db_path=tmp_path / "peer.db",
        peer_key_path=tmp_path / "peer.key",
        peer_allow_dev_verification=allow_dev,
        peer_password_iterations=1000,
        preload_default_model=False,
    )


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    from backend.app import config as config_module
    import backend.app.main as main_module

    settings = _settings(tmp_path, allow_dev=True)
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    with TestClient(main_module.app) as client:
        yield client


class Clinician:
    """A registered test account with a session token."""

    def __init__(self, client, name, specialty="Cardiology", verify=True, **extra):
        self.client = client
        self.email = f"{uuid.uuid4().hex[:8]}@test.invalid"
        body = {
            "email": self.email, "password": PASSWORD, "display_name": name, "specialty": specialty,
            "years_experience": extra.pop("years_experience", 10), **extra,
        }
        r = client.post("/api/peer/auth/register", json=body)
        assert r.status_code == 201, r.text
        self.id = r.json()["id"]
        if verify:
            client.app.state.peer.set_dev_override(self.email)
        login = client.post("/api/peer/auth/login", json={"email": self.email, "password": PASSWORD})
        assert login.status_code == 200, login.text
        self.headers = {"Authorization": f"Bearer {login.json()['token']}"}

    def get(self, path, **kw):
        return self.client.get(f"/api/peer{path}", headers=self.headers, **kw)

    def post(self, path, json=None):
        return self.client.post(f"/api/peer{path}", headers=self.headers, json=json)


def request_body(**overrides):
    body = {
        "client_request_id": uuid.uuid4().hex,
        "title": "Recurrent syncope with preserved ejection fraction",
        "specialty": "Cardiology",
        "summary": "Patient in their 70s admitted with two episodes of syncope; telemetry unremarkable.",
        "findings": "Orthostatic drop of 25 mmHg systolic; echo shows preserved EF.",
        "diagnosis": "Orthostatic hypotension vs arrhythmia",
        "question": "Would you recommend an implantable loop recorder before discharge?",
        "consult_type": "diagnostic",
        "priority": "routine",
        "deidentified_confirmed": True,
        "authorized_confirmed": True,
    }
    body.update(overrides)
    return body


# -----------------------------------------------------------------------------------

class TestAuthentication:
    def test_endpoints_require_a_session(self, app_client):
        for path in ("/api/peer/me", "/api/peer/clinicians", "/api/peer/dashboard",
                     "/api/peer/consultations?box=sent"):
            assert app_client.get(path).status_code == 401

    def test_wrong_password_is_rejected(self, app_client):
        a = Clinician(app_client, "Dr Test A")
        r = app_client.post("/api/peer/auth/login", json={"email": a.email, "password": "wrong-password"})
        assert r.status_code == 401

    def test_duplicate_email_is_rejected(self, app_client):
        a = Clinician(app_client, "Dr Test A")
        r = app_client.post("/api/peer/auth/register", json={
            "email": a.email, "password": PASSWORD, "display_name": "Other", "specialty": "Cardiology"})
        assert r.status_code == 409

    def test_logout_invalidates_the_token(self, app_client):
        a = Clinician(app_client, "Dr Test A")
        assert a.get("/me").status_code == 200
        app_client.post("/api/peer/auth/logout", headers=a.headers)
        assert a.get("/me").status_code == 401

    def test_registration_validates_input(self, app_client):
        r = app_client.post("/api/peer/auth/register", json={
            "email": "not-an-email", "password": "short", "display_name": "X", "specialty": "Cardiology"})
        assert r.status_code == 422
        r = app_client.post("/api/peer/auth/register", json={
            "email": "a@test.invalid", "password": PASSWORD, "display_name": "Dr Test", "specialty": "Astrology"})
        assert r.status_code == 422


class TestVerificationGate:
    def test_new_accounts_are_unverified_and_blocked(self, app_client):
        u = Clinician(app_client, "Dr Unverified", verify=False)
        me = u.get("/me").json()
        assert me["verification"]["status"] == "unverified"
        assert me["can_use_peer_consult"] is False
        assert u.get("/clinicians").status_code == 403
        assert u.get("/dashboard").status_code == 403
        assert u.post("/consultations", request_body()).status_code == 403

    def test_verification_request_cannot_succeed_without_a_provider(self, app_client):
        u = Clinician(app_client, "Dr Unverified", verify=False)
        r = u.post("/me/verification", {"registration_jurisdiction": "Test Board", "registration_number": "T-1"})
        assert r.status_code == 200
        assert r.json()["result"] == "unavailable"
        assert r.json()["profile"]["can_use_peer_consult"] is False

    def test_unverified_clinicians_are_hidden_from_the_directory(self, app_client):
        a = Clinician(app_client, "Dr Verified")
        Clinician(app_client, "Dr Hidden", verify=False)
        names = [c["display_name"] for c in a.get("/clinicians").json()["items"]]
        assert "Dr Hidden" not in names

    def test_unverified_clinician_cannot_be_chosen_as_consultant(self, app_client):
        a = Clinician(app_client, "Dr Verified")
        hidden = Clinician(app_client, "Dr Hidden", verify=False)
        r = a.post("/consultations", request_body(preferred_clinician_id=hidden.id))
        assert r.status_code == 422

    def test_dev_override_is_ignored_when_the_setting_is_off(self, tmp_path):
        service = PeerService(
            tmp_path / "p.db", FieldCipher.from_settings(None, tmp_path / "k"), vf.UnavailableVerifier(),
            allow_dev_override=False, session_hours=1, password_iterations=1000,
        )
        service.register({"email": "x@test.invalid", "password": PASSWORD, "display_name": "Dr X",
                          "specialty": "Cardiology"})
        service.set_dev_override("x@test.invalid")
        _, _, profile = service.login("x@test.invalid", PASSWORD)
        assert profile["verification"]["status"] == "dev_override"
        assert profile["verification"]["authoritative"] is False
        assert profile["can_use_peer_consult"] is False
        clinician = service.authenticate(service.login("x@test.invalid", PASSWORD)[0])
        with pytest.raises(Forbidden):
            service.dashboard(clinician)

    def test_override_is_labelled_as_not_a_licensing_check(self, app_client):
        a = Clinician(app_client, "Dr Verified")
        label = a.get("/me").json()["verification"]["label"]
        assert "NOT checked against a licensing authority" in label


class TestDirectory:
    def test_filters_and_pagination(self, app_client):
        me = Clinician(app_client, "Dr Searcher", specialty="Internal Medicine")
        Clinician(app_client, "Dr Heart", specialty="Cardiology", years_experience=20,
                  subspecialty="Electrophysiology", interests=["syncope"])
        Clinician(app_client, "Dr Kidney", specialty="Nephrology", years_experience=5)
        assert [c["display_name"] for c in me.get("/clinicians", params={"specialty": "Cardiology"}).json()["items"]] == ["Dr Heart"]
        assert me.get("/clinicians", params={"min_experience": 15}).json()["total"] == 1
        assert me.get("/clinicians", params={"interest": "syncope"}).json()["total"] == 1
        assert me.get("/clinicians", params={"subspecialty": "electro"}).json()["total"] == 1
        page = me.get("/clinicians", params={"page_size": 1, "page": 2}).json()
        assert page["total"] == 2 and len(page["items"]) == 1

    def test_directory_excludes_self_and_private_fields(self, app_client):
        me = Clinician(app_client, "Dr Searcher", registration_number="SECRET-1")
        other = Clinician(app_client, "Dr Other", registration_number="SECRET-2")
        items = me.get("/clinicians").json()["items"]
        assert [c["id"] for c in items] == [other.id]
        assert "email" not in items[0] and "registration_number" not in items[0]


class TestConsultationWorkflow:
    def test_full_lifecycle_with_history(self, app_client):
        req = Clinician(app_client, "Dr Requester", specialty="Internal Medicine")
        con = Clinician(app_client, "Dr Consultant", specialty="Cardiology")
        r = req.post("/consultations", request_body(preferred_clinician_id=con.id))
        assert r.status_code == 201
        cid = r.json()["consultation"]["id"]
        assert r.json()["consultation"]["status"] == "requested"

        assert con.get("/dashboard").json()["counts"]["received"] == 1
        assert con.post(f"/consultations/{cid}/actions", {"action": "accept"}).json()["status"] == "accepted"
        assert req.post(f"/consultations/{cid}/messages", {"text": "Thanks - any thoughts?"}).json()["status"] == "in_discussion"
        opinion = con.post(f"/consultations/{cid}/messages", {
            "kind": "opinion", "interpretation": "Picture fits orthostatic hypotension.",
            "considerations": "Review antihypertensives.", "limitations": "Based on summary only.",
        })
        assert opinion.status_code == 201 and opinion.json()["status"] == "response_received"
        closed = req.post(f"/consultations/{cid}/actions", {"action": "close"}).json()
        assert closed["status"] == "closed" and closed["closed_at"]

        detail = req.get(f"/consultations/{cid}").json()
        assert [m["author_name"] for m in detail["messages"]] == ["Dr Requester", "Dr Consultant"]
        assert detail["messages"][1]["content"]["limitations"] == "Based on summary only."
        assert all(m["created_at"] for m in detail["messages"])

        history = [(h["action"], h["to_status"]) for h in req.get(f"/consultations/{cid}/history").json()]
        assert history == [
            ("created", "requested"), ("accept", "accepted"),
            ("message_posted", "in_discussion"), ("opinion_posted", "response_received"), ("close", "closed"),
        ]
        assert req.get("/dashboard").json()["counts"]["completed"] == 1

    def test_decline_records_reason_and_only_consultant_may_decline(self, app_client):
        req = Clinician(app_client, "Dr Requester")
        con = Clinician(app_client, "Dr Consultant")
        cid = req.post("/consultations", request_body(preferred_clinician_id=con.id)).json()["consultation"]["id"]
        assert req.post(f"/consultations/{cid}/actions", {"action": "decline"}).status_code == 409
        r = con.post(f"/consultations/{cid}/actions", {"action": "decline", "reason": "Outside my area."})
        assert r.json()["status"] == "declined" and r.json()["decline_reason"] == "Outside my area."

    def test_invalid_transitions_are_conflicts(self, app_client):
        req = Clinician(app_client, "Dr Requester")
        con = Clinician(app_client, "Dr Consultant")
        cid = req.post("/consultations", request_body(preferred_clinician_id=con.id)).json()["consultation"]["id"]
        assert req.post(f"/consultations/{cid}/actions", {"action": "close"}).status_code == 409
        assert req.post(f"/consultations/{cid}/actions", {"action": "accept"}).status_code == 409
        req.post(f"/consultations/{cid}/actions", {"action": "cancel"})
        assert con.post(f"/consultations/{cid}/actions", {"action": "accept"}).status_code == 409

    def test_open_request_reaches_matching_specialists_and_first_accept_wins(self, app_client):
        req = Clinician(app_client, "Dr Requester", specialty="Internal Medicine")
        a = Clinician(app_client, "Dr Cardio A", specialty="Cardiology")
        b = Clinician(app_client, "Dr Cardio B", specialty="Cardiology")
        neph = Clinician(app_client, "Dr Neph", specialty="Nephrology")
        cid = req.post("/consultations", request_body()).json()["consultation"]["id"]
        assert a.get("/dashboard").json()["counts"]["received"] == 1
        assert neph.get("/dashboard").json()["counts"]["received"] == 0
        assert neph.get(f"/consultations/{cid}").status_code == 404
        assert a.post(f"/consultations/{cid}/actions", {"action": "accept"}).json()["assigned"]["id"] == a.id
        assert b.post(f"/consultations/{cid}/actions", {"action": "accept"}).status_code == 404
        assert b.get(f"/consultations/{cid}").status_code == 404


class TestValidationAndDuplicates:
    def test_missing_required_fields_are_rejected(self, app_client):
        a = Clinician(app_client, "Dr A")
        for field in ("title", "summary", "findings", "question", "consult_type", "priority"):
            body = request_body()
            body.pop(field)
            assert a.post("/consultations", body).status_code == 422, field

    def test_confirmations_are_required(self, app_client):
        a = Clinician(app_client, "Dr A")
        assert a.post("/consultations", request_body(deidentified_confirmed=False)).status_code == 422
        assert a.post("/consultations", request_body(authorized_confirmed=False)).status_code == 422

    def test_obvious_identifiers_are_rejected(self, app_client):
        a = Clinician(app_client, "Dr A")
        for text in ("Contact at patient@test.invalid for details",
                     "Seen on 12/03/2024 in clinic with chest pain and dyspnoea",
                     "MRN 99887766 admitted with chest pain and dyspnoea"):
            r = a.post("/consultations", request_body(summary=text))
            assert r.status_code == 422, text

    def test_duplicate_submission_returns_the_original(self, app_client):
        a = Clinician(app_client, "Dr A")
        body = request_body()
        first = a.post("/consultations", body)
        second = a.post("/consultations", body)
        assert first.status_code == 201 and second.status_code == 200
        assert second.json()["created"] is False
        assert first.json()["consultation"]["id"] == second.json()["consultation"]["id"]
        assert a.get("/dashboard").json()["counts"]["sent"] == 1

    def test_cannot_request_yourself(self, app_client):
        a = Clinician(app_client, "Dr A")
        assert a.post("/consultations", request_body(preferred_clinician_id=a.id)).status_code == 422


class TestMessagePermissions:
    def _accepted(self, app_client):
        req = Clinician(app_client, "Dr Requester", specialty="Internal Medicine")
        con = Clinician(app_client, "Dr Consultant", specialty="Cardiology")
        cid = req.post("/consultations", request_body(preferred_clinician_id=con.id)).json()["consultation"]["id"]
        return req, con, cid

    def test_no_messages_before_acceptance(self, app_client):
        req, con, cid = self._accepted(app_client)
        assert req.post(f"/consultations/{cid}/messages", {"text": "Hello"}).status_code == 409

    def test_outsider_cannot_read_or_write(self, app_client):
        req, con, cid = self._accepted(app_client)
        con.post(f"/consultations/{cid}/actions", {"action": "accept"})
        outsider = Clinician(app_client, "Dr Outsider", specialty="Neurology")
        assert outsider.get(f"/consultations/{cid}").status_code == 404
        assert outsider.get(f"/consultations/{cid}/history").status_code == 404
        assert outsider.post(f"/consultations/{cid}/messages", {"text": "Hi"}).status_code == 404
        assert outsider.post(f"/consultations/{cid}/actions", {"action": "close"}).status_code == 404

    def test_only_the_consultant_posts_opinions(self, app_client):
        req, con, cid = self._accepted(app_client)
        con.post(f"/consultations/{cid}/actions", {"action": "accept"})
        r = req.post(f"/consultations/{cid}/messages", {"kind": "opinion", "interpretation": "My own view"})
        assert r.status_code == 403

    def test_no_messages_after_closing(self, app_client):
        req, con, cid = self._accepted(app_client)
        con.post(f"/consultations/{cid}/actions", {"action": "accept"})
        req.post(f"/consultations/{cid}/actions", {"action": "close"})
        assert con.post(f"/consultations/{cid}/messages", {"text": "One more thing"}).status_code == 409

    def test_messages_are_screened_for_identifiers(self, app_client):
        req, con, cid = self._accepted(app_client)
        con.post(f"/consultations/{cid}/actions", {"action": "accept"})
        assert req.post(f"/consultations/{cid}/messages", {"text": "Call me on +44 7700 900123"}).status_code == 422


class TestPersistenceAndEncryption:
    def test_case_content_is_encrypted_at_rest(self, app_client, tmp_path):
        req = Clinician(app_client, "Dr Requester")
        con = Clinician(app_client, "Dr Consultant")
        body = request_body(preferred_clinician_id=con.id, summary="Distinctive phrase zebra-quartz in the summary.")
        req.post("/consultations", body)
        raw = sqlite3.connect(str(tmp_path / "peer.db")).execute(
            "SELECT title_enc, summary_enc, question_enc FROM consultations").fetchone()
        assert all("zebra-quartz" not in v and "syncope" not in v.lower() for v in raw)

    def test_records_survive_a_restart(self, tmp_path):
        def make():
            return PeerService(
                tmp_path / "p.db", FieldCipher.from_settings(None, tmp_path / "k"), vf.UnavailableVerifier(),
                allow_dev_override=True, session_hours=1, password_iterations=1000,
            )

        first = make()
        first.register({"email": "r@test.invalid", "password": PASSWORD, "display_name": "Dr R",
                        "specialty": "Cardiology"})
        first.set_dev_override("r@test.invalid")
        viewer = first.authenticate(first.login("r@test.invalid", PASSWORD)[0])
        consult, _ = first.create_consultation(viewer, request_body())
        second = make()
        assert second.get_consultation(viewer, consult["id"])["title"] == request_body()["title"]

    def test_audit_log_holds_no_case_content(self, app_client, tmp_path):
        req = Clinician(app_client, "Dr Requester")
        req.post("/consultations", request_body())
        rows = sqlite3.connect(str(tmp_path / "peer.db")).execute("SELECT * FROM audit_events").fetchall()
        assert rows and all("syncope" not in str(r).lower() for r in rows)


def test_identifier_screen_leaves_clinical_values_alone():
    assert find_identifiers("BP 112/67, HR 121, Cr 5.1, lactate 1.1, EF 25 percent, 2 weeks") == []
