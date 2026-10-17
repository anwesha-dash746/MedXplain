"""SQLite storage for Peer Consult, with ordered, idempotent migrations.

SQLite keeps the prototype dependency-free (stdlib only) and runnable on a
laptop. Every connection enables foreign keys; schema changes are appended to
``MIGRATIONS`` and never edited once released, and ``schema_migrations``
records which versions have run.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE clinicians (
            id                       TEXT PRIMARY KEY,
            email                    TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash            TEXT NOT NULL,
            display_name             TEXT NOT NULL,
            specialty                TEXT NOT NULL,
            subspecialty             TEXT,
            qualifications           TEXT,
            years_experience         INTEGER,
            registration_jurisdiction TEXT,
            registration_number      TEXT,
            interests                TEXT NOT NULL DEFAULT '[]',
            accepting_consults       INTEGER NOT NULL DEFAULT 1,
            verification_status      TEXT NOT NULL DEFAULT 'unverified',
            verification_source      TEXT,
            verification_reference   TEXT,
            verification_checked_at  REAL,
            verification_note        TEXT,
            created_at               REAL NOT NULL
        );
        CREATE INDEX idx_clinicians_specialty ON clinicians(specialty);

        CREATE TABLE sessions (
            token_hash    TEXT PRIMARY KEY,
            clinician_id  TEXT NOT NULL REFERENCES clinicians(id) ON DELETE CASCADE,
            created_at    REAL NOT NULL,
            expires_at    REAL NOT NULL
        );

        CREATE TABLE consultations (
            id                  TEXT PRIMARY KEY,
            requester_id        TEXT NOT NULL REFERENCES clinicians(id),
            assigned_id         TEXT REFERENCES clinicians(id),
            client_request_id   TEXT NOT NULL,
            specialty           TEXT NOT NULL,
            subspecialty        TEXT,
            consult_type        TEXT NOT NULL,
            priority            TEXT NOT NULL,
            status              TEXT NOT NULL,
            -- encrypted at rest:
            title_enc           TEXT NOT NULL,
            summary_enc         TEXT NOT NULL,
            findings_enc        TEXT NOT NULL,
            diagnosis_enc       TEXT,
            question_enc        TEXT NOT NULL,
            matching_enc        TEXT,
            decline_reason_enc  TEXT,
            deidentified_confirmed_at REAL NOT NULL,
            created_at          REAL NOT NULL,
            updated_at          REAL NOT NULL,
            closed_at           REAL,
            UNIQUE (requester_id, client_request_id)
        );
        CREATE INDEX idx_consult_requester ON consultations(requester_id, status);
        CREATE INDEX idx_consult_assigned ON consultations(assigned_id, status);
        CREATE INDEX idx_consult_open ON consultations(specialty, status);

        CREATE TABLE messages (
            id               TEXT PRIMARY KEY,
            consultation_id  TEXT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
            author_id        TEXT NOT NULL REFERENCES clinicians(id),
            kind             TEXT NOT NULL,
            body_enc         TEXT NOT NULL,
            created_at       REAL NOT NULL
        );
        CREATE INDEX idx_messages_consult ON messages(consultation_id, created_at);

        -- audit rows hold identifiers and state changes only, never case content
        CREATE TABLE audit_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            consultation_id  TEXT REFERENCES consultations(id) ON DELETE CASCADE,
            actor_id         TEXT,
            action           TEXT NOT NULL,
            from_status      TEXT,
            to_status        TEXT,
            created_at       REAL NOT NULL
        );
        CREATE INDEX idx_audit_consult ON audit_events(consultation_id, created_at);
        """,
    ),
)


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(path: Path) -> list[int]:
    """Apply pending migrations; returns the versions applied by this call."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[int] = []
    with connect(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, sql in MIGRATIONS:
            if version in done:
                continue
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, time.time())
            )
            applied.append(version)
    return applied


def schema_version(path: Path) -> int:
    with connect(path) as conn:
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)
