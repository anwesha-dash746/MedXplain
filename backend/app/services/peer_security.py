"""Password hashing, session tokens and field encryption for Peer Consult.

* Passwords: salted PBKDF2-HMAC-SHA256 (stdlib), constant-time comparison.
* Sessions: random bearer tokens; only their SHA-256 hash is stored, so a copy
  of the database does not yield usable tokens.
* Case content: Fernet (AES-128-CBC + HMAC-SHA256) via ``cryptography``. The key
  comes from ``MEDXPLAIN_PEER_KEY``; without it a development key is generated
  on local disk and the status endpoint says so. A key on the same disk as the
  database protects against a leaked database file, not a compromised host.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from medxplain.utils import get_logger

LOGGER = get_logger("medxplain.peer.security")


# -- passwords ----------------------------------------------------------------------

def hash_password(password: str, iterations: int) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iterations)
        )
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    except (ValueError, TypeError):
        return False


# -- session tokens -----------------------------------------------------------------

def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# -- field encryption ---------------------------------------------------------------

class FieldCipher:
    def __init__(self, key: bytes, source: str):
        self._fernet = Fernet(key)
        #: where the key came from: "environment" or "development key file"
        self.source = source

    @classmethod
    def from_settings(cls, env_key: str | None, key_path: Path) -> "FieldCipher":
        if env_key:
            return cls(env_key.encode("utf-8"), "environment")
        key_path = Path(key_path)
        if key_path.exists():
            return cls(key_path.read_bytes().strip(), "development key file")
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        try:
            os.chmod(key_path, 0o600)
        except OSError:  # pragma: no cover - best effort on Windows
            pass
        LOGGER.warning(
            "MEDXPLAIN_PEER_KEY is not set; generated a development encryption key at %s. "
            "Set MEDXPLAIN_PEER_KEY for any shared deployment.", key_path
        )
        return cls(key, "development key file")

    def encrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken:
            # never surface ciphertext; a wrong key means the content is unreadable
            return "[unreadable: encryption key mismatch]"
