"""Peer Consult administration (development only).

    python scripts/peer_consult_admin.py status
    python scripts/peer_consult_admin.py dev-verify clinician@example.org

``dev-verify`` marks an existing account as ``dev_override``. It is NOT a
licensing check: the status is stored and displayed as a development override,
and the API honours it only when MEDXPLAIN_PEER_ALLOW_DEV_VERIFICATION=1. Use it
to exercise the workflow locally until a licensing-authority integration
exists. Never use it to vouch for a real clinician.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.services.peer_service import build_peer_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show storage, verification and encryption status")
    dev = sub.add_parser("dev-verify", help="set a development verification override on an account")
    dev.add_argument("email")
    args = parser.parse_args()

    settings = Settings.from_env()
    service = build_peer_service(settings)
    if args.command == "status":
        status = service.status()
        print(json.dumps({k: status[k] for k in ("database", "verification", "encryption")}, indent=2))
        return 0

    profile = service.set_dev_override(args.email)
    print(f"{profile['display_name']}: {profile['verification']['label']}")
    if not settings.peer_allow_dev_verification:
        print("Note: MEDXPLAIN_PEER_ALLOW_DEV_VERIFICATION is not set, so the API will still treat "
              "this account as unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
