"""Cache-first HuggingFace loading, so a machine without network still runs.

Every backbone this project uses is pulled once by ``.cache/fetch_models.py``
and then lives in the local HuggingFace cache. That should be enough to train
offline, but it is not: ``from_pretrained`` issues a HEAD request for the
current revision on every call, and once DNS fails the whole stage dies with

    Failed to resolve 'huggingface.co' ... getaddrinfo failed

even though the weights are sitting on disk. Air-gapped lab machines, CI
runners and a laptop on a bad connection all hit this, and the pipeline is long
enough that losing stage 2 to it is expensive.

:func:`from_pretrained` wraps the HF loaders with a cache-first fallback: it
honours an explicit offline setting, and otherwise retries with
``local_files_only=True`` when - and only when - the first attempt failed for a
network reason. A model that is genuinely absent from the cache still raises,
so a typo in a model id is never silently swallowed; the error just names the
cache as well as the hub.

Set ``MEDXPLAIN_OFFLINE=1`` (or the standard ``HF_HUB_OFFLINE`` /
``TRANSFORMERS_OFFLINE``) to skip the network probe entirely and go straight to
the cache, which is also a few seconds faster per load.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

#: Env vars that force cache-only loading. The first is ours; the other two are
#: the standard HuggingFace switches, honoured so an existing offline
#: environment needs no extra configuration.
_OFFLINE_VARS = ("MEDXPLAIN_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

#: Substrings that identify a failure as "could not reach the hub" rather than
#: "this model does not exist". Matched against the exception chain's text
#: because the hub raises a moving target of wrapped urllib3/requests errors.
_NETWORK_MARKERS = (
    "getaddrinfo failed",
    "failed to resolve",
    "nameresolutionerror",
    "max retries exceeded",
    "connectionerror",
    "connection aborted",
    "connection refused",
    "temporary failure in name resolution",
    "read timed out",
    "connecttimeout",
    "offlinemodeisenabled",
    "couldn't connect to",
    "could not connect to",
    "we couldn't connect",
    "localentrynotfounderror",
    "network is unreachable",
    "ssl",
)

#: Set once we have proven the hub is unreachable, so the remaining loads in a
#: run go straight to the cache instead of each paying the same timeout.
_network_known_down = False


def offline_requested() -> bool:
    """True if any offline env var is set to something truthy."""
    return any(os.environ.get(var, "").strip().lower() not in ("", "0", "false", "no")
               for var in _OFFLINE_VARS)


def _is_network_error(exc: BaseException) -> bool:
    """True if ``exc`` (or anything it wraps) looks like a connectivity failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}".lower()
        if any(marker in text for marker in _NETWORK_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def from_pretrained(loader: Callable[..., T], model_id: str, **kwargs: Any) -> T:
    """Call ``loader.from_pretrained(model_id, **kwargs)``, falling back to cache.

    ``loader`` is an ``AutoTokenizer`` / ``AutoConfig`` / ``AutoModel`` class (or
    anything else exposing ``from_pretrained``). Behaviour:

    * offline explicitly requested, or the hub already proved unreachable in
      this process -> load with ``local_files_only=True`` straight away;
    * otherwise try normally, and on a *network* failure retry from the cache;
    * any other failure, and any failure of the cache retry, propagates.
    """
    global _network_known_down

    if kwargs.get("local_files_only"):
        return loader.from_pretrained(model_id, **kwargs)

    if offline_requested() or _network_known_down:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)

    try:
        return loader.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised below unless it's the network
        if not _is_network_error(exc):
            raise
        if not _network_known_down:
            LOGGER.warning(
                "hub unreachable (%s); loading %s and later models from the local cache. "
                "Set MEDXPLAIN_OFFLINE=1 to skip this probe.",
                type(exc).__name__, model_id,
            )
            _network_known_down = True
        try:
            return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
        except Exception as cache_exc:  # noqa: BLE001 - annotate, then re-raise
            raise OSError(
                f"{model_id!r} could not be downloaded (no network) and is not in the "
                f"local HuggingFace cache. Run .cache/fetch_models.py on a connected "
                f"machine, or copy ~/.cache/huggingface/hub across."
            ) from cache_exc
