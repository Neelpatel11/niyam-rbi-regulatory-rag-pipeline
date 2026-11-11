from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)


class _NullObservation:
    def update(self, **_: Any) -> None:
        return None


class Tracer:
    """Thin wrapper over Langfuse. Every call is a no-op when tracing is disabled or misconfigured."""

    def __init__(self, enabled: bool) -> None:
        self._client: Any = None
        if not enabled:
            return
        try:
            from langfuse import get_client

            self._client = get_client()
        except Exception:
            log.warning("Langfuse is enabled but could not be initialised", exc_info=True)

    @property
    def active(self) -> bool:
        return self._client is not None

    @contextmanager
    def observation(self, name: str, *, as_type: str = "span", **fields: Any) -> Iterator[Any]:
        if self._client is None:
            yield _NullObservation()
            return
        try:
            manager = self._client.start_as_current_observation(
                name=name, as_type=as_type, **fields
            )
        except Exception:
            log.debug("failed to start observation %s", name, exc_info=True)
            yield _NullObservation()
            return
        with manager as observation:
            yield observation

    def trace_url(self) -> str | None:
        if self._client is None:
            return None
        try:
            return self._client.get_trace_url()
        except Exception:
            return None

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                log.debug("langfuse flush failed", exc_info=True)
