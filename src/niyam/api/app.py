from __future__ import annotations

import hmac
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from niyam import __version__
from niyam.api.schemas import AskBody, AskOut, DocumentOut, FeedbackBody, SourceOut
from niyam.config import Settings, get_settings
from niyam.db.repository import record_feedback
from niyam.log_config import configure_logging
from niyam.models import RetrievedChunk
from niyam.qa import AskRequest, AskResponse
from niyam.services import Services, build_services

log = logging.getLogger(__name__)


class RateLimiter:
    """Sliding one-minute window per client key. Process-local; use a shared store behind a fleet."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self._per_minute <= 0:
            return True
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self._per_minute:
            return False
        window.append(now)
        return True


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def _sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        SourceOut(
            chunk_id=c.chunk_id,
            title=c.title,
            circular_no=c.circular_no,
            clause_ref=c.clause_ref,
            heading_path=c.heading_path,
            url=c.citation_url,
            issued_on=c.issued_on,
            withdrawn_on=c.withdrawn_on,
            rerank_score=c.rerank_score,
            fused_score=c.fused_score,
        ).model_dump(mode="json")
        for c in chunks
    ]


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = services is None
        app.state.services = services or await build_services(settings)
        try:
            yield
        finally:
            if owned:
                await app.state.services.close()

    app = FastAPI(
        title="Niyam",
        version=__version__,
        summary="Citation-checked question answering over RBI Master Directions.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    def guard(request: Request) -> None:
        if settings.api_keys:
            header = request.headers.get("authorization", "")
            token = header.removeprefix("Bearer ").strip()
            if not any(hmac.compare_digest(token, key) for key in settings.api_keys):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(svc: ServicesDep) -> JSONResponse:
        ok = await svc.db.ping()
        return JSONResponse(
            {"status": "ok" if ok else "degraded", "database": ok},
            status_code=200 if ok else 503,
        )

    def _request(body: AskBody) -> AskRequest:
        return AskRequest(
            question=body.question,
            history=[(t.role, t.content) for t in body.history],
            as_of=body.as_of,
            include_withdrawn=body.include_withdrawn,
        )

    @app.post("/v1/ask", response_model=AskOut, dependencies=[Depends(guard)])
    async def ask(body: AskBody, svc: ServicesDep) -> AskOut:
        response = await svc.qa.ask(_request(body))
        return AskOut.from_response(response)

    @app.post("/v1/ask/stream", dependencies=[Depends(guard)])
    async def ask_stream(body: AskBody, svc: ServicesDep) -> EventSourceResponse:
        async def events() -> AsyncIterator[dict[str, str]]:
            try:
                async for name, payload in svc.qa.ask_events(_request(body)):
                    if name == "status":
                        yield {"event": "status", "data": json.dumps(payload)}
                    elif name == "sources":
                        yield {"event": "sources", "data": json.dumps(_sources(payload))}
                    elif name == "final":
                        assert isinstance(payload, AskResponse)
                        yield {
                            "event": "answer",
                            "data": AskOut.from_response(payload).model_dump_json(),
                        }
            except Exception:
                log.exception("streaming ask failed")
                yield {"event": "error", "data": json.dumps({"detail": "internal error"})}

        return EventSourceResponse(events(), ping=15)

    @app.post("/v1/feedback", status_code=204, dependencies=[Depends(guard)])
    async def feedback(body: FeedbackBody, svc: ServicesDep) -> None:
        async with svc.db.connection() as conn:
            found = await record_feedback(conn, body.trace_id, body.rating, body.comment)
        if not found:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown trace_id")

    @app.get("/v1/documents", response_model=list[DocumentOut])
    async def documents(
        svc: ServicesDep,
        q: Annotated[str | None, Query(max_length=200)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> list[DocumentOut]:
        async with svc.db.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT d.id, d.source_key, d.title, d.circular_no, d.issued_on,
                           d.last_amended_on, d.withdrawn_on, d.department, d.source_url,
                           COALESCE(array_agg(n.title ORDER BY n.issued_on DESC)
                                    FILTER (WHERE n.id IS NOT NULL), '{}') AS superseded_by
                    FROM documents d
                    LEFT JOIN supersessions s ON s.old_document_id = d.id
                    LEFT JOIN documents n ON n.id = s.new_document_id
                    WHERE %(q)s::text IS NULL OR d.title ILIKE '%%' || %(q)s || '%%'
                    GROUP BY d.id
                    ORDER BY d.issued_on DESC NULLS LAST
                    LIMIT %(limit)s
                    """,
                    {"q": q, "limit": limit},
                )
            ).fetchall()
        return [DocumentOut(**row) for row in rows]

    @app.get("/v1/stats")
    async def stats(svc: ServicesDep) -> dict[str, Any]:
        async with svc.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM documents) AS documents,
                        (SELECT count(*) FROM documents WHERE withdrawn_on IS NOT NULL) AS withdrawn,
                        (SELECT count(*) FROM chunks c JOIN chunk_sets s ON s.id = c.chunk_set_id
                          WHERE s.name = %s) AS chunks,
                        (SELECT max(fetched_at) FROM documents) AS last_ingested
                    """,
                    (svc.settings.chunk_set,),
                )
            ).fetchone()
        return {
            **(row or {}),
            "chunk_set": svc.settings.chunk_set,
            "embedding_model": svc.embedder.name,
            "reranker": svc.reranker.name if svc.reranker else None,
            "llm": f"{svc.llm.provider}:{svc.llm.model}",
        }

    return app
