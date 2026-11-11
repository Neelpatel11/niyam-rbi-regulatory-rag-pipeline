from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from niyam.config import Settings
from niyam.db import Database
from niyam.db.repository import record_query
from niyam.generation.answer import GenerationOutcome, generate_answer
from niyam.generation.llm import Effort, LLMClient
from niyam.generation.pricing import PriceBook, Usage
from niyam.generation.rewrite import rewrite_query
from niyam.models import AnswerStatus, RetrievedChunk
from niyam.retrieval import HybridRetriever, RetrievalConfig, RetrievalResult, TimeFilter
from niyam.tracing import Tracer

log = logging.getLogger(__name__)

_TEMPORAL_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\bas (?:of|on)\b|\bbefore\b|\bearlier\b|\bprevious(?:ly)?\b|"
    r"\bprior\b|\bused to\b|\bat the time\b|\bold(?:er)? (?:rule|direction|circular)s?\b",
    re.I,
)


def needs_rewrite(question: str, history: list[tuple[str, str]]) -> bool:
    """Only follow-ups and time-anchored questions pay for an LLM rewrite."""
    return bool(history) or _TEMPORAL_RE.search(question) is not None


NOT_FOUND_MESSAGE = "Not found in the documents."
FAILED_MESSAGE = (
    "I could not produce an answer whose citations check out against the documents. "
    "Try rephrasing the question."
)


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    retrieval: RetrievalConfig
    query_rewrite: bool = True
    refusal_threshold: float = 0.15
    refusal_threshold_dense: float = 0.55
    effort: Effort = "medium"
    max_output_tokens: int = 4000

    @classmethod
    def from_settings(cls, settings: Settings) -> PipelineConfig:
        return cls(
            retrieval=RetrievalConfig.from_settings(settings),
            query_rewrite=settings.query_rewrite,
            refusal_threshold=settings.refusal_threshold,
            refusal_threshold_dense=settings.refusal_threshold_dense,
            effort=settings.llm_effort,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.to_dict(),
            "query_rewrite": self.query_rewrite,
            "refusal_threshold": self.refusal_threshold,
            "refusal_threshold_dense": self.refusal_threshold_dense,
            "effort": self.effort,
        }


@dataclass(slots=True)
class AskRequest:
    question: str
    history: list[tuple[str, str]] = field(default_factory=list)
    as_of: date | None = None
    include_withdrawn: bool = False


@dataclass(slots=True)
class Citation:
    label: str
    chunk_id: str
    title: str
    circular_no: str | None
    clause_ref: str | None
    heading_path: list[str]
    url: str
    issued_on: date | None
    withdrawn_on: date | None
    superseded: bool
    amended_after_as_of: bool
    excerpt: str


@dataclass(slots=True)
class AskResponse:
    trace_id: str
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    question: str
    rewritten_question: str | None
    as_of: date
    retrieval_score: float | None
    retrieved: list[RetrievedChunk]
    timings_ms: dict[str, float]
    usage: list[Usage]
    cost_usd: float | None
    attempts: int
    problems: list[str] = field(default_factory=list)
    trace_url: str | None = None


def refusal_signal(result: RetrievalResult, config: PipelineConfig) -> tuple[float | None, float]:
    """Return (score, threshold). Reranker probabilities when available, else cosine similarity."""
    rerank_scores = [c.rerank_score for c in result.chunks if c.rerank_score is not None]
    if rerank_scores:
        return max(rerank_scores), config.refusal_threshold
    dense_scores = [c.dense_score for c in result.chunks if c.dense_score is not None]
    if dense_scores:
        return max(dense_scores), config.refusal_threshold_dense
    return None, config.refusal_threshold_dense


def _excerpt(text: str, limit: int = 320) -> str:
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_citations(outcome: GenerationOutcome, as_of: date) -> list[Citation]:
    return [
        Citation(
            label=label,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            circular_no=chunk.circular_no,
            clause_ref=chunk.clause_ref,
            heading_path=chunk.heading_path,
            url=chunk.citation_url,
            issued_on=chunk.issued_on,
            withdrawn_on=chunk.withdrawn_on,
            superseded=chunk.superseded,
            amended_after_as_of=bool(chunk.last_amended_on and chunk.last_amended_on > as_of),
            excerpt=_excerpt(chunk.text),
        )
        for label, chunk in zip(outcome.labels, outcome.cited, strict=True)
    ]


class QAService:
    def __init__(
        self,
        *,
        db: Database | None,
        retriever: HybridRetriever,
        llm: LLMClient,
        rewrite_llm: LLMClient | None,
        config: PipelineConfig,
        prices: PriceBook,
        tracer: Tracer | None = None,
        persist: bool = True,
    ) -> None:
        self._db = db
        self._retriever = retriever
        self._llm = llm
        self._rewrite_llm = rewrite_llm or llm
        self.config = config
        self._prices = prices
        self._tracer = tracer or Tracer(enabled=False)
        self._persist = persist and db is not None

    async def ask(self, request: AskRequest, config: PipelineConfig | None = None) -> AskResponse:
        response: AskResponse | None = None
        async for event, payload in self.ask_events(request, config):
            if event == "final":
                response = payload
        assert response is not None
        return response

    async def ask_events(
        self, request: AskRequest, config: PipelineConfig | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        cfg = config or self.config
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        timings: dict[str, float] = {}
        usages: list[Usage] = []
        today = date.today()

        with self._tracer.observation(
            "ask", input={"question": request.question}, metadata={"trace_id": trace_id}
        ) as root:
            question = request.question.strip()
            as_of = request.as_of
            rewritten: str | None = None

            if cfg.query_rewrite and (
                request.history or (as_of is None and needs_rewrite(question, request.history))
            ):
                yield "status", {"stage": "rewriting"}
                with self._tracer.observation("rewrite", as_type="generation") as span:
                    rewrite = await rewrite_query(
                        self._rewrite_llm, question, request.history, today=today
                    )
                    span.update(output={"question": rewrite.question, "as_of": str(rewrite.as_of)})
                usages.append(rewrite.usage)
                timings["rewrite"] = rewrite.latency_ms
                if rewrite.applied:
                    rewritten = rewrite.question
                if as_of is None:
                    as_of = rewrite.as_of

            search_question = rewritten or question
            effective_as_of = as_of or today
            time_filter = TimeFilter(as_of=as_of, include_withdrawn=request.include_withdrawn)

            yield "status", {"stage": "retrieving"}
            with self._tracer.observation("retrieve", as_type="retriever") as span:
                retrieval = await self._retriever.retrieve(
                    search_question, time_filter=time_filter, config=cfg.retrieval
                )
                span.update(
                    output=[
                        {"chunk_id": c.chunk_id, "rerank": c.rerank_score, "fused": c.fused_score}
                        for c in retrieval.chunks
                    ]
                )
            timings.update(retrieval.timings_ms)

            score, threshold = refusal_signal(retrieval, cfg)
            yield "sources", retrieval.chunks

            if score is None or score < threshold:
                outcome = GenerationOutcome(
                    AnswerStatus.NOT_FOUND,
                    "",
                    [],
                    [],
                    0,
                    Usage(self._llm.provider, self._llm.model),
                    0.0,
                )
            else:
                yield "status", {"stage": "generating"}
                with self._tracer.observation(
                    "generate", as_type="generation", model=self._llm.model
                ) as span:
                    outcome = await generate_answer(
                        self._llm,
                        search_question,
                        retrieval.chunks,
                        as_of=effective_as_of,
                        effort=cfg.effort,
                        max_tokens=cfg.max_output_tokens,
                    )
                    span.update(
                        output={"status": outcome.status.value, "answer": outcome.answer},
                        usage_details={
                            "input": outcome.usage.input_tokens,
                            "output": outcome.usage.output_tokens,
                        },
                    )
                timings["generate"] = outcome.latency_ms
            usages.append(outcome.usage)

            answer = outcome.answer
            if outcome.status is AnswerStatus.NOT_FOUND:
                answer = NOT_FOUND_MESSAGE
            elif outcome.status is AnswerStatus.FAILED_VALIDATION:
                answer = FAILED_MESSAGE

            costs = [self._prices.cost(u) for u in usages if u.calls]
            cost = None if any(c is None for c in costs) else float(sum(c or 0.0 for c in costs))
            timings["total"] = (time.perf_counter() - started) * 1000

            response = AskResponse(
                trace_id=trace_id,
                status=outcome.status,
                answer=answer,
                citations=build_citations(outcome, effective_as_of),
                question=question,
                rewritten_question=rewritten,
                as_of=effective_as_of,
                retrieval_score=score,
                retrieved=retrieval.chunks,
                timings_ms={k: round(v, 1) for k, v in timings.items()},
                usage=[u for u in usages if u.calls],
                cost_usd=cost,
                attempts=outcome.attempts,
                problems=outcome.problems,
                trace_url=self._tracer.trace_url(),
            )
            root.update(output={"status": response.status.value, "answer": response.answer})

        if self._persist:
            await self._save(response, cfg)
        yield "final", response

    async def _save(self, response: AskResponse, cfg: PipelineConfig) -> None:
        assert self._db is not None
        try:
            async with self._db.connection() as conn:
                await record_query(
                    conn,
                    {
                        "trace_id": response.trace_id,
                        "question": response.question,
                        "rewritten": response.rewritten_question,
                        "as_of": response.as_of,
                        "status": response.status.value,
                        "answer": {
                            "text": response.answer,
                            "citations": [c.chunk_id for c in response.citations],
                            "retrieved": [c.chunk_id for c in response.retrieved],
                            "retrieval_score": response.retrieval_score,
                            "problems": response.problems,
                        },
                        "config": cfg.to_dict(),
                        "timings_ms": response.timings_ms,
                        "usage": [
                            {
                                "model": u.key,
                                "input": u.input_tokens,
                                "output": u.output_tokens,
                                "cache_read": u.cache_read_tokens,
                            }
                            for u in response.usage
                        ],
                        "cost_usd": response.cost_usd or 0,
                    },
                )
        except Exception:
            log.exception("failed to persist query %s", response.trace_id)
