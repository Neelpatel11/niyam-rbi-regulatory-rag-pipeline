from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from niyam.qa import AskResponse


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=12)
    as_of: date | None = None
    include_withdrawn: bool = False

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("question is too short")
        return stripped


class CitationOut(BaseModel):
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


class SourceOut(BaseModel):
    chunk_id: str
    title: str
    circular_no: str | None
    clause_ref: str | None
    heading_path: list[str]
    url: str
    issued_on: date | None
    withdrawn_on: date | None
    rerank_score: float | None
    fused_score: float


class UsageOut(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


class AskOut(BaseModel):
    trace_id: str
    status: Literal["answered", "not_found", "failed_validation"]
    answer: str
    citations: list[CitationOut]
    sources: list[SourceOut]
    question: str
    rewritten_question: str | None
    as_of: date
    retrieval_score: float | None
    timings_ms: dict[str, float]
    usage: list[UsageOut]
    cost_usd: float | None
    attempts: int
    trace_url: str | None

    @classmethod
    def from_response(cls, response: AskResponse) -> AskOut:
        return cls(
            trace_id=response.trace_id,
            status=response.status.value,
            answer=response.answer,
            citations=[_citation(c) for c in response.citations],
            sources=[
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
                )
                for c in response.retrieved
            ],
            question=response.question,
            rewritten_question=response.rewritten_question,
            as_of=response.as_of,
            retrieval_score=response.retrieval_score,
            timings_ms=response.timings_ms,
            usage=[
                UsageOut(
                    model=u.key,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cache_read_tokens=u.cache_read_tokens,
                )
                for u in response.usage
            ],
            cost_usd=response.cost_usd,
            attempts=response.attempts,
            trace_url=response.trace_url,
        )


def _citation(c: object) -> CitationOut:
    fields = CitationOut.model_fields
    return CitationOut(**{name: getattr(c, name) for name in fields})


class FeedbackBody(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)


class DocumentOut(BaseModel):
    id: int
    source_key: str
    title: str
    circular_no: str | None
    issued_on: date | None
    last_amended_on: date | None
    withdrawn_on: date | None
    department: str | None
    source_url: str
    superseded_by: list[str]
