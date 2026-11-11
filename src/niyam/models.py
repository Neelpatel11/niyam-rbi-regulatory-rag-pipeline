from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOT_FOUND = "not_found"
    FAILED_VALIDATION = "failed_validation"


@dataclass(slots=True)
class DocumentMeta:
    source_key: str
    source_url: str
    title: str
    pdf_url: str | None = None
    doc_type: str = "master_direction"
    circular_no: str | None = None
    reference_nos: list[str] = field(default_factory=list)
    issued_on: date | None = None
    amended_on: list[date] = field(default_factory=list)
    withdrawn_on: date | None = None
    department: str | None = None

    @property
    def last_amended_on(self) -> date | None:
        return max(self.amended_on) if self.amended_on else None


@dataclass(slots=True)
class Element:
    kind: str
    text: str
    page: int | None = None
    level: int = 0
    label: str | None = None
    anchor: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    meta: DocumentMeta
    elements: list[Element]
    content_hash: str
    parser: str
    page_count: int | None = None


@dataclass(slots=True)
class Chunk:
    ordinal: int
    text: str
    token_count: int
    page_from: int | None = None
    page_to: int | None = None
    anchor: str | None = None
    heading_path: list[str] = field(default_factory=list)
    clause_ref: str | None = None
    has_table: bool = False

    @property
    def heading_text(self) -> str:
        return " > ".join(self.heading_path)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: int
    text: str
    heading_path: list[str]
    clause_ref: str | None
    page_from: int | None
    page_to: int | None
    anchor: str | None
    title: str
    source_key: str
    source_url: str
    pdf_url: str | None
    circular_no: str | None
    issued_on: date | None
    withdrawn_on: date | None
    last_amended_on: date | None
    superseded: bool
    token_count: int
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def citation_url(self) -> str:
        if self.page_from is not None and self.pdf_url:
            return f"{self.pdf_url}#page={self.page_from}"
        if self.anchor:
            return f"{self.source_url}#{self.anchor}"
        return self.source_url

    @property
    def is_current(self) -> bool:
        return self.withdrawn_on is None
