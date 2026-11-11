from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Category(StrEnum):
    SINGLE_FACT = "single_fact"
    MULTI_CLAUSE = "multi_clause"
    NUMERIC_TABLE = "numeric_table"
    TEMPORAL = "temporal"
    UNANSWERABLE = "unanswerable"


class Evidence(BaseModel):
    source_key: str = Field(description="Document key, e.g. rbi-md-13141.")
    quote: str = Field(
        min_length=12, description="Verbatim span from the document that supports the answer."
    )


class GoldenItem(BaseModel):
    """One evaluation question.

    Evidence is stored as verbatim quotes rather than chunk IDs, so the same golden set scores
    every chunking strategy fairly.
    """

    id: str
    question: str
    category: Category
    expected_answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    as_of: date | None = None
    include_withdrawn: bool = False
    history: list[tuple[str, str]] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> GoldenItem:
        if self.category is Category.UNANSWERABLE:
            if self.evidence:
                raise ValueError(f"{self.id}: unanswerable items must not carry evidence")
        else:
            if not self.evidence:
                raise ValueError(f"{self.id}: answerable items need at least one evidence quote")
            if not self.expected_answer.strip():
                raise ValueError(f"{self.id}: answerable items need an expected_answer")
        return self

    @property
    def answerable(self) -> bool:
        return self.category is not Category.UNANSWERABLE

    def split(self, dev_fraction: float = 0.3) -> str:
        """Stable dev/test assignment. Thresholds are tuned on dev and reported on test."""
        bucket = int(hashlib.sha256(self.id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "dev" if bucket < dev_fraction else "test"


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                item = GoldenItem.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {item.id}")
            seen.add(item.id)
            items.append(item)
    return items


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_defaults=True) + "\n")
