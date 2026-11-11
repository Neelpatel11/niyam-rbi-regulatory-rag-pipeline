from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from niyam.models import Chunk, Element
from niyam.text import Tokenizer, get_tokenizer, split_sentences

_TOP_CLAUSE_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){0,3}$")


class Chunker(Protocol):
    name: str
    strategy: str

    def params(self) -> dict[str, int]: ...

    def chunk(self, elements: list[Element]) -> list[Chunk]: ...


@dataclass(slots=True)
class _Unit:
    texts: list[str]
    heading_path: list[str]
    anchor: str | None
    label: str | None
    page_from: int | None
    page_to: int | None
    tokens: int
    has_table: bool = False


@dataclass(slots=True)
class _Pending:
    units: list[_Unit] = field(default_factory=list)
    tokens: int = 0

    def clear(self) -> None:
        self.units.clear()
        self.tokens = 0


def _clause_ref(labels: list[str]) -> str | None:
    top = [
        label.rstrip(".") for label in labels if label and _TOP_CLAUSE_RE.match(label.rstrip("."))
    ]
    if not top:
        return None
    if len(top) == 1 or top[0] == top[-1]:
        return top[0]
    return f"{top[0]}-{top[-1]}"


def _pages(units: list[_Unit]) -> tuple[int | None, int | None]:
    pages = [p for u in units for p in (u.page_from, u.page_to) if p is not None]
    return (min(pages), max(pages)) if pages else (None, None)


class StructureChunker:
    """Splits on headings and numbered clauses, packs whole clauses up to a token budget."""

    strategy = "structure"

    def __init__(
        self,
        name: str = "structure-v1",
        target_tokens: int = 450,
        max_tokens: int = 800,
        min_tokens: int = 120,
        overlap_tokens: int = 60,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("expected 0 < min_tokens <= target_tokens <= max_tokens")
        self.name = name
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self._tok = tokenizer or get_tokenizer()

    def params(self) -> dict[str, int]:
        return {
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "overlap_tokens": self.overlap_tokens,
        }

    def _units(self, elements: list[Element]) -> list[_Unit]:
        units: list[_Unit] = []
        path: list[str] = []
        heading_anchor: str | None = None
        current: _Unit | None = None

        def close() -> None:
            nonlocal current
            if current is not None and current.texts:
                current.tokens = self._tok.count("\n".join(current.texts))
                units.append(current)
            current = None

        for element in elements:
            if element.kind == "heading":
                close()
                path = [element.text] if element.level <= 1 else [*path[:1], element.text]
                heading_anchor = element.anchor or heading_anchor
                continue

            if element.kind == "table":
                close()
                units.append(
                    _Unit(
                        texts=[element.text],
                        heading_path=list(path),
                        anchor=element.anchor or heading_anchor,
                        label=None,
                        page_from=element.page,
                        page_to=element.page,
                        tokens=self._tok.count(element.text),
                        has_table=True,
                    )
                )
                continue

            starts_clause = bool(element.label and _TOP_CLAUSE_RE.match(element.label.rstrip(".")))
            if current is None or starts_clause:
                close()
                current = _Unit(
                    texts=[],
                    heading_path=list(path),
                    anchor=element.anchor or heading_anchor,
                    label=element.label,
                    page_from=element.page,
                    page_to=element.page,
                    tokens=0,
                )
            current.texts.append(element.text)
            if element.page is not None:
                current.page_to = element.page
                if current.page_from is None:
                    current.page_from = element.page
        close()
        return units

    def _split_unit(self, unit: _Unit) -> list[_Unit]:
        if unit.tokens <= self.max_tokens:
            return [unit]
        if unit.has_table:
            return self._split_table(unit)

        pieces: list[str] = []
        for text in unit.texts:
            pieces.extend(split_sentences(text))
        windows: list[list[str]] = []
        window: list[str] = []
        window_tokens = 0
        for sentence in pieces:
            sentence_tokens = self._tok.count(sentence)
            if window and window_tokens + sentence_tokens > self.target_tokens:
                windows.append(window)
                overlap: list[str] = []
                overlap_tokens = 0
                for prev in reversed(window):
                    prev_tokens = self._tok.count(prev)
                    if overlap_tokens + prev_tokens > self.overlap_tokens:
                        break
                    overlap.insert(0, prev)
                    overlap_tokens += prev_tokens
                window = list(overlap)
                window_tokens = overlap_tokens
            window.append(sentence)
            window_tokens += sentence_tokens
        if window:
            windows.append(window)

        return [
            _Unit(
                texts=[" ".join(w)],
                heading_path=unit.heading_path,
                anchor=unit.anchor,
                label=unit.label,
                page_from=unit.page_from,
                page_to=unit.page_to,
                tokens=self._tok.count(" ".join(w)),
            )
            for w in windows
        ]

    def _split_table(self, unit: _Unit) -> list[_Unit]:
        lines = unit.texts[0].split("\n")
        header, rows = lines[:2], lines[2:]
        header_tokens = self._tok.count("\n".join(header))
        out: list[_Unit] = []
        batch: list[str] = []
        batch_tokens = header_tokens
        for row in rows:
            row_tokens = self._tok.count(row)
            if batch and batch_tokens + row_tokens > self.max_tokens:
                out.append(self._table_part(unit, header, batch))
                batch, batch_tokens = [], header_tokens
            batch.append(row)
            batch_tokens += row_tokens
        if batch:
            out.append(self._table_part(unit, header, batch))
        return out

    def _table_part(self, unit: _Unit, header: list[str], rows: list[str]) -> _Unit:
        text = "\n".join([*header, *rows])
        return _Unit(
            texts=[text],
            heading_path=unit.heading_path,
            anchor=unit.anchor,
            label=None,
            page_from=unit.page_from,
            page_to=unit.page_to,
            tokens=self._tok.count(text),
            has_table=True,
        )

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        units = [piece for unit in self._units(elements) for piece in self._split_unit(unit)]
        chunks: list[Chunk] = []
        pending = _Pending()

        def flush() -> None:
            if not pending.units:
                return
            text = "\n\n".join("\n".join(u.texts) for u in pending.units)
            first = pending.units[0]
            page_from, page_to = _pages(pending.units)
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    text=text,
                    token_count=self._tok.count(text),
                    page_from=page_from,
                    page_to=page_to,
                    anchor=first.anchor,
                    heading_path=list(first.heading_path),
                    clause_ref=_clause_ref([u.label or "" for u in pending.units]),
                    has_table=any(u.has_table for u in pending.units),
                )
            )
            pending.clear()

        for unit in units:
            if pending.units:
                same_section = unit.heading_path == pending.units[0].heading_path
                same_chapter = unit.heading_path[:1] == pending.units[0].heading_path[:1]
                fits = pending.tokens + unit.tokens <= self.max_tokens
                small = pending.tokens < self.min_tokens
                if (
                    not fits
                    or not (same_section or (small and same_chapter))
                    or pending.tokens >= self.target_tokens
                ):
                    flush()
            pending.units.append(unit)
            pending.tokens += unit.tokens
        flush()
        return chunks


class FixedChunker:
    """Baseline: flattens the document and cuts fixed token windows, ignoring structure."""

    strategy = "fixed"

    def __init__(
        self,
        name: str = "fixed-512",
        size_tokens: int = 512,
        overlap_tokens: int = 64,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if not 0 <= overlap_tokens < size_tokens:
            raise ValueError("expected 0 <= overlap_tokens < size_tokens")
        self.name = name
        self.size_tokens = size_tokens
        self.overlap_tokens = overlap_tokens
        self._tok = tokenizer or get_tokenizer()

    def params(self) -> dict[str, int]:
        return {"size_tokens": self.size_tokens, "overlap_tokens": self.overlap_tokens}

    def chunk(self, elements: list[Element]) -> list[Chunk]:
        words: list[tuple[str, int, Element]] = []
        for element in elements:
            for word in element.text.split():
                words.append((word, self._tok.count(" " + word), element))

        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end = start
            tokens = 0
            while end < len(words) and tokens + words[end][1] <= self.size_tokens:
                tokens += words[end][1]
                end += 1
            end = max(end, start + 1)
            window = words[start:end]
            text = " ".join(w for w, _, _ in window)
            pages = [e.page for _, _, e in window if e.page is not None]
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    text=text,
                    token_count=self._tok.count(text),
                    page_from=min(pages) if pages else None,
                    page_to=max(pages) if pages else None,
                    anchor=next((e.anchor for _, _, e in window if e.anchor), None),
                    heading_path=[],
                    clause_ref=None,
                    has_table=any(e.kind == "table" for _, _, e in window),
                )
            )
            if end >= len(words):
                break
            back = 0
            back_tokens = 0
            while (
                back < end - start - 1
                and back_tokens + words[end - 1 - back][1] <= self.overlap_tokens
            ):
                back_tokens += words[end - 1 - back][1]
                back += 1
            start = end - back
        return chunks


CHUNKERS: dict[str, type[StructureChunker] | type[FixedChunker]] = {
    "structure": StructureChunker,
    "fixed": FixedChunker,
}


def chunker_from_name(name: str) -> StructureChunker | FixedChunker:
    """Resolve a chunk-set name such as ``structure-v1`` or ``fixed-512``."""
    if name.startswith("fixed"):
        size = (
            int(name.split("-", 1)[1]) if "-" in name and name.split("-", 1)[1].isdigit() else 512
        )
        return FixedChunker(name=name, size_tokens=size, overlap_tokens=max(16, size // 8))
    if name.startswith("structure"):
        return StructureChunker(name=name)
    raise ValueError(f"unknown chunk set {name!r}")
