from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

from niyam.models import Element

log = logging.getLogger(__name__)

PARSER_VERSION = "pdf-v1"

_CHAPTER_RE = re.compile(
    r"^(chapter|part|annex(?:ure)?|appendix|schedule)\s*[-–—:]?\s*([ivxlcdm]+|\d+|[a-z])?\b",
    re.I,
)
_SECTION_LETTER_RE = re.compile(r"^[A-Z]\.\s+\S")
_CLAUSE_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+(?=\S)")
_SUBCLAUSE_RE = re.compile(r"^\(([a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)\s+")
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
_DIGITS_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"[ \t ]+")


@dataclass(slots=True)
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float


@dataclass(slots=True)
class _Block:
    page: int
    lines: list[_Line]
    bbox: tuple[float, float, float, float]

    @property
    def text(self) -> str:
        return _join_lines([line.text for line in self.lines])

    @property
    def size(self) -> float:
        return max((line.size for line in self.lines), default=0.0)

    @property
    def bold(self) -> bool:
        return bool(self.lines) and all(line.bold for line in self.lines)


def _join_lines(lines: list[str]) -> str:
    out: list[str] = []
    for raw in lines:
        line = _SPACE_RE.sub(" ", raw).strip()
        if not line:
            continue
        if out and out[-1].endswith("-") and line[:1].islower():
            out[-1] = out[-1][:-1] + line
        else:
            out.append(line)
    return " ".join(out)


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    overlap_w = min(ax1, bx1) - max(ax0, bx0)
    overlap_h = min(ay1, by1) - max(ay0, by0)
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    return (overlap_w * overlap_h) / area > 0.5


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [
        [_SPACE_RE.sub(" ", (cell or "").replace("\n", " ")).strip() for cell in row]
        for row in rows
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _normalize_margin_text(text: str) -> str:
    return _DIGITS_RE.sub("#", text.lower()).strip()


class PDFParser:
    def __init__(self, extract_tables: bool = True, margin_ratio: float = 0.08) -> None:
        self._extract_tables = extract_tables
        self._margin_ratio = margin_ratio

    def parse(self, data: bytes) -> tuple[list[Element], int, int]:
        """Return (elements, page_count, text_chars)."""
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
            blocks: list[_Block] = []
            tables: list[tuple[int, float, str]] = []
            margin_counts: Counter[str] = Counter()
            page_heights: dict[int, float] = {}

            for page_index in range(page_count):
                page = doc[page_index]
                page_no = page_index + 1
                page_heights[page_no] = page.rect.height
                table_boxes: list[tuple[float, float, float, float]] = []
                if self._extract_tables:
                    table_boxes = self._collect_tables(page, page_no, tables)
                for block in self._page_blocks(page, page_no):
                    if any(_intersects(block.bbox, box) for box in table_boxes):
                        continue
                    blocks.append(block)
                    if self._in_margin(block, page.rect.height):
                        margin_counts[_normalize_margin_text(block.text)] += 1

        repeated = {
            text
            for text, count in margin_counts.items()
            if page_count >= 3 and count >= max(3, int(page_count * 0.5))
        }
        kept = [
            b
            for b in blocks
            if b.text
            and not _PAGE_NUMBER_RE.match(b.text)
            and not (
                self._in_margin(b, page_heights[b.page])
                and _normalize_margin_text(b.text) in repeated
            )
        ]
        body_size = self._body_font_size(kept)
        elements = self._classify(kept, tables, body_size)
        text_chars = sum(len(e.text) for e in elements)
        return elements, page_count, text_chars

    def _in_margin(self, block: _Block, height: float) -> bool:
        margin = height * self._margin_ratio
        return block.bbox[3] <= margin or block.bbox[1] >= height - margin

    def _collect_tables(
        self,
        page: pymupdf.Page,
        page_no: int,
        sink: list[tuple[int, float, str]],
    ) -> list[tuple[float, float, float, float]]:
        boxes: list[tuple[float, float, float, float]] = []
        try:
            found = page.find_tables()
        except Exception:
            log.debug("table detection failed on page %d", page_no, exc_info=True)
            return boxes
        for table in found.tables:
            rows = table.extract()
            if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                continue
            markdown = table_to_markdown(rows)
            if markdown:
                bbox = tuple(table.bbox)
                boxes.append(bbox)
                sink.append((page_no, bbox[1], markdown))
        return boxes

    def _page_blocks(self, page: pymupdf.Page, page_no: int) -> list[_Block]:
        data = page.get_text("dict", sort=True)
        out: list[_Block] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[_Line] = []
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans)
                size = max(float(s.get("size", 0.0)) for s in spans)
                bold = all(
                    (int(s.get("flags", 0)) & 16) or "bold" in str(s.get("font", "")).lower()
                    for s in spans
                )
                y0, y1 = float(line["bbox"][1]), float(line["bbox"][3])
                lines.append(_Line(text=text, size=size, bold=bool(bold), y0=y0, y1=y1))
            if lines:
                bbox = tuple(float(v) for v in block["bbox"])
                out.append(_Block(page=page_no, lines=lines, bbox=bbox))  # type: ignore[arg-type]
        return out

    @staticmethod
    def _body_font_size(blocks: list[_Block]) -> float:
        weights: Counter[float] = Counter()
        for block in blocks:
            for line in block.lines:
                weights[round(line.size, 1)] += len(line.text)
        return weights.most_common(1)[0][0] if weights else 10.0

    def _classify(
        self,
        blocks: list[_Block],
        tables: list[tuple[int, float, str]],
        body_size: float,
    ) -> list[Element]:
        positioned: list[tuple[int, float, Element]] = []
        for block in blocks:
            text = block.text
            positioned.append(
                (block.page, block.bbox[1], self._element_for(block, text, body_size))
            )
        for page_no, y0, markdown in tables:
            positioned.append((page_no, y0, Element(kind="table", text=markdown, page=page_no)))
        positioned.sort(key=lambda item: (item[0], item[1]))
        return [element for _, _, element in positioned]

    @staticmethod
    def _element_for(block: _Block, text: str, body_size: float) -> Element:
        short = len(text) <= 140 and not text.endswith((".", ";", ":", ","))
        larger = block.size >= body_size + 1.0
        chapter = _CHAPTER_RE.match(text) is not None and len(text) <= 160

        if chapter or (short and larger and block.size >= body_size + 3.0):
            return Element(kind="heading", text=text, page=block.page, level=1)
        if short and (larger or block.bold or _SECTION_LETTER_RE.match(text)) and len(text) >= 3:
            label = None
            clause = _CLAUSE_RE.match(text)
            if clause:
                label = clause.group(1)
            return Element(kind="heading", text=text, page=block.page, level=2, label=label)

        label = None
        clause = _CLAUSE_RE.match(text)
        if clause:
            label = clause.group(1)
        else:
            sub = _SUBCLAUSE_RE.match(text)
            if sub:
                label = f"({sub.group(1)})"
        return Element(kind="paragraph", text=text, page=block.page, label=label)
