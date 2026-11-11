from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from niyam.ingest.pdf import PDFParser
from niyam.models import DocumentMeta, Element

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{#([\w-]+)\})?\s*$")
_CLAUSE_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+(?=\S)")
_SUBCLAUSE_RE = re.compile(r"^\(([a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)\s+")


def _as_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def meta_from_mapping(data: dict[str, Any], default_key: str) -> DocumentMeta:
    amended = [d for d in (_as_date(v) for v in data.get("amended_on", []) or []) if d]
    return DocumentMeta(
        source_key=str(data.get("source_key") or default_key),
        source_url=str(data.get("source_url") or f"local://{default_key}"),
        title=str(data["title"]),
        pdf_url=data.get("pdf_url"),
        doc_type=str(data.get("doc_type") or "master_direction"),
        circular_no=data.get("circular_no"),
        reference_nos=list(data.get("reference_nos") or []),
        issued_on=_as_date(data.get("issued_on")),
        amended_on=sorted(amended),
        withdrawn_on=_as_date(data.get("withdrawn_on")),
        department=data.get("department"),
    )


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    _, front, body = text.split("---", 2)
    return yaml.safe_load(front) or {}, body.lstrip("\n")


def parse_markdown(body: str) -> list[Element]:
    elements: list[Element] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(line.strip() for line in paragraph).strip()
            paragraph.clear()
            if text:
                label = None
                clause = _CLAUSE_RE.match(text)
                if clause:
                    label = clause.group(1)
                elif sub := _SUBCLAUSE_RE.match(text):
                    label = f"({sub.group(1)})"
                elements.append(Element(kind="paragraph", text=text, label=label))

    def flush_table() -> None:
        if table:
            elements.append(Element(kind="table", text="\n".join(table)))
            table.clear()

    for raw in body.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("|"):
            flush_paragraph()
            table.append(line.strip())
            continue
        flush_table()
        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            hashes, text, anchor = heading.groups()
            clause = _CLAUSE_RE.match(text)
            elements.append(
                Element(
                    kind="heading",
                    text=text,
                    level=1 if len(hashes) <= 2 else 2,
                    label=clause.group(1) if clause else None,
                    anchor=anchor,
                )
            )
            continue
        if not line.strip():
            flush_paragraph()
            continue
        if _CLAUSE_RE.match(line.strip()) or _SUBCLAUSE_RE.match(line.strip()):
            flush_paragraph()
        paragraph.append(line)
    flush_paragraph()
    flush_table()
    return elements


def load_local_document(path: Path) -> tuple[DocumentMeta, list[Element], int | None, str]:
    """Load a Markdown file with YAML front matter, or a PDF with a sibling ``.yaml`` metadata file."""
    if path.suffix.lower() in {".md", ".markdown"}:
        front, body = split_front_matter(path.read_text(encoding="utf-8"))
        if "title" not in front:
            raise ValueError(f"{path} is missing a `title` in its front matter")
        return meta_from_mapping(front, path.stem), parse_markdown(body), None, "markdown-v1"

    if path.suffix.lower() == ".pdf":
        sidecar = path.with_suffix(".yaml")
        if not sidecar.exists():
            raise ValueError(f"{path} needs metadata in {sidecar.name}")
        meta = meta_from_mapping(yaml.safe_load(sidecar.read_text(encoding="utf-8")), path.stem)
        elements, page_count, _ = PDFParser().parse(path.read_bytes())
        return meta, elements, page_count, "pdf-v1"

    raise ValueError(f"unsupported file type: {path}")
