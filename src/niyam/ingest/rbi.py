from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import parse_qs, urljoin, urlsplit

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from niyam.ingest.fetcher import PoliteFetcher
from niyam.models import DocumentMeta

log = logging.getLogger(__name__)

RBI_BASE = "https://www.rbi.org.in/Scripts/"
MASTER_DIRECTIONS_INDEX = urljoin(RBI_BASE, "BS_ViewMasDirections.aspx")

_DATE_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(\d{1,2}),\s*(\d{4})\b"
)
_UPDATED_RE = re.compile(r"Updated\s+as\s+on\s+([A-Za-z]+\.?\s+\d{1,2},\s*\d{4})", re.I)
_WITHDRAWN_RE = re.compile(r"Withdrawn(\d{2})(\d{2})(\d{4})")
_REFERENCE_RE = re.compile(
    r"\b(?:RBI/[A-Za-z&.\-]+/\d{4}-\d{2}/\d+|"
    r"[A-Z][A-Za-z&]*(?:\.[A-Za-z&()]+)*\.No\.\s?[A-Za-z]*\.?\d+/[\d.]+/\d{4}-\d{2})"
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def parse_rbi_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    month, day, year = match.groups()
    candidate = f"{month.rstrip('.')} {int(day):02d}, {year}"
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    if month.lower().startswith("sept"):
        return datetime.strptime(f"Sep {int(day):02d}, {year}", "%b %d, %Y").date()
    return None


def extract_reference_numbers(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _REFERENCE_RE.finditer(text):
        seen.setdefault(match.group(0).replace(" ", ""), None)
    return list(seen)


def source_key_for(url: str) -> str:
    query = parse_qs(urlsplit(url).query)
    ident = (query.get("id") or query.get("Id") or [""])[0]
    if not ident:
        raise ValueError(f"cannot derive a source key from {url}")
    return f"rbi-md-{ident}"


@dataclass(slots=True)
class IndexEntry:
    source_key: str
    url: str
    title: str
    department: str | None
    listed_on: date | None
    pdf_url: str | None


def parse_master_direction_index(html: str, base_url: str = RBI_BASE) -> list[IndexEntry]:
    tree = HTMLParser(html)
    entries: dict[str, IndexEntry] = {}
    department: str | None = None
    listed_on: date | None = None

    for row in tree.css("table.tablebg tr"):
        header = row.css_first("td.tableheader")
        if header is not None and row.css_first("a.link2") is None:
            label = normalize_space(header.text())
            parsed = parse_rbi_date(label)
            if parsed and len(label) <= 14:
                listed_on = parsed
            elif label:
                department = label
            continue

        link = row.css_first("a.link2")
        if link is None:
            continue
        href = (link.attributes.get("href") or "").strip()
        if "BS_ViewMasDirections.aspx" not in href or "id=" not in href:
            continue
        url = urljoin(base_url, href)
        try:
            key = source_key_for(url)
        except ValueError:
            continue
        pdf_node = row.css_first("a[id^='APDF_']")
        pdf_url = (pdf_node.attributes.get("href") or "").strip() if pdf_node else None
        entries.setdefault(
            key,
            IndexEntry(
                source_key=key,
                url=url,
                title=normalize_space(link.text()),
                department=department,
                listed_on=listed_on,
                pdf_url=pdf_url or None,
            ),
        )
    return list(entries.values())


@dataclass(slots=True)
class DetailPage:
    meta: DocumentMeta
    body_html: str | None


def _content_root(tree: HTMLParser) -> Node | None:
    for row in tree.css("tr.tablecontent2"):
        return row
    return tree.css_first("div#annual") or tree.body


def parse_master_direction_detail(
    html: str, url: str, fallback: IndexEntry | None = None
) -> DetailPage:
    tree = HTMLParser(html)

    title_node = tree.css_first("td.tableheader[align='center'] b") or tree.css_first(
        "td.tableheader b"
    )
    raw_title = normalize_space(title_node.text()) if title_node else ""
    title = re.sub(r"\s*\(Updated as on [^)]*\)\s*$", "", raw_title, flags=re.I) or (
        fallback.title if fallback else url
    )

    pdf_node = tree.css_first("a[id^='APDF_']")
    pdf_url = (pdf_node.attributes.get("href") or "").strip() if pdf_node else None
    if not pdf_url and fallback:
        pdf_url = fallback.pdf_url

    root = _content_root(tree)
    scope = root if root is not None else tree.body
    head_texts = [
        normalize_space(p.text(separator=" ")) for p in (scope.css("p") if scope else [])[:8]
    ]
    head_texts = [t for t in head_texts if t]
    header_blob = " ".join(head_texts[:6])

    reference_nos = extract_reference_numbers(header_blob)
    circular_no = reference_nos[-1] if reference_nos else None

    issued_on: date | None = None
    for text in head_texts[:6]:
        stripped = _UPDATED_RE.sub("", text)
        parsed = parse_rbi_date(stripped)
        if parsed:
            issued_on = parsed
            break
    if issued_on is None and fallback:
        issued_on = fallback.listed_on

    amended = sorted(
        {
            d
            for d in (parse_rbi_date(m) for m in _UPDATED_RE.findall(raw_title + " " + header_blob))
            if d
        }
    )

    withdrawn_on: date | None = None
    withdrawn = _WITHDRAWN_RE.search(html)
    if withdrawn:
        day, month, year = (int(x) for x in withdrawn.groups())
        try:
            withdrawn_on = date(year, month, day)
        except ValueError:
            log.warning("unparseable withdrawal marker %s on %s", withdrawn.group(0), url)

    meta = DocumentMeta(
        source_key=source_key_for(url),
        source_url=url,
        title=title,
        pdf_url=pdf_url or None,
        circular_no=circular_no,
        reference_nos=reference_nos,
        issued_on=issued_on,
        amended_on=amended,
        withdrawn_on=withdrawn_on,
        department=fallback.department if fallback else None,
    )
    return DetailPage(meta=meta, body_html=root.html if root is not None else None)


class RBIMasterDirections:
    def __init__(self, fetcher: PoliteFetcher) -> None:
        self._fetcher = fetcher

    async def list_entries(self, *, refresh: bool = False) -> list[IndexEntry]:
        result = await self._fetcher.get(MASTER_DIRECTIONS_INDEX, use_cache=not refresh)
        return parse_master_direction_index(result.text())

    async def fetch_detail(self, entry: IndexEntry, *, refresh: bool = False) -> DetailPage:
        result = await self._fetcher.get(entry.url, use_cache=not refresh)
        return parse_master_direction_detail(result.text(), entry.url, fallback=entry)

    async def fetch_pdf(self, pdf_url: str, *, refresh: bool = False) -> bytes:
        result = await self._fetcher.get(pdf_url, use_cache=not refresh)
        return result.content
