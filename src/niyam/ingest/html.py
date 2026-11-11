from __future__ import annotations

import re

from selectolax.lexbor import LexborHTMLParser, LexborNode

from niyam.ingest.pdf import table_to_markdown
from niyam.ingest.rbi import extract_reference_numbers, normalize_space, parse_rbi_date
from niyam.models import Element

PARSER_VERSION = "html-v1"

_CHAPTER_RE = re.compile(
    r"^(chapter|part|annex(?:ure)?|appendix|schedule)\s*[-–—:]?\s*([ivxlcdm]+|\d+|[a-z])?\b",
    re.I,
)
_CLAUSE_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+(?=\S)")
_SUBCLAUSE_RE = re.compile(r"^\(([a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)\s+")
_BLOCK_SELECTOR = "p, li, table, h1, h2, h3, h4, h5"
_SKIP_HEAD_RE = re.compile(
    r"(updated as on|^previous versions?$|^table of contents$|^contents$)", re.I
)


def _descendants(node: LexborNode, selector: str) -> list[LexborNode]:
    return [child for child in node.css(selector) if child.mem_id != node.mem_id]


def _anchor_of(node: LexborNode) -> str | None:
    for anchor in node.css("a[id], a[name]"):
        value = anchor.attributes.get("id") or anchor.attributes.get("name")
        if value:
            return value.strip()
    return None


def _is_toc_table(table: LexborNode) -> bool:
    if _descendants(table, "table") or len(_descendants(table, "p")) > 3:
        return False
    links = table.css("a[href^='#']")
    cells = table.css("td")
    return bool(cells) and len(links) >= max(3, int(len(cells) * 0.6))


def _is_data_table(table: LexborNode) -> bool:
    if _descendants(table, "table"):
        return False
    rows = table.css("tr")
    if len(rows) < 2:
        return False
    return max(len(row.css("td, th")) for row in rows) >= 2


def _table_rows(table: LexborNode) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for row in table.css("tr"):
        rows.append([normalize_space(cell.text(separator=" ")) for cell in row.css("td, th")])
    return rows


def _inside(node: LexborNode, container_ids: set[int], root_id: int) -> bool:
    parent = node.parent
    while parent is not None and parent.mem_id != root_id:
        if parent.mem_id in container_ids:
            return True
        parent = parent.parent
    return False


def _is_heading(node: LexborNode, text: str) -> bool:
    if node.tag in {"h1", "h2", "h3", "h4", "h5"}:
        return True
    classes = (node.attributes.get("class") or "").split()
    if "head" in classes:
        return True
    if len(text) > 160:
        return False
    strong = node.css("b, strong")
    if strong:
        strong_text = normalize_space(" ".join(s.text() for s in strong))
        return len(strong_text) >= 0.9 * len(text)
    return False


_ROMAN = [
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
]


def _roman(value: int) -> str:
    out = []
    for number, numeral in _ROMAN:
        while value >= number:
            out.append(numeral)
            value -= number
    return "".join(out)


def _alpha(value: int) -> str:
    out = ""
    while value > 0:
        value, rem = divmod(value - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def ordered_list_label(list_type: str | None, position: int) -> str:
    kind = (list_type or "1").strip()
    if kind == "a":
        return f"({_alpha(position)})"
    if kind == "A":
        return f"({_alpha(position).upper()})"
    if kind == "i":
        return f"({_roman(position)})"
    if kind == "I":
        return f"({_roman(position).upper()})"
    return f"{position}."


def _list_item_label(node: LexborNode) -> str | None:
    parent = node.parent
    if parent is None or parent.tag != "ol":
        return None
    try:
        start = int(parent.attributes.get("start") or 1)
    except ValueError:
        start = 1
    position = start
    sibling = parent.child
    while sibling is not None:
        if sibling.mem_id == node.mem_id:
            return ordered_list_label(parent.attributes.get("type"), position)
        if sibling.tag == "li":
            position += 1
        sibling = sibling.next
    return None


def _is_front_matter(text: str, normalized_title: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(normalized_title[:60]) and len(text) <= len(normalized_title) + 60:
        return True
    if _SKIP_HEAD_RE.search(text):
        return True
    if extract_reference_numbers(text) and len(text) <= 160:
        return True
    return bool(parse_rbi_date(text) and len(text) <= 40)


def parse_detail_body(body_html: str, title: str) -> list[Element]:
    tree = LexborHTMLParser(body_html)
    root = tree.body
    if root is None:
        return []
    root_id = root.mem_id
    for sup in root.css("sup"):
        if normalize_space(sup.text()).isdigit():
            sup.decompose()

    tables = [t for t in root.css("table") if _is_data_table(t) or _is_toc_table(t)]
    table_ids = {t.mem_id for t in tables}

    elements: list[Element] = []
    pending_anchor: str | None = None
    normalized_title = normalize_space(title).lower()

    for node in root.css(_BLOCK_SELECTOR):
        if node.tag == "table":
            if node.mem_id not in table_ids or _inside(node, table_ids, root_id):
                continue
            if _is_toc_table(node):
                continue
            markdown = table_to_markdown(_table_rows(node))
            if markdown:
                elements.append(Element(kind="table", text=markdown, anchor=pending_anchor))
                pending_anchor = None
            continue

        if _inside(node, table_ids, root_id):
            continue
        if node.tag in {"p", "li"} and _descendants(node, "p, li, table"):
            continue

        text = normalize_space(node.text(separator=" "))
        anchor = _anchor_of(node)
        if not text:
            pending_anchor = anchor or pending_anchor
            continue

        if _is_front_matter(text, normalized_title) and not elements:
            continue

        if _is_heading(node, text):
            if _is_front_matter(text, normalized_title):
                continue
            align = (node.attributes.get("align") or "").lower()
            level = 1 if align == "center" or _CHAPTER_RE.match(text) else 2
            clause = _CLAUSE_RE.match(text)
            elements.append(
                Element(
                    kind="heading",
                    text=text,
                    level=level,
                    label=clause.group(1) if clause else None,
                    anchor=anchor or pending_anchor,
                )
            )
            pending_anchor = None
            continue

        if node.tag == "li":
            list_label = _list_item_label(node)
            if list_label and not text.lower().startswith(list_label.strip("()").lower()):
                text = f"{list_label} {text}"

        label: str | None = None
        clause = _CLAUSE_RE.match(text)
        if clause:
            label = clause.group(1)
        else:
            sub = _SUBCLAUSE_RE.match(text)
            if sub:
                label = f"({sub.group(1)})"
        elements.append(
            Element(kind="paragraph", text=text, label=label, anchor=anchor or pending_anchor)
        )
        pending_anchor = None

    return elements
