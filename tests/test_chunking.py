import pytest

from niyam.ingest.chunking import FixedChunker, StructureChunker, chunker_from_name
from niyam.ingest.local import load_local_document, parse_markdown
from niyam.models import Element
from niyam.text import Tokenizer
from tests.conftest import CORPUS


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer()


def _para(text: str, label: str | None = None) -> Element:
    return Element(kind="paragraph", text=text, label=label)


def test_structure_chunker_respects_headings(tok: Tokenizer) -> None:
    elements = [
        Element(kind="heading", text="Chapter I", level=1, anchor="c1"),
        Element(kind="heading", text="A. Scope", level=2, anchor="a"),
        _para("1. First clause about scope.", "1"),
        Element(kind="heading", text="B. Limits", level=2, anchor="b"),
        _para("2. Second clause about limits.", "2"),
        _para("(a) sub clause one.", "(a)"),
    ]
    chunker = StructureChunker(target_tokens=40, max_tokens=60, min_tokens=5, tokenizer=tok)
    chunks = chunker.chunk(elements)
    assert [c.heading_path for c in chunks] == [
        ["Chapter I", "A. Scope"],
        ["Chapter I", "B. Limits"],
    ]
    assert chunks[1].clause_ref == "2"
    assert chunks[1].anchor == "b"
    assert "(a) sub clause one." in chunks[1].text


def test_structure_chunker_merges_tiny_sections_in_same_chapter(tok: Tokenizer) -> None:
    elements = [
        Element(kind="heading", text="Chapter I", level=1),
        Element(kind="heading", text="A", level=2),
        _para("1. Tiny.", "1"),
        Element(kind="heading", text="B", level=2),
        _para("2. Also tiny.", "2"),
    ]
    chunks = StructureChunker(
        target_tokens=100, max_tokens=200, min_tokens=50, tokenizer=tok
    ).chunk(elements)
    assert len(chunks) == 1
    assert chunks[0].clause_ref == "1-2"


def test_long_clause_is_split_with_overlap(tok: Tokenizer) -> None:
    sentences = " ".join(
        f"Sentence number {i} explains a separate requirement in detail." for i in range(80)
    )
    chunker = StructureChunker(
        target_tokens=120, max_tokens=200, min_tokens=20, overlap_tokens=30, tokenizer=tok
    )
    chunks = chunker.chunk([_para("5. " + sentences, "5")])
    assert len(chunks) > 3
    assert all(c.token_count <= 200 for c in chunks)
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.split()[0] in chunks[1].text, "consecutive windows should overlap"


def test_large_table_split_repeats_header(tok: Tokenizer) -> None:
    rows = "\n".join(f"| item {i} | value {i} with some descriptive text |" for i in range(120))
    table = "| Item | Value |\n| --- | --- |\n" + rows
    chunks = StructureChunker(
        target_tokens=150, max_tokens=300, min_tokens=20, tokenizer=tok
    ).chunk([Element(kind="table", text=table)])
    assert len(chunks) > 1
    assert all(c.text.startswith("| Item | Value |\n| --- | --- |") for c in chunks)
    assert all(c.has_table for c in chunks)


def test_fixed_chunker_windows(tok: Tokenizer) -> None:
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = FixedChunker(size_tokens=200, overlap_tokens=20, tokenizer=tok).chunk([_para(text)])
    assert len(chunks) > 5
    assert all(c.token_count <= 210 for c in chunks)
    assert all(c.heading_path == [] for c in chunks)


def test_chunker_from_name() -> None:
    assert isinstance(chunker_from_name("fixed-256"), FixedChunker)
    assert chunker_from_name("fixed-256").size_tokens == 256
    assert isinstance(chunker_from_name("structure-v1"), StructureChunker)
    with pytest.raises(ValueError, match="unknown"):
        chunker_from_name("semantic")


def test_markdown_loader_reads_front_matter() -> None:
    meta, elements, pages, parser = load_local_document(CORPUS / "fx-kyc-2016.md")
    assert meta.source_key == "fx-kyc-2016"
    assert meta.withdrawn_on is not None
    assert parser == "markdown-v1" and pages is None
    assert any(e.kind == "heading" and e.anchor == "p3" for e in elements)


def test_markdown_tables_and_clauses() -> None:
    elements = parse_markdown(
        "# Title\n\n1. Clause one\ncontinues here.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    )
    assert [e.kind for e in elements] == ["heading", "paragraph", "table"]
    assert elements[1].text == "1. Clause one continues here."
    assert elements[1].label == "1"
