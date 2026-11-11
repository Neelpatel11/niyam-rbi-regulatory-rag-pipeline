from datetime import date

from niyam.ingest.html import ordered_list_label, parse_detail_body
from niyam.ingest.rbi import (
    extract_reference_numbers,
    parse_master_direction_detail,
    parse_master_direction_index,
    parse_rbi_date,
    source_key_for,
)

INDEX_HTML = """
<table class="tablebg">
<tr><td class="tableheader" colspan="4"><b>Commercial Banks</b></td></tr>
<tr><td class="tableheader" colspan="4"><b>Nov 28, 2025</b></td></tr>
<tr><td><a class="link2" href=BS_ViewMasDirections.aspx?id=13141> Reserve Bank of India (Commercial Banks – Know Your Customer) Directions, 2025</a></td>
<td><a id='APDF_169MD' href='https://rbidocs.rbi.org.in/rdocs/notification/PDFs/169MD.PDF'>pdf</a></td></tr>
<tr><td class="tableheader" colspan="4"><b>Consumer Education and Protection</b></td></tr>
<tr><td class="tableheader" colspan="4"><b>Jan 14, 2026</b></td></tr>
<tr><td><a class="link2" href=BS_ViewMasDirections.aspx?id=13271> Internal Ombudsman Directions, 2026</a></td></tr>
<tr><td><a class="link2" href=BS_ViewMasDirections.aspx?id=13271> duplicate row</a></td></tr>
</table>
"""

DETAIL_HTML = """
<table><tr><td class="tableheader" align="left"><a id='APDF_X' href="https://rbidocs.rbi.org.in/x.PDF">pdf</a></td></tr>
<tr><td align="center" class="tableheader"><b>Master Direction - Know Your Customer (KYC) Direction, 2016 (Updated as on August 14, 2025)</b></td></tr>
<tr class="tablecontent2"><td>
<table class="td" style="background: url(https://www.rbi.org.in/scripts/images/Withdrawn04122025.jpg);"><tr><td>
<p class="head">RBI/DBR/2015-16/18<br> Master Direction DBR.AML.BC.No.81/14.01.001/2015-16</p>
<p align="right" class="head">February 25, 2016<br>(Updated as on August 14, 2025)<br>(Updated as on May 29, 2019)</p>
<p align="center" class="head">Master Direction - Know Your Customer (KYC) Direction, 2016</p>
<table class="tablebg"><tr><td><a href="#1" class="links">CHAPTER I</a></td></tr>
<tr><td><a href="#2" class="links">Definitions</a></td></tr><tr><td><a href="#3" class="links">More</a></td></tr></table>
<p align="center" class="head"><a id="1"></a>CHAPTER I PRELIMINARY</p>
<p class="head"><a id="2"></a>1. Short Title and Commencement.</p>
<p>1. These Directions shall be called the KYC Directions.<sup>2</sup></p>
<ol type="a"><li>first item</li><li>second item</li></ol>
<table><tr><td>Type</td><td>Limit</td></tr><tr><td>Small account</td><td>Rs 50,000</td></tr></table>
</td></tr></table>
</td></tr></table>
"""


def test_parse_rbi_date_variants() -> None:
    assert parse_rbi_date("Updated as on Sept 4, 2024") == date(2024, 9, 4)
    assert parse_rbi_date("February 25, 2016") == date(2016, 2, 25)
    assert parse_rbi_date("Nov 28, 2025") == date(2025, 11, 28)
    assert parse_rbi_date("no date here") is None


def test_reference_numbers() -> None:
    text = "RBI/DOR/2025-26/169 DOR.AML.REC.No.88/14.01.002/2025-26 and again RBI/DOR/2025-26/169"
    assert extract_reference_numbers(text) == [
        "RBI/DOR/2025-26/169",
        "DOR.AML.REC.No.88/14.01.002/2025-26",
    ]


def test_source_key() -> None:
    assert (
        source_key_for("https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=13141")
        == "rbi-md-13141"
    )


def test_index_groups_and_dedupes() -> None:
    entries = parse_master_direction_index(INDEX_HTML)
    assert [e.source_key for e in entries] == ["rbi-md-13141", "rbi-md-13271"]
    first = entries[0]
    assert first.department == "Commercial Banks"
    assert first.listed_on == date(2025, 11, 28)
    assert first.pdf_url and first.pdf_url.endswith("169MD.PDF")
    assert entries[1].department == "Consumer Education and Protection"


def test_detail_metadata() -> None:
    page = parse_master_direction_detail(
        DETAIL_HTML, "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566"
    )
    meta = page.meta
    assert meta.title == "Master Direction - Know Your Customer (KYC) Direction, 2016"
    assert meta.circular_no == "DBR.AML.BC.No.81/14.01.001/2015-16"
    assert meta.issued_on == date(2016, 2, 25)
    assert meta.amended_on == [date(2019, 5, 29), date(2025, 8, 14)]
    assert meta.withdrawn_on == date(2025, 12, 4)


def test_detail_body_structure() -> None:
    page = parse_master_direction_detail(
        DETAIL_HTML, "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566"
    )
    assert page.body_html
    elements = parse_detail_body(page.body_html, page.meta.title)
    kinds = [(e.kind, e.text) for e in elements]
    assert ("heading", "CHAPTER I PRELIMINARY") in kinds
    assert all("RBI/DBR" not in text for _, text in kinds), "front matter leaked into the body"
    clause = next(e for e in elements if e.text.startswith("1. These Directions"))
    assert clause.label == "1"
    assert clause.text.endswith("KYC Directions."), "footnote markers must be stripped"
    assert [e.text for e in elements if e.text.endswith("item")] == [
        "(a) first item",
        "(b) second item",
    ]
    table = next(e for e in elements if e.kind == "table")
    assert "| Small account | Rs 50,000 |" in table.text
    heading = next(e for e in elements if e.text.startswith("1. Short Title"))
    assert heading.anchor == "2"


def test_ordered_list_labels() -> None:
    assert ordered_list_label("a", 28) == "(ab)"
    assert ordered_list_label("i", 4) == "(iv)"
    assert ordered_list_label("I", 9) == "(IX)"
    assert ordered_list_label(None, 3) == "3."


WRAPPED_TOC_HTML = """
<table class="td"><tr><td>
<p>RBI/DOR/2025-26/218<br> DOR.AML.REC.No.137/14.01.009/2025-26</p>
<table class="tablebg">
<tr><td><a href="#C1">Chapter I</a></td></tr><tr><td><a href="#C1A">A. Title</a></td></tr>
<tr><td><a href="#C1B">B. Applicability</a></td></tr><tr><td><a href="#C2">Chapter II</a></td></tr>
</table>
<p align="center" class="head"><a id="C1"></a>Chapter I - Preliminary</p>
<p>1. These Directions shall be called the Payments Banks Directions.</p>
</td></tr></table>
"""


def test_layout_table_holding_contents_is_not_a_contents_table() -> None:
    elements = parse_detail_body(WRAPPED_TOC_HTML, "Payments Banks Directions")
    texts = [e.text for e in elements]
    assert "1. These Directions shall be called the Payments Banks Directions." in texts
    assert all("Applicability" not in t for t in texts), "contents entries must be dropped"
