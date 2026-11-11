import pytest

from niyam.evals.dataset import Evidence, GoldenItem
from niyam.evals.metrics import (
    ChunkView,
    best_refusal_threshold,
    cohen_kappa,
    coverage,
    evidence_hits,
    ndcg_at_k,
    normalize_tokens,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def view(cid: str, text: str, key: str = "doc") -> ChunkView:
    return ChunkView(cid, key, tuple(normalize_tokens(text)))


def test_normalize_folds_unicode_and_punctuation() -> None:
    assert normalize_tokens("‘Officially’ valid—document, ₹50,000") == [
        "officially",
        "valid",
        "document",
        "50",
        "000",
    ]


def test_coverage_partial_across_boundary() -> None:
    quote = normalize_tokens("the balance at any point of time does not exceed rupees one lakh")
    left = normalize_tokens("conditions: the balance at any point of time does not")
    assert 0.5 < coverage(quote, left) < 1.0
    assert coverage(quote, quote) == 1.0


def test_ranking_metrics() -> None:
    evidence = [
        Evidence(source_key="doc", quote="aggregate credits do not exceed two lakh"),
        Evidence(source_key="doc", quote="balance does not exceed one lakh rupees"),
    ]
    ranked = [
        view("x", "unrelated text about deposits"),
        view("a", "the aggregate credits do not exceed two lakh in a year"),
        view("b", "the balance does not exceed one lakh rupees", key="other-doc"),
        view("c", "and the balance does not exceed one lakh rupees at any time"),
    ]
    hits = evidence_hits(evidence, ranked)
    assert hits[1][2] is False, "a matching quote in another document is not a hit"
    assert recall_at_k(hits, 1) == 0.0
    assert recall_at_k(hits, 2) == 0.5
    assert recall_at_k(hits, 4) == 1.0
    assert reciprocal_rank(hits) == 0.5
    assert 0.0 < ndcg_at_k(hits, 10) < 1.0


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.95) == 5


def test_cohen_kappa() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_best_threshold_separates_classes() -> None:
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, None]
    answerable = [True, True, True, False, False, False]
    threshold, counts = best_refusal_threshold(scores, answerable)
    assert 0.2 < threshold <= 0.7
    assert counts.f1 == 1.0


def test_golden_validation() -> None:
    with pytest.raises(ValueError, match="evidence"):
        GoldenItem(id="x", question="q?", category="single_fact", expected_answer="a")
    with pytest.raises(ValueError, match="must not carry evidence"):
        GoldenItem(
            id="y",
            question="q?",
            category="unanswerable",
            evidence=[Evidence(source_key="d", quote="some quote text here")],
        )
    item = GoldenItem(id="z", question="q?", category="unanswerable")
    assert item.split() in {"dev", "test"}
    assert item.split() == item.split()
