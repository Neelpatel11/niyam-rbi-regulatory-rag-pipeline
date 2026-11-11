from datetime import date

import pytest

from niyam.retrieval.search import TimeFilter, reciprocal_rank_fusion
from tests.test_generation import chunk


def test_rrf_rewards_agreement() -> None:
    dense = [chunk(1), chunk(2), chunk(3)]
    for rank, c in enumerate(dense, start=1):
        c.dense_rank = rank
    lexical = [chunk(3), chunk(4)]
    for rank, c in enumerate(lexical, start=1):
        c.lexical_rank = rank
    fused = reciprocal_rank_fusion(dense, lexical, k=60)
    assert fused[0].chunk_id == "c3", "a chunk found by both retrievers should rank first"
    assert fused[0].dense_rank == 3 and fused[0].lexical_rank == 1
    assert {c.chunk_id for c in fused} == {"c1", "c2", "c3", "c4"}
    assert fused[0].fused_score == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_weights() -> None:
    fused = reciprocal_rank_fusion([chunk(1)], [chunk(2)], k=60, weights=[1.0, 2.0])
    assert fused[0].chunk_id == "c2"


def test_time_filter_sql() -> None:
    _, params = TimeFilter(as_of=date(2020, 1, 1)).clause()
    assert params == [date(2020, 1, 1), date(2020, 1, 1)]
    _, params = TimeFilter(include_withdrawn=True).clause()
    assert params == []
