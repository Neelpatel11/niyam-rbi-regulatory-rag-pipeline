from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from niyam.config import Settings
from niyam.db import Database
from niyam.db.repository import lookup_ids
from niyam.embeddings import Embedder
from niyam.models import RetrievedChunk
from niyam.retrieval.rerank import Reranker
from niyam.retrieval.search import (
    TimeFilter,
    dense_search,
    lexical_search,
    reciprocal_rank_fusion,
)


@dataclass(slots=True, frozen=True)
class RetrievalConfig:
    chunk_set: str = "structure-v1"
    use_dense: bool = True
    use_lexical: bool = True
    use_rerank: bool = True
    dense_k: int = 50
    lexical_k: int = 50
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rerank_candidates: int = 20
    context_k: int = 8
    max_context_tokens: int = 6000

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalConfig:
        return cls(
            chunk_set=settings.chunk_set,
            use_rerank=settings.rerank_enabled,
            dense_k=settings.dense_k,
            lexical_k=settings.lexical_k,
            rrf_k=settings.rrf_k,
            rerank_candidates=settings.rerank_candidates,
            context_k=settings.context_k,
            max_context_tokens=settings.max_context_tokens,
        )

    def with_overrides(self, **overrides: Any) -> RetrievalConfig:
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    candidates: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float | None:
        if not self.chunks:
            return None
        first = self.chunks[0]
        if first.rerank_score is not None:
            return first.rerank_score
        return first.dense_score


def _within_budget(chunks: list[RetrievedChunk], limit: int, budget: int) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        if len(kept) >= limit:
            break
        if kept and used + chunk.token_count > budget:
            continue
        kept.append(chunk)
        used += chunk.token_count
    return kept


class HybridRetriever:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        reranker: Reranker | None,
        config: RetrievalConfig,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._reranker = reranker
        self.config = config
        self._ids: dict[str, tuple[int, int, int]] = {}

    async def _resolve(self, chunk_set: str) -> tuple[int, int, int]:
        cached = self._ids.get(chunk_set)
        if cached is None:
            async with self._db.connection() as conn:
                cached = await lookup_ids(conn, chunk_set, self._embedder.name)
            self._ids[chunk_set] = cached
        return cached

    async def _dense(
        self, question: str, chunk_set_id: int, model_id: int, dim: int, tf: TimeFilter, k: int
    ) -> tuple[list[RetrievedChunk], float, float]:
        start = time.perf_counter()
        vector = await self._embedder.embed_query(question)
        embedded = time.perf_counter()
        async with self._db.connection() as conn:
            hits = await dense_search(
                conn,
                query_vector=vector,
                model_id=model_id,
                chunk_set_id=chunk_set_id,
                dim=dim,
                k=k,
                time_filter=tf,
            )
        return hits, (embedded - start) * 1000, (time.perf_counter() - embedded) * 1000

    async def _lexical(
        self, question: str, chunk_set_id: int, tf: TimeFilter, k: int
    ) -> tuple[list[RetrievedChunk], float]:
        start = time.perf_counter()
        async with self._db.connection() as conn:
            hits = await lexical_search(
                conn, question=question, chunk_set_id=chunk_set_id, k=k, time_filter=tf
            )
        return hits, (time.perf_counter() - start) * 1000

    async def retrieve(
        self,
        question: str,
        *,
        time_filter: TimeFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = config or self.config
        tf = time_filter or TimeFilter()
        if not (cfg.use_dense or cfg.use_lexical):
            raise ValueError("at least one of dense or lexical retrieval must be enabled")
        chunk_set_id, model_id, dim = await self._resolve(cfg.chunk_set)
        timings: dict[str, float] = {}

        tasks: list[asyncio.Task[Any]] = []
        if cfg.use_dense:
            tasks.append(
                asyncio.create_task(
                    self._dense(question, chunk_set_id, model_id, dim, tf, cfg.dense_k)
                )
            )
        if cfg.use_lexical:
            tasks.append(
                asyncio.create_task(self._lexical(question, chunk_set_id, tf, cfg.lexical_k))
            )
        results = await asyncio.gather(*tasks)

        ranked_lists: list[list[RetrievedChunk]] = []
        weights: list[float] = []
        index = 0
        if cfg.use_dense:
            dense_hits, embed_ms, dense_ms = results[index]
            index += 1
            timings["embed_query"] = embed_ms
            timings["dense_search"] = dense_ms
            ranked_lists.append(dense_hits)
            weights.append(cfg.dense_weight)
        if cfg.use_lexical:
            lexical_hits, lexical_ms = results[index]
            timings["lexical_search"] = lexical_ms
            ranked_lists.append(lexical_hits)
            weights.append(cfg.lexical_weight)

        start = time.perf_counter()
        fused = reciprocal_rank_fusion(*ranked_lists, k=cfg.rrf_k, weights=weights)
        timings["fusion"] = (time.perf_counter() - start) * 1000

        ordered = fused
        if cfg.use_rerank and self._reranker is not None:
            start = time.perf_counter()
            head = await self._reranker.rerank(question, fused[: cfg.rerank_candidates])
            ordered = head + fused[cfg.rerank_candidates :]
            timings["rerank"] = (time.perf_counter() - start) * 1000

        selected = _within_budget(ordered, cfg.context_k, cfg.max_context_tokens)
        return RetrievalResult(chunks=selected, candidates=ordered, timings_ms=timings)
