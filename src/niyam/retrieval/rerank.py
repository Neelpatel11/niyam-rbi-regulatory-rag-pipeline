from __future__ import annotations

import asyncio
import math
from typing import Protocol

from niyam.embeddings import embedding_input
from niyam.models import RetrievedChunk


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


def _passage(chunk: RetrievedChunk) -> str:
    return embedding_input(chunk.title, " > ".join(chunk.heading_path), chunk.text)


class CrossEncoderReranker:
    """Local ONNX cross-encoder. Scores are logits; we store sigmoid(logit) in [0, 1]."""

    def __init__(self, model: str, cache_dir: str | None = None, batch_size: int = 64) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = f"fastembed:{model}"
        self._model = TextCrossEncoder(model_name=model, cache_dir=cache_dir)
        self._batch_size = batch_size
        self._lock = asyncio.Lock()

    def _score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, passages, batch_size=self._batch_size)]

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        passages = [_passage(c) for c in chunks]
        async with self._lock:
            scores = await asyncio.to_thread(self._score, query, passages)
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = sigmoid(score)
        return sorted(chunks, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))


class LexicalOverlapReranker:
    """Dependency-free reranker for tests: fraction of query terms present in the passage."""

    name = "overlap"

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        for chunk in chunks:
            passage = _passage(chunk).lower()
            hits = sum(1 for t in terms if t in passage)
            chunk.rerank_score = hits / len(terms) if terms else 0.0
        return sorted(chunks, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
