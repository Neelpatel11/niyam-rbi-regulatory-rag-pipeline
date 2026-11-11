from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import math
import re
from collections import OrderedDict
from typing import Any, Protocol, cast

import numpy as np

from niyam.config import Settings

log = logging.getLogger(__name__)

Vector = np.ndarray

QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
}
DOCUMENT_PREFIXES: dict[str, str] = {
    "nomic-ai/nomic-embed-text-v1.5": "search_document: ",
}


class Embedder(Protocol):
    name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


class _QueryCache:
    def __init__(self, size: int = 2048) -> None:
        self._size = size
        self._items: OrderedDict[str, Vector] = OrderedDict()

    def get(self, key: str) -> Vector | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: str, value: Vector) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._size:
            self._items.popitem(last=False)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed. No network after the first model download."""

    def __init__(self, model: str, batch_size: int = 32, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self._model_name = model
        self._batch_size = batch_size
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self.dim = int(self._model.embedding_size)
        self.name = f"fastembed:{model}"
        self._query_prefix = QUERY_PREFIXES.get(model, "")
        self._doc_prefix = DOCUMENT_PREFIXES.get(model, "")
        self._cache = _QueryCache()
        self._lock = asyncio.Lock()

    def _embed_sync(self, texts: list[str]) -> list[Vector]:
        return [
            _normalize(np.asarray(v, dtype=np.float32))
            for v in self._model.embed(texts, batch_size=self._batch_size)
        ]

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        prefixed = [self._doc_prefix + t for t in texts]
        async with self._lock:
            return await asyncio.to_thread(self._embed_sync, prefixed)

    async def embed_query(self, text: str) -> Vector:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        async with self._lock:
            [vector] = await asyncio.to_thread(self._embed_sync, [self._query_prefix + text])
        self._cache.put(text, vector)
        return vector


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str | None, batch_size: int = 128) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self.dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 1536}.get(model, 1536)
        self._dimensions = 1536 if model == "text-embedding-3-large" else None
        self.name = f"openai:{model}"
        self._cache = _QueryCache()

    async def _embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            kwargs: dict[str, Any] = {"model": self._model, "input": batch}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = await self._client.embeddings.create(**kwargs)
            out.extend(_normalize(np.asarray(d.embedding, dtype=np.float32)) for d in response.data)
        return out

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return await self._embed(texts) if texts else []

    async def embed_query(self, text: str) -> Vector:
        cached = self._cache.get(text)
        if cached is None:
            [cached] = await self._embed([text])
            self._cache.put(text, cached)
        return cached


class GeminiEmbedder:
    def __init__(
        self, model: str, api_key: str | None, dim: int = 768, batch_size: int = 64
    ) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self.dim = dim
        self.name = f"gemini:{model}@{dim}"
        self._cache = _QueryCache()

    async def _embed(self, texts: list[str], task_type: str) -> list[Vector]:
        from google.genai import types

        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            result = await self._client.aio.models.embed_content(
                model=self._model,
                contents=cast(Any, batch),
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dim
                ),
            )
            out.extend(
                _normalize(np.asarray(e.values, dtype=np.float32)) for e in result.embeddings or []
            )
        return out

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    async def embed_query(self, text: str) -> Vector:
        cached = self._cache.get(text)
        if cached is None:
            [cached] = await self._embed([text], "RETRIEVAL_QUERY")
            self._cache.put(text, cached)
        return cached


_HASH_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing. Used for tests and offline smoke runs only."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vector(self, text: str) -> Vector:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _HASH_TOKEN_RE.findall(text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log1p(len(gram)) / 4)
        return _normalize(vec)

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._vector(text)


def build_embedder(
    settings: Settings, model: str | None = None, provider: str | None = None
) -> Embedder:
    provider = provider or settings.embedding_provider
    model = model or settings.embedding_model
    if provider == "fastembed":
        return FastEmbedEmbedder(
            model,
            batch_size=settings.embedding_batch_size,
            cache_dir=str(settings.data_dir / "models"),
        )
    if provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIEmbedder(model, api_key=key)
    if provider == "gemini":
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        return GeminiEmbedder(model, api_key=key)
    if provider == "hashing":
        return HashingEmbedder(int(model) if model.isdigit() else 256)
    raise ValueError(f"unknown embedding provider {provider!r}")


def embedding_input(title: str, heading_text: str, text: str) -> str:
    header = title if not heading_text else f"{title}\n{heading_text}"
    return f"{header}\n\n{text}"
