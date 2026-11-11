from __future__ import annotations

from dataclasses import dataclass

from niyam.config import Settings
from niyam.db import Database, apply_schema
from niyam.embeddings import Embedder, build_embedder
from niyam.generation.llm import LLMClient, build_llm
from niyam.generation.pricing import PriceBook
from niyam.qa import PipelineConfig, QAService
from niyam.retrieval import HybridRetriever, RetrievalConfig
from niyam.retrieval.rerank import CrossEncoderReranker, Reranker
from niyam.tracing import Tracer


@dataclass(slots=True)
class Services:
    settings: Settings
    db: Database
    embedder: Embedder
    reranker: Reranker | None
    retriever: HybridRetriever
    llm: LLMClient
    qa: QAService
    tracer: Tracer

    async def close(self) -> None:
        self.tracer.flush()
        await self.db.close()


def build_reranker(settings: Settings, model: str | None = None) -> Reranker | None:
    if not settings.rerank_enabled and model is None:
        return None
    return CrossEncoderReranker(
        model or settings.rerank_model, cache_dir=str(settings.data_dir / "models")
    )


async def build_services(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
    migrate: bool = True,
    persist: bool = True,
) -> Services:
    db = Database(settings.database_url, settings.db_pool_min, settings.db_pool_max)
    await db.open()
    if migrate:
        await apply_schema(db)

    embedder = embedder or build_embedder(settings)
    if reranker is None and settings.rerank_enabled:
        reranker = build_reranker(settings)
    llm = llm or build_llm(settings)
    rewrite_llm = (
        build_llm(settings, model=settings.rewrite_model) if settings.rewrite_model else None
    )
    tracer = Tracer(enabled=settings.langfuse_enabled)

    retriever = HybridRetriever(db, embedder, reranker, RetrievalConfig.from_settings(settings))
    qa = QAService(
        db=db,
        retriever=retriever,
        llm=llm,
        rewrite_llm=rewrite_llm,
        config=PipelineConfig.from_settings(settings),
        prices=PriceBook.load(settings.prices_file),
        tracer=tracer,
        persist=persist,
    )
    return Services(settings, db, embedder, reranker, retriever, llm, qa, tracer)
