from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest

from niyam.api.app import create_app
from niyam.config import Settings
from niyam.db import Database
from niyam.embeddings import HashingEmbedder
from niyam.evals.dataset import load_golden
from niyam.evals.runner import EvalConfig, EvalRunner
from niyam.generation.llm import ExtractiveClient
from niyam.ingest.pipeline import embed_missing, ingest_local
from niyam.qa import AskRequest
from niyam.retrieval import TimeFilter
from niyam.retrieval.rerank import LexicalOverlapReranker
from niyam.services import Services, build_services
from tests.conftest import CORPUS, ROOT

pytestmark = pytest.mark.db

CHUNK_SETS = ["structure-v1", "fixed-512"]


@pytest.fixture(scope="module")
async def indexed(db: Database) -> Database:
    report = await ingest_local(db, [CORPUS], chunk_sets=CHUNK_SETS)
    assert not report.failed
    embedder = HashingEmbedder(256)
    for name in CHUNK_SETS:
        await embed_missing(db, embedder, name)
    return db


@pytest.fixture(scope="module")
async def services(settings: Settings, indexed: Database) -> AsyncIterator[Services]:
    svc = await build_services(
        settings,
        embedder=HashingEmbedder(256),
        reranker=LexicalOverlapReranker(),
        llm=ExtractiveClient(),
        migrate=False,
    )
    yield svc
    await svc.close()


async def test_reingest_is_idempotent(indexed: Database) -> None:
    report = await ingest_local(indexed, [CORPUS], chunk_sets=CHUNK_SETS)
    assert report.changed == 0 and report.unchanged == report.seen
    assert report.chunks_inserted == 0


async def test_supersession_detected_from_repeal_clause(indexed: Database) -> None:
    async with indexed.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT o.source_key AS old, n.source_key AS new FROM supersessions s
                JOIN documents o ON o.id = s.old_document_id JOIN documents n ON n.id = s.new_document_id
                """
            )
        ).fetchall()
    assert {(r["old"], r["new"]) for r in rows} == {("fx-kyc-2016", "fx-kyc-2025")}


async def test_withdrawn_documents_hidden_by_default(services: Services) -> None:
    result = await services.retriever.retrieve("small account balance limit")
    assert result.chunks
    assert all(c.source_key != "fx-kyc-2016" for c in result.candidates)


async def test_as_of_returns_rules_in_force_then(services: Services) -> None:
    result = await services.retriever.retrieve(
        "small account balance limit", time_filter=TimeFilter(as_of=date(2020, 6, 30))
    )
    keys = {c.source_key for c in result.candidates}
    assert "fx-kyc-2016" in keys
    assert "fx-kyc-2025" not in keys, "a direction issued after as_of must not be returned"


async def test_lexical_finds_reference_numbers(services: Services) -> None:
    result = await services.retriever.retrieve(
        "FIX.AML.BC.No.81/14.01.001/2015-16 repealed",
        config=services.retriever.config.with_overrides(use_dense=False, use_rerank=False),
    )
    assert result.chunks[0].source_key == "fx-kyc-2025"


async def test_qa_answers_with_valid_citations(services: Services) -> None:
    response = await services.qa.ask(
        AskRequest(question="What is the minimum tenor of a domestic term deposit?")
    )
    assert response.status.value == "answered"
    assert response.citations
    assert response.citations[0].url.startswith(
        "https://example.invalid/fixtures/fx-deposits-2024#"
    )
    assert "[S" in response.answer
    assert response.cost_usd == 0.0


async def test_qa_refuses_out_of_scope(services: Services) -> None:
    response = await services.qa.ask(
        AskRequest(question="What is the penalty for late GST filing?")
    )
    assert response.status.value == "not_found"
    assert response.answer == "Not found in the documents."
    assert response.citations == []


async def test_api_ask_feedback_and_documents(settings: Settings, services: Services) -> None:
    app = create_app(settings, services=services)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        ask = await client.post(
            "/v1/ask", json={"question": "How is interest on savings deposits calculated?"}
        )
        assert ask.status_code == 200, ask.text
        body = ask.json()
        assert body["status"] == "answered"
        assert body["sources"]

        fb = await client.post("/v1/feedback", json={"trace_id": body["trace_id"], "rating": 1})
        assert fb.status_code == 204
        missing = await client.post(
            "/v1/feedback", json={"trace_id": "00000000-0000-0000-0000-000000000000", "rating": -1}
        )
        assert missing.status_code == 404

        docs = await client.get("/v1/documents")
        withdrawn = next(d for d in docs.json() if d["source_key"] == "fx-kyc-2016")
        assert withdrawn["withdrawn_on"] == "2025-12-04"
        assert withdrawn["superseded_by"] == [
            "Fixture Direction (Commercial Banks - Customer Due Diligence), 2025"
        ]

        bad = await client.post("/v1/ask", json={"question": "  "})
        assert bad.status_code == 422


async def test_api_stream_emits_events(settings: Settings, services: Services) -> None:
    app = create_app(settings, services=services)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        response = await client.post(
            "/v1/ask/stream", json={"question": "Where must digital lending data be stored?"}
        )
        events = [
            line.removeprefix("event: ").strip()
            for line in response.text.splitlines()
            if line.startswith("event:")
        ]
    assert events[0] == "status"
    assert "sources" in events and events[-1] == "answer"


async def test_retrieval_eval_runs_on_smoke_set(settings: Settings, indexed: Database) -> None:
    items = load_golden(ROOT / "evals" / "smoke.jsonl")
    runner = EvalRunner(settings, indexed)
    runner.register(embedder=HashingEmbedder(256), reranker=LexicalOverlapReranker())
    cfg = EvalConfig(
        name="hybrid-overlap",
        embedding_provider="hashing",
        embedding_model="256",
        rerank_model="overlap",
        retrieval={"use_rerank": True},
    )
    run = await runner.run(cfg, items, mode="retrieval")
    metrics = run["metrics"]
    assert metrics["n"] == len(items)
    assert metrics["recall@20"] >= metrics["recall@5"] >= 0.5
    temporal = run["by_category"]["temporal"]
    assert temporal["recall@5"] == 1.0, "dated questions must retrieve the withdrawn direction"
