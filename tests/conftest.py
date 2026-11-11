from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest

from niyam.config import Settings
from niyam.db import Database, apply_schema

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
CORPUS = FIXTURES / "corpus"


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()  # psycopg async needs a selector loop
    return asyncio.DefaultEventLoopPolicy()


def _server_url() -> str | None:
    url = os.environ.get("NIYAM_TEST_DATABASE_URL")
    if url:
        return url
    try:
        import pgserver
    except ImportError:
        return None
    data_dir = Path(os.environ.get("NIYAM_TEST_PGDATA", ROOT / ".pytest_pgdata"))
    return pgserver.get_server(data_dir, cleanup_mode=None).get_uri()


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    server = _server_url()
    if server is None:
        pytest.skip("no Postgres available: set NIYAM_TEST_DATABASE_URL or install pgserver")
    name = f"niyam_test_{uuid.uuid4().hex[:8]}"
    base, _, _ = server.rpartition("/")
    with psycopg.connect(server, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    yield f"{base}/{name}"
    with psycopg.connect(server, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def settings(database_url: str, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        env="test",
        database_url=database_url,
        data_dir=tmp_path_factory.mktemp("data"),
        embedding_provider="hashing",
        embedding_model="256",
        llm_provider="extractive",
        llm_model="extractive",
        rerank_enabled=False,
        query_rewrite=False,
        refusal_threshold=0.5,
        refusal_threshold_dense=0.2,
        rate_limit_per_minute=0,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture(scope="session")
async def db(settings: Settings) -> AsyncIterator[Database]:
    database = Database(settings.database_url, 1, 4)
    await database.open()
    await apply_schema(database)
    yield database
    await database.close()
