from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

Conn = AsyncConnection[DictRow]


async def _configure(conn: Conn) -> None:
    await register_vector_async(conn)
    try:
        await conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except Exception:
        log.debug("pgvector < 0.8: iterative index scans unavailable")


class Database:
    def __init__(self, url: str, min_size: int = 1, max_size: int = 10) -> None:
        self._url = url
        self._pool: AsyncConnectionPool[Conn] = AsyncConnectionPool(
            url,
            connection_class=AsyncConnection[DictRow],
            min_size=min_size,
            max_size=max_size,
            open=False,
            configure=_configure,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )

    async def open(self) -> None:
        await self._bootstrap_extension()
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    async def _bootstrap_extension(self) -> None:
        async with await AsyncConnection.connect(self._url, autocommit=True) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Conn]:
        async with self._pool.connection() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Conn]:
        async with self._pool.connection() as conn, conn.transaction():
            yield conn

    async def ping(self) -> bool:
        try:
            async with self.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            log.exception("database ping failed")
            return False


def load_schema_sql() -> str:
    return resources.files("niyam.db").joinpath("schema.sql").read_text(encoding="utf-8")


async def apply_schema(db: Database) -> None:
    async with db.transaction() as conn:
        await conn.execute(load_schema_sql())


async def ensure_vector_index(conn: Conn, model_id: int, chunk_set_id: int, dim: int) -> None:
    """One partial HNSW index per (embedding model, chunk set), so dimensions can differ per model."""
    index_name = sql.Identifier(f"embeddings_hnsw_m{model_id}_s{chunk_set_id}")
    statement = sql.SQL(
        "CREATE INDEX IF NOT EXISTS {index} ON embeddings "
        "USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64) "
        "WHERE model_id = {model_id} AND chunk_set_id = {chunk_set_id}"
    ).format(
        index=index_name,
        dim=sql.Literal(dim),
        model_id=sql.Literal(model_id),
        chunk_set_id=sql.Literal(chunk_set_id),
    )
    await conn.execute(statement)
