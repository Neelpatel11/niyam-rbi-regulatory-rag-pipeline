import time
from pathlib import Path

import httpx
import pytest

from niyam.ingest.fetcher import FetchError, PoliteFetcher


def _transport(statuses: list[int], calls: list[float]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        status = statuses.pop(0) if statuses else 200
        return httpx.Response(
            status, content=b"<html>ok</html>", headers={"content-type": "text/html"}
        )

    return httpx.MockTransport(handler)


async def test_caches_responses(tmp_path: Path) -> None:
    calls: list[float] = []
    async with PoliteFetcher(tmp_path, "test", min_delay_s=0, transport=_transport([], calls)) as f:
        first = await f.get("https://example.invalid/a")
        second = await f.get("https://example.invalid/a")
    assert not first.from_cache and second.from_cache
    assert second.content == b"<html>ok</html>"
    assert len(calls) == 1


async def test_retries_transient_errors(tmp_path: Path) -> None:
    calls: list[float] = []
    async with PoliteFetcher(
        tmp_path, "test", min_delay_s=0, transport=_transport([503, 200], calls)
    ) as f:
        result = await f.get("https://example.invalid/b")
    assert result.content == b"<html>ok</html>"
    assert len(calls) == 2


async def test_does_not_retry_client_errors(tmp_path: Path) -> None:
    calls: list[float] = []
    async with PoliteFetcher(
        tmp_path, "test", min_delay_s=0, transport=_transport([404], calls)
    ) as f:
        with pytest.raises(FetchError):
            await f.get("https://example.invalid/c")
    assert len(calls) == 1


async def test_spaces_requests_to_the_same_host(tmp_path: Path) -> None:
    calls: list[float] = []
    async with PoliteFetcher(
        tmp_path, "test", min_delay_s=0.2, transport=_transport([], calls)
    ) as f:
        await f.get("https://example.invalid/d")
        await f.get("https://example.invalid/e")
    assert calls[1] - calls[0] >= 0.19
