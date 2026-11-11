from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


@dataclass(slots=True)
class FetchResult:
    url: str
    content: bytes
    content_type: str
    from_cache: bool

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class PoliteFetcher:
    """HTTP client with per-host spacing, retries and a content-addressed disk cache."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        min_delay_s: float = 2.0,
        timeout_s: float = 60.0,
        max_cache_age_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_delay_s = min_delay_s
        self._max_cache_age_s = max_cache_age_s
        self._last_request: dict[str, float] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept-Language": "en"},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> PoliteFetcher:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        base = self._cache_dir / key[:2] / key
        return base.with_suffix(".bin"), base.with_suffix(".json")

    def _read_cache(self, url: str) -> FetchResult | None:
        body_path, meta_path = self._cache_paths(url)
        if not body_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self._max_cache_age_s is not None:
            age = time.time() - float(meta.get("fetched_at", 0))
            if age > self._max_cache_age_s:
                return None
        return FetchResult(
            url=url,
            content=body_path.read_bytes(),
            content_type=meta.get("content_type", ""),
            from_cache=True,
        )

    def _write_cache(self, result: FetchResult) -> None:
        body_path, meta_path = self._cache_paths(result.url)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = body_path.with_suffix(".tmp")
        tmp.write_bytes(result.content)
        tmp.replace(body_path)
        meta_path.write_text(
            json.dumps(
                {
                    "url": result.url,
                    "content_type": result.content_type,
                    "sha256": result.sha256,
                    "fetched_at": time.time(),
                }
            ),
            encoding="utf-8",
        )

    async def _wait_turn(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            delay = self._min_delay_s - (time.monotonic() - last)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last_request[host] = time.monotonic()

    async def get(self, url: str, *, use_cache: bool = True) -> FetchResult:
        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        host = urlsplit(url).netloc
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(4),
                    wait=wait_exponential_jitter(initial=2, max=30),
                    retry=retry_if_exception(_is_retryable),
                    reraise=True,
                ):
                    with attempt:
                        await self._wait_turn(host)
                        response = await self._client.get(url)
                        response.raise_for_status()
            except httpx.HTTPError as exc:
                raise FetchError(f"failed to fetch {url}: {exc}") from exc

        result = FetchResult(
            url=url,
            content=response.content,
            content_type=response.headers.get("content-type", ""),
            from_cache=False,
        )
        self._write_cache(result)
        log.info("fetched %s (%d bytes)", url, len(result.content))
        return result
