from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True, frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


# USD per million tokens. Anthropic list prices as published on 2026-11-03.
# Other providers are intentionally absent: add verified prices through NIYAM_PRICES_FILE
# rather than shipping numbers that may be stale.
PRICES_AS_OF = "2026-11-03"
DEFAULT_PRICES: dict[str, Price] = {
    "anthropic:claude-opus-5-5": Price(
        4.00, 20.00, cache_read_per_mtok=0.20, cache_write_per_mtok=5.00
    ),
    "anthropic:claude-sonnet-5-5": Price(
        2.00, 10.00, cache_read_per_mtok=0.10, cache_write_per_mtok=2.50
    ),
    "anthropic:claude-haiku-5-5": Price(0.10, 0.50),
    "extractive:extractive": Price(0.0, 0.0),
}


@dataclass(slots=True)
class Usage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.calls += other.calls


class PriceBook:
    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        self._prices = dict(DEFAULT_PRICES if prices is None else prices)

    @classmethod
    def load(cls, path: Path | None) -> PriceBook:
        book = cls()
        if path is not None and path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for key, value in data.items():
                book._prices[key] = Price(**value)
        return book

    def known(self, key: str) -> bool:
        return key in self._prices

    def cost(self, usage: Usage) -> float | None:
        price = self._prices.get(usage.key)
        if price is None:
            return None
        cache_read = (
            price.cache_read_per_mtok
            if price.cache_read_per_mtok is not None
            else price.input_per_mtok
        )
        cache_write = (
            price.cache_write_per_mtok
            if price.cache_write_per_mtok is not None
            else price.input_per_mtok
        )
        total = (
            usage.input_tokens * price.input_per_mtok
            + usage.output_tokens * price.output_per_mtok
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
        )
        return total / 1_000_000
