from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.;:?!])\s+(?=[(\"'‘“A-Z0-9])")
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer:
    """Token counter backed by tiktoken, with a deterministic fallback when it is unavailable."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding)
        except Exception:
            log.warning("tiktoken unavailable; falling back to a regex token estimate")

    def encode(self, text: str) -> list[int]:
        if self._encoding is not None:
            return self._encoding.encode(text, disallowed_special=())
        return [hash(tok) & 0xFFFFFFFF for tok in _WORD_RE.findall(text)]

    def decode(self, tokens: list[int]) -> str:
        if self._encoding is None:
            raise RuntimeError("decode requires tiktoken")
        return self._encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def exact(self) -> bool:
        return self._encoding is not None


@lru_cache
def get_tokenizer() -> Tokenizer:
    return Tokenizer()


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text)]
    return [p for p in parts if p]
