from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from niyam.evals.dataset import Evidence

_TOKEN_RE = re.compile(r"[^\W_]+")

COVERAGE_THRESHOLD = 0.6


def normalize_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with accents removed; every other character separates."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _TOKEN_RE.findall(stripped.lower())


@dataclass(slots=True, frozen=True)
class ChunkView:
    chunk_id: str
    source_key: str
    tokens: tuple[str, ...]


def coverage(quote_tokens: Sequence[str], chunk_tokens: Sequence[str]) -> float:
    """Share of the quote found as one contiguous run inside the chunk.

    A quote that straddles a chunk boundary gets partial credit from each side, which is why the
    threshold is below 1.0.
    """
    if not quote_tokens:
        return 0.0
    matcher = SequenceMatcher(None, quote_tokens, chunk_tokens, autojunk=False)
    match = matcher.find_longest_match(0, len(quote_tokens), 0, len(chunk_tokens))
    return match.size / len(quote_tokens)


def evidence_hits(
    evidence: list[Evidence], ranked: list[ChunkView], threshold: float = COVERAGE_THRESHOLD
) -> list[list[bool]]:
    """Matrix [evidence][rank] of whether the chunk at that rank supports the evidence quote."""
    out: list[list[bool]] = []
    for item in evidence:
        quote = normalize_tokens(item.quote)
        out.append(
            [
                chunk.source_key == item.source_key and coverage(quote, chunk.tokens) >= threshold
                for chunk in ranked
            ]
        )
    return out


def recall_at_k(hits: list[list[bool]], k: int) -> float:
    if not hits:
        return 0.0
    return sum(any(row[:k]) for row in hits) / len(hits)


def hit_at_k(hits: list[list[bool]], k: int) -> float:
    return 1.0 if any(any(row[:k]) for row in hits) else 0.0


def reciprocal_rank(hits: list[list[bool]]) -> float:
    if not hits:
        return 0.0
    width = len(hits[0])
    for rank in range(width):
        if any(row[rank] for row in hits):
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(hits: list[list[bool]], k: int) -> float:
    """Binary-relevance nDCG; a chunk is relevant if it supports any evidence quote."""
    if not hits:
        return 0.0
    width = min(k, len(hits[0]))
    relevant = [any(row[rank] for row in hits) for rank in range(width)]
    dcg = sum(1.0 / math.log2(rank + 2) for rank, rel in enumerate(relevant) if rel)
    ideal_count = min(len(hits), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return min(1.0, dcg / idcg) if idcg else 0.0


@dataclass(slots=True)
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be non-empty and the same length")
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def best_refusal_threshold(
    scores: Sequence[float | None], answerable: Sequence[bool]
) -> tuple[float, BinaryCounts]:
    """Pick the score cut-off that maximises refusal F1 (positive class: should refuse)."""
    candidates = sorted({s for s in scores if s is not None} | {0.0, 1.0})
    best: tuple[float, BinaryCounts] | None = None
    for threshold in candidates:
        counts = BinaryCounts()
        for score, is_answerable in zip(scores, answerable, strict=True):
            refuse = score is None or score < threshold
            counts.add(refuse, not is_answerable)
        if best is None or counts.f1 > best[1].f1:
            best = (threshold, counts)
    assert best is not None
    return best
