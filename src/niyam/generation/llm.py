from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from niyam.config import Settings
from niyam.generation.pricing import Usage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Effort = Literal["low", "medium", "high"]

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MODELS_WITH_SERVER_FALLBACKS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMResult(Generic[T]):
    parsed: T | None
    usage: Usage
    latency_ms: float
    stop_reason: str | None
    refusal: bool = False


class LLMClient(Protocol):
    provider: str
    model: str

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: Effort = "medium",
        max_tokens: int = 4000,
    ) -> LLMResult[T]: ...


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        use_fallbacks: bool = True,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model
        self._use_fallbacks = use_fallbacks and model in MODELS_WITH_SERVER_FALLBACKS

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: Effort = "medium",
        max_tokens: int = 4000,
    ) -> LLMResult[T]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": schema,
            "output_config": {"effort": effort},
        }
        start = time.perf_counter()
        response: Any
        try:
            if self._use_fallbacks:
                response = await self._client.beta.messages.parse(
                    **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = await self._client.messages.parse(**kwargs)
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"anthropic request failed ({exc.status_code}): {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000

        usage = response.usage
        accounted = Usage(
            provider=self.provider,
            model=getattr(response, "model", None) or self.model,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            calls=1,
        )
        if response.stop_reason == "refusal":
            log.warning("model declined request %s", getattr(response, "_request_id", ""))
            return LLMResult(None, accounted, latency, response.stop_reason, refusal=True)
        if response.stop_reason == "max_tokens":
            return LLMResult(None, accounted, latency, response.stop_reason)
        return LLMResult(response.parsed_output, accounted, latency, response.stop_reason)


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str, api_key: str | None = None, timeout_s: float = 60.0) -> None:
        import openai

        self._openai = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: Effort = "medium",
        max_tokens: int = 4000,
    ) -> LLMResult[T]:
        start = time.perf_counter()
        try:
            response = await self._client.responses.parse(
                model=self.model,
                instructions=system,
                input=user,
                text_format=schema,
                max_output_tokens=max_tokens,
                reasoning={"effort": effort},
            )
        except self._openai.APIStatusError as exc:
            raise LLMError(f"openai request failed ({exc.status_code}): {exc.message}") from exc
        except self._openai.APIConnectionError as exc:
            raise LLMError(f"openai connection error: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        usage = response.usage
        cached = 0
        if usage is not None and usage.input_tokens_details is not None:
            cached = usage.input_tokens_details.cached_tokens or 0
        accounted = Usage(
            provider=self.provider,
            model=self.model,
            input_tokens=(usage.input_tokens - cached) if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cache_read_tokens=cached,
            calls=1,
        )
        return LLMResult(response.output_parsed, accounted, latency, response.status)


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: Effort = "medium",
        max_tokens: int = 4000,
    ) -> LLMResult[T]:
        from google.genai import errors, types

        start = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=max_tokens,
                ),
            )
        except errors.APIError as exc:
            raise LLMError(f"gemini request failed ({exc.code}): {exc.message}") from exc
        latency = (time.perf_counter() - start) * 1000
        meta = response.usage_metadata
        accounted = Usage(
            provider=self.provider,
            model=self.model,
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
            if meta
            else 0,
            cache_read_tokens=(meta.cached_content_token_count or 0) if meta else 0,
            calls=1,
        )
        parsed = response.parsed
        if parsed is None and response.text:
            try:
                parsed = schema.model_validate_json(response.text)
            except ValidationError:
                parsed = None
        return LLMResult(parsed if isinstance(parsed, schema) else None, accounted, latency, None)


_PASSAGE_RE = re.compile(r"<passage id=\"(S\d+)\"[^>]*>\s*(.*?)\s*</passage>", re.S)
_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.S)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOP = {
    "the",
    "and",
    "for",
    "what",
    "which",
    "are",
    "under",
    "with",
    "that",
    "this",
    "does",
    "how",
}


class ExtractiveClient:
    """Offline stand-in used in tests and CI: answers with the best-overlapping passage sentence.

    It only understands the answer and rewrite schemas defined in this package.
    """

    provider = "extractive"
    model = "extractive"

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: Effort = "medium",
        max_tokens: int = 4000,
    ) -> LLMResult[T]:
        fields = set(schema.model_fields)
        usage = Usage(self.provider, self.model, input_tokens=len(user) // 4, calls=1)
        if {"standalone_question", "as_of"} <= fields:
            found = _QUESTION_RE.search(user)
            payload: dict[str, Any] = {
                "standalone_question": found.group(1) if found else user,
                "as_of": None,
            }
            return LLMResult(schema.model_validate(payload), usage, 0.0, "end_turn")

        if {"answerable", "answer", "citations"} <= fields:
            question_match = _QUESTION_RE.search(user)
            question = question_match.group(1) if question_match else ""
            terms = {w for w in _WORD_RE.findall(question.lower()) if w not in _STOP}
            best: tuple[float, str, str] | None = None
            for label, passage in _PASSAGE_RE.findall(user):
                for sentence in re.split(r"(?<=[.;])\s+", passage):
                    words = set(_WORD_RE.findall(sentence.lower()))
                    score = len(terms & words) / len(terms) if terms else 0.0
                    if best is None or score > best[0]:
                        best = (score, label, sentence.strip())
            if best is None or best[0] < 0.5:
                payload = {"answerable": False, "answer": "", "citations": []}
            else:
                payload = {
                    "answerable": True,
                    "answer": f"{best[2]} [{best[1]}]",
                    "citations": [best[1]],
                }
            usage.output_tokens = len(json.dumps(payload)) // 4
            return LLMResult(schema.model_validate(payload), usage, 0.0, "end_turn")

        raise LLMError(f"extractive client cannot produce {schema.__name__}")


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None


def build_llm(
    settings: Settings, model: str | None = None, provider: str | None = None
) -> LLMClient:
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    if provider == "anthropic":
        return AnthropicClient(
            model, api_key=_secret(settings.anthropic_api_key), timeout_s=settings.llm_timeout_s
        )
    if provider == "openai":
        return OpenAIClient(
            model, api_key=_secret(settings.openai_api_key), timeout_s=settings.llm_timeout_s
        )
    if provider == "gemini":
        return GeminiClient(model, api_key=_secret(settings.gemini_api_key))
    if provider == "extractive":
        return ExtractiveClient()
    raise ValueError(f"unknown LLM provider {provider!r}")
