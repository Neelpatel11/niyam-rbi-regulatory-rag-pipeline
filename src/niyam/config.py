from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "gemini", "extractive"]
EmbeddingProvider = Literal["fastembed", "openai", "gemini", "hashing"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NIYAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    database_url: str = "postgresql://niyam:niyam@localhost:5432/niyam"
    db_pool_min: int = 1
    db_pool_max: int = 10

    data_dir: Path = Path("data")

    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    chunk_set: str = "structure-v1"

    dense_k: int = 50
    lexical_k: int = 50
    rrf_k: int = 60
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    context_k: int = 8
    max_context_tokens: int = 6000

    refusal_threshold: float = 0.15
    refusal_threshold_dense: float = 0.55
    query_rewrite: bool = True

    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high"] = "medium"
    rewrite_model: str | None = None
    judge_model: str | None = None
    llm_max_output_tokens: int = 4000
    llm_timeout_s: float = 60.0

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    api_keys: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_per_minute: int = 30

    langfuse_enabled: bool = False
    prices_file: Path | None = None

    fetch_user_agent: str = "NiyamBot/0.1 (+https://github.com/Neelpatel11/niyam-rbi-regulatory-rag-pipeline)"
    fetch_min_delay_s: float = 2.0
    fetch_timeout_s: float = 60.0
    fetch_cache_max_age_s: float = 72000.0

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
