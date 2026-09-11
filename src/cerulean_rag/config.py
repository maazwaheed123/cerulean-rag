"""Application settings.

All configuration comes from environment variables or a ``.env`` file in the
working directory, with the defaults below. ``AS_OF_DATE`` is deliberately a
setting rather than the system clock so that answers to "what is current"
questions are reproducible (assignment as-of date: 27 August 2026).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class Settings(BaseSettings):
    """Typed settings; every field can be overridden by an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GEN_MODEL: str = "qwen2.5:7b-instruct"
    EMBED_MODEL: str = "nomic-embed-text"

    # "Current" date for the corpus
    AS_OF_DATE: date = date(2026, 8, 27)

    # Paths
    CORPUS_DIR: Path = Path("./corpus")
    CHROMA_DIR: Path = Path("./data/chroma")
    CHUNKS_FILE: Path = Path("./data/chunks.jsonl")
    COLLECTION_NAME: str = "cerulean_docs"

    # Retrieval
    TOP_K: int = Field(default=8, ge=1)
    RETRIEVE_K_PER_SOURCE: int = Field(default=10, ge=1)
    SIM_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
    USE_BM25: bool = True

    # Generation
    NUM_CTX: int = Field(default=8192, ge=1024)
    TEMPERATURE: float = Field(default=0.0, ge=0.0)
    SEED: int = 42
    MAX_ANSWER_TOKENS: int = Field(default=700, ge=64)

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("./logs/app.log")
    QUERY_LOG: Path = Path("./logs/queries.jsonl")

    @field_validator("AS_OF_DATE", mode="before")
    @classmethod
    def _parse_as_of_date(cls, value: object) -> object:
        """Accept ISO strings such as ``2026-08-27``; reject anything unparseable."""
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip())
            except ValueError as exc:
                raise ValueError(
                    f"AS_OF_DATE must be an ISO date (YYYY-MM-DD), got {value!r}"
                ) from exc
        raise ValueError(
            f"AS_OF_DATE must be a date or ISO string, got {type(value).__name__}"
        )

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            level = value.strip().upper()
            if level not in _LOG_LEVELS:
                raise ValueError(
                    f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {value!r}"
                )
            return level
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (loaded once)."""
    return Settings()
