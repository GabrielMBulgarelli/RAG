"""Validated application settings."""

from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strict settings loaded from ``RAG_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
        strict=True,
        validate_default=True,
    )

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    sources_dir: Path = PROJECT_ROOT / "sources"
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.json"
    trace_dir: Path = PROJECT_ROOT / "data" / "traces"
    logs_dir: Path = PROJECT_ROOT / "logs"
    benchmark_results_dir: Path = PROJECT_ROOT / "evals" / "results" / "full_rag"

    chunk_size: int = Field(default=700, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    semantic_candidates: int = Field(default=10, gt=0)
    sparse_candidates: int = Field(default=10, gt=0)
    max_candidates: int = Field(default=20, gt=0)
    max_context_chunks: int = Field(default=6, gt=0)
    max_subqueries: int = Field(default=4, ge=1, le=4)
    max_retries: int = Field(default=1, ge=0, le=1)

    gradio_host: str = "127.0.0.1"
    gradio_port: int = Field(default=7860, ge=1, le=65535)
    gradio_share: bool = False
    log_level: str = "INFO"

    app_title: str = "Complete RAG Assistant"
    app_description: str = "AI assistant with advanced RAG capabilities"
    k_retrieval: int = Field(default=3, gt=0)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ollama_base_url must be an HTTP(S) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def vector_db_dir(self) -> str:
        """Compatibility path for the existing Chroma integration."""
        return str(self.chroma_dir)


config = Settings()
