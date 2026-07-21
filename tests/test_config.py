from pathlib import Path

import pytest
from pydantic import ValidationError

from modules.config import Settings


def test_settings_read_rag_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MODEL", "test-model")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "900")

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.llm_model == "test-model"
    assert settings.chunk_size == 900


def test_temperature_is_deterministic_by_default_and_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert settings.temperature == 0

    monkeypatch.setenv("RAG_TEMPERATURE", "0.25")
    overridden = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert overridden.temperature == 0.25


def test_settings_reject_unknown_environment_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_UNKNOWN_SETTING=value\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_CHUNK_SIZE", "0"),
        ("RAG_CHUNK_OVERLAP", "-1"),
        ("RAG_MAX_RETRIES", "2"),
    ],
)
def test_settings_reject_invalid_bounds(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_settings_reject_overlap_not_smaller_than_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_settings_reject_invalid_ollama_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OLLAMA_BASE_URL", "not-a-url")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
