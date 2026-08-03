from pathlib import Path

from modules.config import Settings
from modules.error_handler import analyze_error


def test_missing_chat_model_uses_configured_model_name() -> None:
    settings = Settings(llm_model="qwen3.5:9b")

    analysis = analyze_error(Exception("model qwen3.5:9b not found"), settings)

    assert analysis["message"] == "Required chat model not found"
    assert "Pull the chat model: 'ollama pull qwen3.5:9b'" in analysis["suggestions"]


def test_missing_embedding_model_uses_configured_model_name() -> None:
    settings = Settings(embedding_model="nomic-embed-text")

    analysis = analyze_error(Exception("model nomic-embed-text not found"), settings)

    assert analysis["message"] == "Required embedding model not found"
    assert "Pull the embedding model: 'ollama pull nomic-embed-text'" in analysis["suggestions"]


def test_vector_database_recovery_uses_configured_chroma_directory() -> None:
    settings = Settings(chroma_dir=Path("data/chroma"))

    analysis = analyze_error(Exception("Chroma vector database is corrupt"), settings)

    assert "Delete and recreate the 'data/chroma' directory" in analysis["suggestions"]


def test_recovery_guidance_supports_custom_models_and_chroma_path() -> None:
    settings = Settings(
        llm_model="custom-chat:latest",
        embedding_model="custom-embed:v2",
        chroma_dir=Path("runtime/custom-index"),
    )

    chat = analyze_error(Exception("model custom-chat:latest not found"), settings)
    embedding = analyze_error(Exception("model custom-embed:v2 not found"), settings)
    vector = analyze_error(Exception("vector store unavailable"), settings)

    assert "ollama pull custom-chat:latest" in " ".join(chat["suggestions"])
    assert "ollama pull custom-embed:v2" in " ".join(embedding["suggestions"])
    assert "runtime/custom-index" in " ".join(vector["suggestions"])
