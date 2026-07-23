import logging
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypedDict, TypeVar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


class ErrorAnalysis(TypedDict):
    message: str
    type: str
    suggestions: list[str]


def _connection_error() -> ErrorAnalysis:
    return {
        "message": "Cannot connect to Ollama server",
        "type": "CONNECTION_ERROR",
        "suggestions": [
            "Make sure Ollama is running: 'ollama serve'",
            "Check if the correct port (11434) is being used",
            "Verify Ollama is installed: https://ollama.com",
        ],
    }


def _model_error() -> ErrorAnalysis:
    return {
        "message": "Required model not found",
        "type": "MODEL_ERROR",
        "suggestions": [
            "Pull the required model: 'ollama pull llama3.1'",
            "Pull embedding model: 'ollama pull nomic-embed-text'",
            "Check available models: 'ollama list'",
        ],
    }


def _file_error() -> ErrorAnalysis:
    return {
        "message": "Source documents not found",
        "type": "FILE_ERROR",
        "suggestions": [
            "Create a 'sources' directory",
            "Add text or PDF files to the sources directory",
            "Check file permissions and paths",
        ],
    }


def _vector_error() -> ErrorAnalysis:
    return {
        "message": "Vector database error",
        "type": "VECTOR_DB_ERROR",
        "suggestions": [
            "Delete and recreate the chroma_db directory",
            "Check if ChromaDB is properly installed",
            "Ensure embeddings are generated correctly",
        ],
    }


def _unknown_error(error: Exception) -> ErrorAnalysis:
    return {
        "message": f"Unexpected error: {error}",
        "type": "UNKNOWN_ERROR",
        "suggestions": [
            "Check the logs for more details",
            "Verify all dependencies are installed",
            "Try restarting the application",
        ],
    }


def _is_model_error(message: str) -> bool:
    return "model" in message and ("not found" in message or "pull" in message)


class RAGError(Exception):
    """Custom RAG application error"""

    def __init__(
        self, message: str, error_type: str = "RAG_ERROR", suggestions: list[str] | None = None
    ):
        self.message = message
        self.error_type = error_type
        self.suggestions = suggestions or []
        super().__init__(self.message)


def handle_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for handling common errors with helpful messages"""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = analyze_error(e)
            logger.error("Error in %s: %s", func.__name__, error_info["message"])
            raise RAGError(
                message=error_info["message"],
                error_type=error_info["type"],
                suggestions=error_info["suggestions"],
            )

    return wrapper


def analyze_error(error: Exception) -> ErrorAnalysis:
    """Analyze error and provide helpful suggestions"""
    message = str(error).lower()
    classifiers: tuple[tuple[Callable[[str], bool], Callable[[], ErrorAnalysis]], ...] = (
        (lambda text: "connection" in text and "ollama" in text, _connection_error),
        (_is_model_error, _model_error),
        (lambda text: "no such file" in text or "sources" in text, _file_error),
        (lambda text: "chroma" in text or "vector" in text, _vector_error),
    )
    for matches, analysis in classifiers:
        if matches(message):
            return analysis()
    return _unknown_error(error)
