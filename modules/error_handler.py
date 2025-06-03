import logging
import traceback
from typing import Any, Dict, Optional
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Custom RAG application error"""
    def __init__(self, message: str, error_type: str = "RAG_ERROR", suggestions: Optional[list] = None):
        self.message = message
        self.error_type = error_type
        self.suggestions = suggestions or []
        super().__init__(self.message)

def handle_errors(func):
    """Decorator for handling common errors with helpful messages"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = analyze_error(e)
            logger.error(f"Error in {func.__name__}: {error_info['message']}")
            raise RAGError(
                message=error_info['message'],
                error_type=error_info['type'],
                suggestions=error_info['suggestions']
            )
    return wrapper

def analyze_error(error: Exception) -> Dict[str, Any]:
    """Analyze error and provide helpful suggestions"""
    error_str = str(error).lower()
    
    if "connection" in error_str and "ollama" in error_str:
        return {
            'message': "Cannot connect to Ollama server",
            'type': 'CONNECTION_ERROR',
            'suggestions': [
                "Make sure Ollama is running: 'ollama serve'",
                "Check if the correct port (11434) is being used",
                "Verify Ollama is installed: https://ollama.com"
            ]
        }
    
    elif "model" in error_str and ("not found" in error_str or "pull" in error_str):
        return {
            'message': "Required model not found",
            'type': 'MODEL_ERROR',
            'suggestions': [
                "Pull the required model: 'ollama pull llama3.1'",
                "Pull embedding model: 'ollama pull nomic-embed-text'",
                "Check available models: 'ollama list'"
            ]
        }
    
    elif "no such file" in error_str or "sources" in error_str:
        return {
            'message': "Source documents not found",
            'type': 'FILE_ERROR',
            'suggestions': [
                "Create a 'sources' directory",
                "Add text or PDF files to the sources directory",
                "Check file permissions and paths"
            ]
        }
    
    elif "chroma" in error_str or "vector" in error_str:
        return {
            'message': "Vector database error",
            'type': 'VECTOR_DB_ERROR',
            'suggestions': [
                "Delete and recreate the chroma_db directory",
                "Check if ChromaDB is properly installed",
                "Ensure embeddings are generated correctly"
            ]
        }
    
    else:
        return {
            'message': f"Unexpected error: {str(error)}",
            'type': 'UNKNOWN_ERROR',
            'suggestions': [
                "Check the logs for more details",
                "Verify all dependencies are installed",
                "Try restarting the application"
            ]
        }