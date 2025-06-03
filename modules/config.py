import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class RAGConfig:
    # Model settings
    llm_model: str = "llama3.1"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.7
    
    # Vector DB settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    k_retrieval: int = 3
    
    # Paths
    root_dir: Path = Path(__file__).parent.parent
    sources_dir: str = str(root_dir / "sources")
    vector_db_dir: str = str(root_dir / "chroma_db")
    logs_dir: str = str(root_dir / "logs")

    # App settings
    app_title: str = "Complete RAG Assistant"
    app_description: str = "AI assistant with advanced RAG capabilities"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        return cls(
            llm_model=os.getenv('LLM_MODEL', cls.llm_model),
            embedding_model=os.getenv('EMBEDDING_MODEL', cls.embedding_model),
            temperature=float(os.getenv('TEMPERATURE', cls.temperature)),
            chunk_size=int(os.getenv('CHUNK_SIZE', cls.chunk_size)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', cls.chunk_overlap)),
            k_retrieval=int(os.getenv('K_RETRIEVAL', cls.k_retrieval)),
            sources_dir=os.getenv('SOURCES_DIR', cls.sources_dir),
            vector_db_dir=os.getenv('VECTOR_DB_DIR', cls.vector_db_dir),
            ollama_base_url=os.getenv('OLLAMA_BASE_URL', cls.ollama_base_url)
        )

config = RAGConfig.from_env()