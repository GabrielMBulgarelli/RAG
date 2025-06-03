import os
import glob
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import config
from error_handler import handle_errors, logger

class VectorDBManager:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url
        )
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    @handle_errors
    def load_documents(self, sources_dir: str = None) -> List[Document]:
        """Load documents from various file formats"""
        sources_dir = sources_dir or config.sources_dir
        documents = []
        
        if not os.path.exists(sources_dir):
            raise FileNotFoundError(f"Sources directory '{sources_dir}' not found")
        
        # Load text files
        txt_files = glob.glob(os.path.join(sources_dir, "*.txt"))
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_type': 'text',
                        'filename': os.path.basename(file_path)
                    })
                documents.extend(docs)
                logger.info(f"Loaded text file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Load PDF files
        pdf_files = glob.glob(os.path.join(sources_dir, "*.pdf"))
        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_type': 'pdf',
                        'filename': os.path.basename(file_path)
                    })
                documents.extend(docs)
                logger.info(f"Loaded PDF file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    @handle_errors
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create or update vector database"""
        if not documents:
            raise ValueError("No documents provided for vector database creation")
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} text chunks")
        
        # Create vector database
        if os.path.exists(config.vector_db_dir):
            # Load existing database and add new documents
            self.vectorstore = Chroma(
                persist_directory=config.vector_db_dir,
                embedding_function=self.embeddings
            )
            # Add new documents (this will append to existing)
            self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            logger.info("Updated existing vector database")
        else:
            # Create new database
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=config.vector_db_dir
            )
            self.vectorstore.persist()
            logger.info("Created new vector database")
        
        return self.vectorstore
    
    @handle_errors
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector database"""
        if os.path.exists(config.vector_db_dir):
            self.vectorstore = Chroma(
                persist_directory=config.vector_db_dir,
                embedding_function=self.embeddings
            )
            logger.info("Loaded existing vector database")
            return self.vectorstore
        return None
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get retriever for the vector database"""
        if not self.vectorstore:
            raise ValueError("Vector database not initialized")
        
        search_kwargs = search_kwargs or {"k": config.k_retrieval}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    @handle_errors
    def setup(self, force_rebuild: bool = False) -> Chroma:
        """Complete setup of vector database"""
        if not force_rebuild and self.load_existing_vectorstore():
            return self.vectorstore
        
        logger.info("Setting up vector database...")
        documents = self.load_documents()
        return self.create_vectorstore(documents)