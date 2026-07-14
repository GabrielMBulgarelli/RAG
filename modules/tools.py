from datetime import datetime
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Schema for document information"""

    filename: str = Field(description="Name of the document")
    source_type: str = Field(description="Type of source (text, pdf)")
    content_preview: str = Field(description="First 200 characters of content")
    metadata: Dict[str, Any] = Field(description="Additional metadata")


class SearchResult(BaseModel):
    """Schema for search results"""

    query: str = Field(description="The search query")
    results: List[Dict[str, Any]] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results found")


@tool
def list_available_documents(vector_db_manager) -> str:
    """List all available documents in the knowledge base"""
    if not vector_db_manager.vectorstore:
        return "No documents available. Vector database not initialized."

    try:
        # Get all documents from the vector store
        docs = vector_db_manager.vectorstore.get()

        if not docs or not docs.get("metadatas"):
            return "No documents found in the knowledge base."

        # Extract unique documents by filename
        unique_docs = {}
        for metadata in docs["metadatas"]:
            filename = metadata.get("filename", "Unknown")
            if filename not in unique_docs:
                unique_docs[filename] = {
                    "filename": filename,
                    "source_type": metadata.get("source_type", "unknown"),
                    "chunks": 1,
                }
            else:
                unique_docs[filename]["chunks"] += 1

        result = "Available documents in knowledge base:\n\n"
        for doc_info in unique_docs.values():
            result += f"📄 {doc_info['filename']} ({doc_info['source_type']}) - {doc_info['chunks']} chunks\n"

        return result

    except Exception as e:
        return f"Error listing documents: {str(e)}"


@tool
def search_documents(query: str, vector_db_manager, k: int = 3) -> str:
    """Search for specific information in the documents"""
    if not vector_db_manager.vectorstore:
        return "Vector database not available for search."

    try:
        retriever = vector_db_manager.get_retriever({"k": k})
        docs = retriever.get_relevant_documents(query)

        if not docs:
            return f"No relevant documents found for query: '{query}'"

        result = f"Search results for '{query}':\n\n"
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "Unknown")
            content_preview = (
                doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            )
            result += f"{i}. From {filename}:\n{content_preview}\n\n"

        return result

    except Exception as e:
        return f"Error searching documents: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_conversation_summary(messages: List[Dict]) -> str:
    """Summarize the current conversation"""
    if not messages:
        return "No conversation history available."

    summary = "Conversation Summary:\n"
    summary += f"Total messages: {len(messages)}\n"

    if len(messages) > 0:
        summary += f"Started: {messages[0].get('timestamp', 'Unknown')}\n"
        summary += f"Last message: {messages[-1].get('timestamp', 'Unknown')}\n"

    return summary


def get_available_tools(vector_db_manager):
    """Get list of available tools with vector_db_manager bound"""
    return [
        list_available_documents.bind(vector_db_manager=vector_db_manager),
        search_documents.bind(vector_db_manager=vector_db_manager),
        get_current_time,
        get_conversation_summary,
    ]
