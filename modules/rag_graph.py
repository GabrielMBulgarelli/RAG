from typing import TypedDict, List, Dict, Any, Annotated
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import json
from datetime import datetime

from config import config
from vector_db import VectorDBManager
from tools import get_available_tools
from error_handler import handle_errors, logger

class ConversationState(TypedDict):
    """State for the conversation graph"""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_query: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class StructuredResponse(BaseModel):
    """Schema for structured LLM responses"""
    answer: str = Field(description="The main answer to the user's question")
    confidence: float = Field(description="Confidence level (0-1) in the answer")
    sources_used: List[str] = Field(description="List of sources used in the response")
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

class RAGGraph:
    def __init__(self, vector_db_manager: VectorDBManager):
        self.vector_db_manager = vector_db_manager
        self.llm = ChatOllama(
            model=config.llm_model,
            temperature=config.temperature,
            base_url=config.ollama_base_url
        )
        
        # Set up tools
        self.tools = get_available_tools(vector_db_manager)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create graph
        self.graph = self._create_graph()
        self.checkpointer = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
    
    def _create_graph(self) -> StateGraph:
        """Create the conversation graph"""
        graph = StateGraph(ConversationState)
        
        # Add nodes
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("structured_output", self._structured_output_node)
        
        # Add edges
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_conditional_edges(
            "generate",
            tools_condition,
            {
                "tools": "tools",
                "end": "structured_output"
            }
        )
        graph.add_edge("tools", "generate")
        graph.add_edge("structured_output", END)
        
        return graph
    
    @handle_errors
    def _retrieve_node(self, state: ConversationState) -> ConversationState:
        """Retrieve relevant documents"""
        query = state["user_query"]
        
        try:
            retriever = self.vector_db_manager.get_retriever()
            docs = retriever.get_relevant_documents(query)
            
            retrieved_docs = []
            for doc in docs:
                retrieved_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("filename", "Unknown")
                })
            
            state["retrieved_docs"] = retrieved_docs
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
            
        except Exception as e:
            logger.error(f"Error in retrieve node: {e}")
            state["retrieved_docs"] = []
        
        return state
    
    @handle_errors
    def _generate_node(self, state: ConversationState) -> ConversationState:
        """Generate response using LLM"""
        query = state["user_query"]
        docs = state["retrieved_docs"]
        messages = state.get("messages", [])
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['content']}"
            for doc in docs[:config.k_retrieval]
        ])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to a knowledge base and various tools.
            
Use the provided context to answer questions accurately. If you need additional information 
or want to perform specific actions, use the available tools.

Context from knowledge base:
{context}

Guidelines:
- Provide accurate, helpful responses based on the context
- Use tools when appropriate to enhance your response
- If you can't find relevant information, say so honestly
- Include sources when referencing specific information
- Be conversational and engaging"""),
            ("human", "{query}")
        ])
        
        # Format the prompt
        formatted_messages = prompt.format_messages(
            context=context,
            query=query
        )
        
        # Add conversation history
        if messages:
            formatted_messages = messages + formatted_messages
        
        # Generate response
        response = self.llm_with_tools.invoke(formatted_messages)
        
        # Update messages
        new_messages = messages + [
            HumanMessage(content=query),
            response
        ]
        state["messages"] = new_messages
        
        return state
    
    @handle_errors
    def _structured_output_node(self, state: ConversationState) -> ConversationState:
        """Generate structured output"""
        messages = state["messages"]
        docs = state["retrieved_docs"]
        
        if not messages:
            state["final_answer"] = "No response generated."
            return state
        
        # Get the last AI message
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            answer = last_message.content
        else:
            answer = str(last_message)
        
        # Create structured response
        sources_used = [doc["source"] for doc in docs] if docs else []
        
        try:
            # Use LLM to generate structured output
            structured_llm = self.llm.with_structured_output(StructuredResponse)
            
            structure_prompt = f"""
            Based on this conversation response, create a structured output:
            
            Response: {answer}
            Sources used: {sources_used}
            
            Provide confidence level, sources, and suggest follow-up questions.
            """
            
            structured_response = structured_llm.invoke(structure_prompt)
            
            state["final_answer"] = structured_response.answer
            state["metadata"] = {
                "confidence": structured_response.confidence,
                "sources_used": structured_response.sources_used,
                "follow_up_questions": structured_response.follow_up_questions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate structured output: {e}")
            state["final_answer"] = answer
            state["metadata"] = {
                "sources_used": sources_used,
                "timestamp": datetime.now().isoformat()
            }
        
        return state
    
    @handle_errors
    def process_query(self, query: str, conversation_id: str = "default") -> Dict[str, Any]:
        """Process a user query through the graph"""
        
        # Initial state
        initial_state = ConversationState(
            messages=[],
            user_query=query,
            retrieved_docs=[],
            final_answer="",
            conversation_history=[],
            metadata={}
        )
        
        # Run the graph
        config_dict = {"configurable": {"thread_id": conversation_id}}
        
        result = self.compiled_graph.invoke(initial_state, config=config_dict)
        
        return {
            "answer": result["final_answer"],
            "sources": [doc["source"] for doc in result["retrieved_docs"]],
            "metadata": result.get("metadata", {}),
            "conversation_id": conversation_id
        }
    
    def get_conversation_history(self, conversation_id: str = "default") -> List[Dict]:
        """Get conversation history for a thread"""
        try:
            config_dict = {"configurable": {"thread_id": conversation_id}}
            history = []
            
            for state in self.compiled_graph.get_state_history(config=config_dict):
                if state.values.get("messages"):
                    for msg in state.values["messages"]:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            history.append({
                                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                                "content": msg.content,
                                "timestamp": datetime.now().isoformat()
                            })
            
            return history[-10:]  # Return last 10 messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []