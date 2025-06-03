import gradio as gr
import uuid
from typing import List, Tuple, Dict, Any
from datetime import datetime

from config import config
from vector_db import VectorDBManager
from rag_graph import RAGGraph
from error_handler import RAGError, logger

class RAGApplication:
    def __init__(self):
        self.vector_db_manager = VectorDBManager()
        self.rag_graph = None
        self.conversation_sessions = {}
        self.setup_complete = False
    
    def initialize(self):
        """Initialize the RAG system"""
        try:
            logger.info("Initializing RAG application...")
            
            # Setup vector database
            self.vector_db_manager.setup()
            
            # Initialize RAG graph
            self.rag_graph = RAGGraph(self.vector_db_manager)
            
            self.setup_complete = True
            logger.info("RAG application initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG application: {e}")
            raise RAGError(f"Initialization failed: {str(e)}")
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.conversation_sessions[session_id] = {
            "created_at": datetime.now(),
            "message_count": 0
        }
        return session_id
    
    def chat_response(self, message: str, history: List[Tuple[str, str]], session_id: str = None) -> Tuple[str, List[Tuple[str, str]]]:
        """Handle chat messages"""
        if not self.setup_complete:
            return "System not initialized. Please wait...", history
        
        if not session_id:
            session_id = self.create_session()
        
        try:
            # Process query through RAG graph
            result = self.rag_graph.process_query(message, session_id)
            
            # Format response
            response = result["answer"]
            
            # Add metadata if available
            metadata = result.get("metadata", {})
            if metadata:
                response += "\n\n---\n"
                
                if metadata.get("sources_used"):
                    response += f"📚 **Sources**: {', '.join(metadata['sources_used'])}\n"
                
                if metadata.get("confidence"):
                    confidence = metadata["confidence"]
                    response += f"🎯 **Confidence**: {confidence:.1%}\n"
                
                if metadata.get("follow_up_questions"):
                    response += f"💡 **Follow-up questions**: {', '.join(metadata['follow_up_questions'])}\n"
            
            # Update session
            if session_id in self.conversation_sessions:
                self.conversation_sessions[session_id]["message_count"] += 1
            
            # Update history
            history.append((message, response))
            
            return response, history
            
        except RAGError as e:
            error_msg = f"❌ **Error**: {e.message}\n\n"
            if e.suggestions:
                error_msg += "**Suggestions**:\n" + "\n".join(f"• {s}" for s in e.suggestions)
            
            history.append((message, error_msg))
            return error_msg, history
            
        except Exception as e:
            error_msg = f"❌ **Unexpected error**: {str(e)}"
            history.append((message, error_msg))
            return error_msg, history
    
    def get_system_info(self) -> str:
        """Get system information"""
        if not self.setup_complete:
            return "❌ System not initialized"
        
        info = f"""
        ✅ **RAG System Status**: Active
        🤖 **Model**: {config.llm_model}
        📊 **Embedding Model**: {config.embedding_model}
        💾 **Vector DB**: ChromaDB
        📁 **Sources Directory**: {config.sources_dir}
        🔧 **Active Sessions**: {len(self.conversation_sessions)}
        """
        
        return info
    
    def list_documents(self) -> str:
        """List available documents"""
        if not self.setup_complete:
            return "System not initialized"
        
        try:
            from tools import list_available_documents
            return list_available_documents(self.vector_db_manager)
        except Exception as e:
            return f"Error listing documents: {str(e)}"

# Initialize application
app = RAGApplication()

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title=config.app_title,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
        }
        """
    ) as interface:
        
        gr.Markdown(f"# {config.app_title}")
        gr.Markdown(config.app_description)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Chat with RAG Assistant",
                    height=500,
                    show_copy_button=True
                )
                
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask me anything about the documents...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                # Example questions
                gr.Examples(
                    examples=[
                        "What documents do you have access to?",
                        "Tell me something interesting from the documents",
                        "What topics can you help me with?",
                        "Search for specific information"
                    ],
                    inputs=msg
                )
            
            with gr.Column(scale=1):
                # System information panel
                gr.Markdown("## System Info")
                system_info = gr.Textbox(
                    value=app.get_system_info(),
                    label="Status",
                    lines=8,
                    interactive=False
                )
                
                refresh_info_btn = gr.Button("Refresh Info")
                
                gr.Markdown("## Available Documents")
                doc_list = gr.Textbox(
                    value=app.list_documents(),
                    label="Documents",
                    lines=6,
                    interactive=False
                )
                
                refresh_docs_btn = gr.Button("Refresh Documents")
        
        # State for session management
        session_state = gr.State(value=app.create_session())
        
        # Event handlers
        def chat_fn(message, history, session_id):
            if not message.strip():
                return "", history
            
            response, updated_history = app.chat_response(message, history, session_id)
            return "", updated_history
        
        def clear_fn():
            return [], app.create_session()
        
        def refresh_info_fn():
            return app.get_system_info()
        
        def refresh_docs_fn():
            return app.list_documents()
        
        # Connect events
        send_btn.click(
            fn=chat_fn,
            inputs=[msg, chatbot, session_state],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            fn=chat_fn,
            inputs=[msg, chatbot, session_state],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            fn=clear_fn,
            outputs=[chatbot, session_state]
        )
        
        refresh_info_btn.click(
            fn=refresh_info_fn,
            outputs=[system_info]
        )
        
        refresh_docs_btn.click(
            fn=refresh_docs_fn,
            outputs=[doc_list]
        )
    
    return interface

def main():
    """Main function to run the application"""
    try:
        print("🚀 Starting RAG Application...")
        print("⏳ Initializing system (this may take a moment)...")
        
        # Initialize the application
        app.initialize()
        
        print("✅ System initialized successfully!")
        print(f"🌐 Starting web interface...")
        
        # Create and launch interface
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("💡 Check the error messages above for troubleshooting tips")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())