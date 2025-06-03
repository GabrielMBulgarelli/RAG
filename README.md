# RAG Application

A simple Retrieval-Augmented Generation application with advanced capabilities for document-based question answering.

## Features

- 🔍 Document processing and retrieval
- 🤖 LLM-powered question answering
- 📚 Support for multiple document formats (PDF, TXT)
- 💾 Vector database storage with ChromaDB
- 🧠 Conversation memory and context awareness
- 🌐 Web interface with Gradio
- 📊 Document management and system monitoring

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running
- Required models:
  - llama3.1
  - nomic-embed-text

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG
```

2. Run the setup script:
```bash
python modules/setup.py
```

3. Install and start Ollama:
- Download from [ollama.com](https://ollama.com)
- Install and start the application
- Pull required models:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Usage

1. Add your documents:
- Place PDF or TXT files in the `sources/` directory
- The system will automatically process them

2. Start the application:
```bash
python modules/run.py
```

3. Access the web interface:
- Open your browser and go to [http://localhost:7860](http://localhost:7860)
- Use the chat interface to ask questions about your documents

## Project Structure

```
RAG/
├── requirements.txt       # Project dependencies
├── sources/              # Document storage
├── chroma_db/           # Vector database
├── logs/                # Application logs
└── modules/
    ├── app.py           # Main application
    ├── config.py        # Configuration
    ├── error_handler.py # Error handling
    ├── rag_graph.py    # RAG implementation
    ├── run.py          # Application runner
    ├── setup.py        # Setup script
    ├── tools.py        # Utility tools
    └── vector_db.py    # Vector DB management
```

## Configuration

Edit `modules/config.py` to customize:
- Model settings (`llm_model`, `embedding_model`, `temperature`)
- Vector DB settings (`chunk_size`, `chunk_overlap`, `k_retrieval`)
- Application settings (`app_title`, `app_description`)

## Features in Detail

### Document Processing
- Automatic text extraction from PDFs
- Smart text chunking for better retrieval
- Metadata extraction and storage

### RAG System
- Advanced retrieval using vector similarity
- Context-aware response generation
- Source tracking and citation

### User Interface
- Interactive chat interface
- System status monitoring
- Document management
- Conversation history

## Troubleshooting

1. If Ollama is not running:
```bash
ollama serve
```

2. If dependencies conflict:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. To reset the vector database:
```bash
rm -rf chroma_db/*
```
## License

This project is adapted from the [AI Workshop 2025 GenAI Session](https://github.com/Antonio-Tresol/ai_workshop_2025_gen_ai_session). Please refer to the original project's license terms when using this code.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Uses [Ollama](https://ollama.com/) for local LLM
- Interface powered by [Gradio](https://gradio.app/)
- Based on the [AI Workshop 2025 GenAI Session](https://github.com/Antonio-Tresol/ai_workshop_2025_gen_ai_session) by Antonio Tresol
- Special thanks to Antonio-Tresol for providing the foundation for this implementation
