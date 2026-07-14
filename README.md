# RAG Application

A simple Retrieval-Augmented Generation application with advanced capabilities for document-based question answering.

## Features

- Document processing and retrieval
- LLM-powered question answering
- Support for multiple document formats (PDF, TXT)
- Vector database storage with ChromaDB
- Conversation memory and context awareness
- Web interface with Gradio
- Document management and system monitoring

## Prerequisites

- [UV](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running
- Required models:
  - qwen3.5:9b
  - nomic-embed-text

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GabrielMBulgarelli/RAG
cd RAG
```

2. Install the managed Python interpreter and project environment:
```bash
uv python install 3.12
uv sync
```

3. Install and start Ollama:
- Download from [ollama.com](https://ollama.com)
- Install and start the application
- Pull required models:
```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

## Usage

1. Add your documents:
- Place PDF or TXT files in the `sources/` directory
- The system will automatically process them

2. Start the application:
```bash
uv run python -m modules.run
```

3. Access the web interface:
- Open your browser and go to [http://localhost:7860](http://localhost:7860)
- Use the chat interface to ask questions about your documents

## Project Structure

```
RAG/
├── pyproject.toml         # Project metadata, dependencies, and quality configuration
├── uv.lock                # Reproducible dependency lock
├── sources/              # Document storage
├── data/chroma/          # Vector database
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

Copy `.env.example` to `.env` and customize the `RAG_`-prefixed settings.
Unknown or invalid settings are rejected before Ollama or Chroma clients are
constructed.

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

2. If dependencies need to be refreshed:
```bash
uv sync
```

3. To reset the vector database:
```bash
rm -rf chroma_db/*
```
## License

This project is adapted from the [AI Workshop 2025 GenAI Session](https://github.com/Antonio-Tresol/ai_workshop_2025_gen_ai_session). Please refer to the original project's license terms when using this code.

## Planned acceptance checklist

The following items reflect the pending acceptance criteria still to be implemented.

### Ingestion
- [ ] TXT supported
- [ ] Stable document and chunk IDs
- [ ] Unchanged reindex creates no duplicates
- [ ] Modified files replace stale chunks
- [ ] Deleted files remove chunks
- [ ] Duplicate filenames remain distinct by path
- [ ] Failed ingestion preserves prior usable version
- [ ] Page/filename metadata present
- [ ] Atomic manifest writes
- [ ] Manifest/Chroma reconciliation

### Agentic workflow
- [ ] Follow-up rewriting only when needed
- [ ] Catalog, clarification, and out-of-scope avoid retrieval
- [ ] Simple search uses a direct bounded path
- [ ] Complex search may create a bounded plan
- [ ] Strategy selection changes retrieval
- [ ] Evidence assessment changes execution
- [ ] Retry only when justified
- [ ] Default retry never exceeds one
- [ ] Every path terminates
- [ ] Supported, limited, unsupported, clarification, catalog, and out-of-scope outcomes

### Retrieval and citations
- [ ] Semantic, BM25, and hybrid retrieval
- [ ] Reciprocal rank fusion
- [ ] Scores retained
- [ ] ID-based deduplication
- [ ] Filename filters
- [ ] Subquery provenance
- [ ] Every citation maps to a retrieved chunk
- [ ] Correct filename/page
- [ ] Unknown citation IDs rejected
- [ ] Unsupported answers do not fabricate citations
- [ ] No second model call invents sources/confidence

### Trace
- [ ] Structured public events
- [ ] Counts, decisions, and durations where practical
- [ ] No private chain-of-thought

### Evaluation
- [ ] Development and held-out splits
- [ ] Dense, BM25, hybrid, and agentic systems
- [ ] Recall@5, MRR@5, nDCG@5
- [ ] Route and strategy accuracy
- [ ] Retry precision/recall
- [ ] Citation precision/coverage
- [ ] Abstention/conflict accuracy
- [ ] Termination rate
- [ ] Mean and p95 latency
- [ ] LLM calls per query
- [ ] Stored experiment configuration
- [ ] Failure taxonomy

### UI
- [ ] Gradio retained
- [ ] Upload/index/reindex/delete/rebuild
- [ ] Progress, status, page/chunk counts, errors
- [ ] Clear/export chat
- [ ] Citations and excerpts
- [ ] Route, strategy, subqueries, retry, evidence, trace
- [ ] Evaluation comparison and failed cases
- [ ] Diagnostics
- [ ] No uncalibrated confidence percentage
- [ ] No hidden reasoning

### Initial quality gates
- [ ] Route accuracy ≥ 0.90
- [ ] Workflow termination = 1.00
- [ ] Citation integrity = 1.00
- [ ] Retrieval Recall@5 ≥ 0.85
- [ ] Unanswerable abstention accuracy ≥ 0.85
- [ ] No path exceeds configured retry/subquery limits
- [ ] Ordinary CI does not require Ollama

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Uses [Ollama](https://ollama.com/) for local LLM
- Interface powered by [Gradio](https://gradio.app/)
- Special thanks to Antonio-Tresol for providing the foundation for this implementation
