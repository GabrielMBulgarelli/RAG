# Local Document RAG

A local-first Retrieval-Augmented Generation workbench for asking questions
about PDF and TXT documents with inspectable evidence. A React interface and a
FastAPI backend run as one local application, using Ollama for chat and
embedding models.

## Application workspace

The application has one continuous route at `/`:

- The sidebar owns uploads, indexed-document management, model state,
  benchmark actions, and system diagnostics.
- The conversation is the primary surface. It keeps the composer, cited
  answers, clear action, and export action together.
- The collapsible inspector shows sources, retrieval details, query decisions,
  and the execution trace for the selected answer.
- Benchmark progress, benchmark results, case details, diagnostics, and
  confirmations open as overlays without replacing the workspace.

FastAPI exposes the typed `/api` contract and serves the built React application
from `/`. The workspace opens when Ollama is unavailable and explains which
actions are blocked.

## RAG workflow

- Upload, index, list, and delete PDF or TXT sources.
- Combine semantic search and BM25 with Reciprocal Rank Fusion, deterministic
  diversity selection, and stable chunk provenance.
- Route simple questions directly, decompose multi-hop questions, run
  independent retrieval work concurrently, and allow at most one targeted
  retry.
- Grade evidence, filter irrelevant context, validate citations, and abstain
  when the indexed sources do not support an answer.
- Inspect cited excerpts, filenames, pages, retrieval scores, and public trace
  events. Conversation exports omit private reasoning.

Uploaded-document state is stored under `sources/` and `data/`. A missing
`data/manifest.json` represents an empty corpus. If the manifest cannot be read,
the Chroma data is missing, chunks are absent, or the saved index settings do
not match the active settings, diagnostics report an actionable index error
instead of treating the corpus as empty.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 22
- [Ollama](https://ollama.com/)
- `qwen3.5:9b`
- `nomic-embed-text`

## Quickstart

```bash
git clone https://github.com/GabrielMBulgarelli/RAG.git
cd RAG

uv python install 3.12
uv sync
npm --prefix frontend ci
npm --prefix frontend run build

ollama pull qwen3.5:9b
ollama pull nomic-embed-text
ollama serve
```

In another terminal, start the application:

```bash
uv run python -m modules.run
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). When Ollama is offline,
the workspace remains available with blocked-state guidance.

## Full RAG Benchmark

The benchmark uses the canonical 20-case MultiHopRAG development set and seven
systems:

| Group | Systems |
| --- | --- |
| Retrieval only | `dense`, `bm25`, `hybrid` |
| Fixed single-call RAG | `dense-rag`, `bm25-rag`, `hybrid-rag` |
| Bounded workflow | `full-rag` |

Prepare its files and populated index before the first run:

```bash
uv run python scripts/prepare_multihop_eval.py --index
```

The benchmark corpus under `evals/` is separate from documents uploaded through
the application. Preparation does not alter the uploaded-document index.
Diagnostics check the benchmark cases, source mapping, manifest, Chroma data,
and chunk population. The **Run benchmark** action remains disabled until those
checks pass and the models are loaded.

Each run records its commit, dataset identifier and hash, exact case IDs,
models, temperature, prompt identity, graph settings, chunking settings,
retrieval and context limits, retry and subquery limits, request timeout, and
execution timestamps. Run data is stored in
`evals/results/full_rag/<run-id>/` as `run.json`, `summary.json`, `cases.jsonl`,
and `events.jsonl`. A completed run can be reopened with **View latest results**
after a browser refresh or application restart, or downloaded as a ZIP archive.

Results contain these tabs:

- **Summary** shows run progress, recorded metadata, failure aggregates, and
  runtime-failure aggregates.
- **Retrieval**, **Grounding**, and **Execution** show their stable metric
  sections and only the systems named by each section.
- **Cases** lists successful, expectation-failure, and runtime-failure results.
- **Failures** groups recorded failures and opens the affected case.

Case inspection shows the question, outcome, metric observations, execution
trace, and structured evidence cards. Every evidence card includes chunk ID,
document ID, filename, page, and a bounded readable excerpt; raw diagnostics
remain expandable.

Cancellation is cooperative. Requesting cancellation prevents later cases and
model calls from starting, preserves completed case data, and keeps querying
disabled until the active request exits and the benchmark reaches a terminal
state. An Ollama request already in progress may finish before cancellation
completes.

The same evaluation can be run from the command line:

```bash
uv run python -m modules.evaluation \
  --systems all \
  --split development \
  --dataset multihop \
  --model qwen3.5:9b
```

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/runtime` | Read model, corpus, capability, and active-operation state. |
| `POST` | `/api/runtime/models` | Load the selected local chat model and workspace services. |
| `GET` | `/api/diagnostics` | Read runtime, uploaded-index, and benchmark-preparation checks. |
| `GET` | `/api/documents` | List indexed uploaded documents. |
| `POST` | `/api/documents` | Upload and index PDF or TXT files. |
| `DELETE` | `/api/documents/{document_id}` | Remove an indexed document. |
| `POST` | `/api/query` | Ask a question in a conversation session. |
| `DELETE` | `/api/conversations/{session_id}` | Clear a conversation. |
| `POST` | `/api/conversations/export` | Export a privacy-safe conversation record. |
| `POST` | `/api/benchmarks` | Start the seven-system benchmark. |
| `GET` | `/api/benchmarks/latest` | Reopen the latest completed run. |
| `GET` | `/api/benchmarks/{run_id}` | Read a run snapshot. |
| `GET` | `/api/benchmarks/{run_id}/events` | Stream progress with server-sent events. |
| `POST` | `/api/benchmarks/{run_id}/cancel` | Request cooperative cancellation. |
| `GET` | `/api/benchmarks/{run_id}/cases` | List case and system outcomes. |
| `GET` | `/api/benchmarks/{run_id}/cases/{case_id}/systems/{system_id}` | Inspect one case result. |
| `GET` | `/api/benchmarks/{run_id}/download` | Download the persisted run archive. |

## Configuration

Settings use the `RAG_` prefix and can be placed in `.env`:

```dotenv
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_LLM_MODEL=qwen3.5:9b
RAG_EMBEDDING_MODEL=nomic-embed-text
RAG_SERVER_HOST=127.0.0.1
RAG_SERVER_PORT=7860
```

Retrieval budgets, chunking, retry limits, paths, and server settings are
validated in [`modules/config.py`](modules/config.py) before runtime clients are
constructed.

## Verification

Run the repository gate:

```bash
./scripts/verify.sh
```

It installs locked dependencies, runs Vitest, TypeScript checks, the production
frontend build, the frontend dependency audit, Ruff, Pyright, Lanorme, and the
backend suite excluding tests marked for live Ollama. The final diagnostics
step runs only when the standalone diagnostics module is present.

Useful focused commands are:

```bash
npm --prefix frontend test -- --run
npm --prefix frontend run typecheck
npm --prefix frontend run build
uv run pytest -m "not ollama" --no-cov
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The user-facing completion contract is in
[`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md).

## Project structure

```text
frontend/                   # React workspace and benchmark overlays
modules/
├── api/                    # FastAPI routes, lifecycle, and error mapping
├── application/            # Workspace and benchmark services
├── app.py                  # FastAPI and built-frontend launcher
├── bootstrap.py            # Production dependency composition
├── citations.py            # Answer and citation validation
├── evaluation.py           # Seven-system benchmark CLI
├── rag_graph.py            # Bounded retrieval and answer workflow
├── retrieval.py            # Semantic, BM25, fusion, and selection
├── vector_db.py            # Ingestion, manifest, and Chroma lifecycle
└── run.py                  # Package-safe local entry point

scripts/
├── prepare_multihop_eval.py
└── verify.sh

tests/                      # Backend unit and integration coverage
```

## Acknowledgments

The project began from the
[AI Workshop 2025 GenAI Session](https://github.com/Antonio-Tresol/ai_workshop_2025_gen_ai_session)
and has expanded with document lifecycle management, hybrid retrieval, bounded
orchestration, citation validation, evaluation, and a React/FastAPI workspace.
