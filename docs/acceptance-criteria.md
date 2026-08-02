# Acceptance Criteria

This document defines the user-visible completion contract for the local
React/FastAPI workspace. A release record should identify the commit tested and
record the outcome of each applicable check.

## 1. Startup and workspace

- `uv run python -m modules.run` serves the application at
  `http://127.0.0.1:7860` by default.
- `/` displays one continuous workspace containing the sidebar, conversation,
  collapsible inspector, and overlay controller.
- A browser refresh stays on `/`; benchmark results, case inspection,
  diagnostics, and confirmations do not introduce separate application pages.
- With Ollama unavailable, the UI still loads and presents actionable blocked
  state rather than a blank page or startup failure.
- Keyboard users can reach all controls. Dialogue focus is contained while open
  and returns to the invoking control when closed.

## 2. Runtime and documents

- Diagnostics distinguish Ollama connectivity, required model availability,
  model load state, uploaded-index health, and benchmark preparation.
- Loading a model updates capabilities without reloading the browser page.
- PDF and TXT uploads appear in the indexed-document list after successful
  ingestion.
- Deleting a document requires confirmation and removes its manifest entry,
  source copy, and indexed chunks.
- A missing uploaded-document manifest is reported as an empty corpus.
- An unreadable manifest, missing Chroma data, missing chunks, or index settings
  that do not match the active settings is reported as an actionable index
  error and is not silently replaced with an empty corpus.

## 3. Conversation and evidence

- A successful query appends the user question and assistant response to the
  current session.
- Supported answers expose cited sources with filename, page, and readable
  excerpt; limited answers and abstentions are visibly distinct.
- The inspector exposes retrieval hits, query diagnostics, and public execution
  trace without exposing private reasoning.
- Clearing a conversation removes the active session after confirmation.
- Export produces a readable, privacy-safe conversation record.
- Querying remains disabled whenever another coordinated operation owns the
  workspace.

## 4. Benchmark preparation

- The benchmark corpus and index are independent from uploaded documents.
- When benchmark files or populated index data are absent, diagnostics display
  this exact preparation command:

  ```bash
  uv run python scripts/prepare_multihop_eval.py --index
  ```

- Benchmark execution remains disabled until the benchmark cases, source map,
  non-empty manifest, and populated Chroma index are present and models are
  loaded.
- The application does not download or index the benchmark corpus from the UI.

## 5. Benchmark execution and cancellation

- Starting a benchmark creates one run for the canonical 20 case IDs across
  `dense`, `bm25`, `hybrid`, `dense-rag`, `bm25-rag`, `hybrid-rag`, and
  `full-rag`.
- Progress reports the current system and case through server-sent events.
- Metric observations distinguish measured values, metrics that do not apply,
  and metrics with no eligible cases.
- Requesting cancellation prevents later cases and model calls from starting,
  retains completed case data, and keeps querying disabled until the active
  request exits and the run reaches `cancelled`.
- The UI states that an already-running Ollama request may finish before
  cancellation completes; it does not promise immediate interruption.
- Request timeout failures and other runtime failures are recorded against the
  affected case and included in failure aggregates.

## 6. Benchmark results and cases

- A completed run presents Summary, Retrieval, Grounding, Execution, Cases, and
  Failures tabs.
- Summary shows run metadata, progress, expectation-failure aggregates, and
  runtime-failure aggregates.
- Retrieval, Grounding, and Execution use their stable section IDs and render
  only each section's declared systems.
- Cases lists successful, expectation-failure, and runtime-failure outcomes and
  opens each available case for inspection.
- Failures groups failures by recorded classification and opens the affected
  case.
- Case inspection shows the question, system, outcome, observations, trace, and
  structured evidence cards.
- Every evidence item exposes chunk ID, document ID, filename, page, and a
  normalized bounded excerpt, using content when an excerpt is absent.
- Raw evidence diagnostics remain available as an expandable secondary view.
- Case inspection uses the existing dialogue and returns focus to the invoking
  case control when closed.

## 7. Persistence and reopening

- Each run directory under `evals/results/full_rag/` contains `run.json`,
  `summary.json`, `cases.jsonl`, and `events.jsonl`.
- The run records the git commit, dataset identifier and hash, exact case IDs,
  models, temperature, prompt identity, graph configuration, chunking settings,
  retrieval limits, context limits, retry and subquery limits, request timeout,
  and execution timestamps.
- Completed case records remain available after cancellation or a later
  process restart.
- **View latest results** reopens the latest valid completed run after a browser
  refresh or application restart.
- A corrupt or incomplete run directory is not returned as the latest valid
  completed run.
- Download returns a ZIP containing the four persisted run files.

## 8. API contract

The following endpoints respond through FastAPI under `/api`:

- `GET /runtime`, `POST /runtime/models`, and `GET /diagnostics`
- `GET /documents`, `POST /documents`, and
  `DELETE /documents/{document_id}`
- `POST /query`, `DELETE /conversations/{session_id}`, and
  `POST /conversations/export`
- `POST /benchmarks` and `GET /benchmarks/latest`
- `GET /benchmarks/{run_id}` and `GET /benchmarks/{run_id}/events`
- `POST /benchmarks/{run_id}/cancel` and
  `GET /benchmarks/{run_id}/download`
- `GET /benchmarks/{run_id}/cases`
- `GET /benchmarks/{run_id}/cases/{case_id}/systems/{system_id}`

API failures use the shared problem response and do not require the React client
to interpret backend tracebacks.

## 9. Verification record

Run the repository gate from the project root:

```bash
./scripts/verify.sh
```

The gate covers locked dependency installation, Vitest, TypeScript, the
production frontend build, the frontend dependency audit, Playwright, Ruff,
Pyright, Lanorme, and the backend suite excluding live-Ollama tests. It also
runs the offline runtime diagnostic unconditionally and fails if that
diagnostic reports any issue.

For focused development checks:

```bash
npm --prefix frontend test -- --run
npm --prefix frontend run typecheck
npm --prefix frontend run build
uv run pytest -m "not ollama" --no-cov
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Browser behaviour, responsive layouts, and live-Ollama behaviour require their own
recorded validation. Passing unit tests alone is not evidence that those checks
were performed.
