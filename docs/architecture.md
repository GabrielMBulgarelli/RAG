# Architecture

## System context

The project is a local, single-user React and FastAPI application. The browser
owns interaction and presentation; the server owns document lifecycle,
retrieval, generation, evaluation, diagnostics, coordination, and persistence.

```text
Browser
  -> React workspace
  -> FastAPI routes
  -> application services
       -> operation coordinator
       -> document lifecycle
       -> workspace query service
       -> benchmark manager
  -> RAG graph and evaluation adapters
  -> Chroma, Ollama, source files, and persisted benchmark artifacts
```

## Component ownership

- `frontend/src/` owns routing, workspace state, overlays, responsive layouts,
  accessible interaction, progress presentation, result exploration, and API
  consumption.
- `modules/api/routes.py` owns the HTTP boundary and maps API requests to
  application services.
- `modules/application/` owns workspace orchestration, document operations,
  diagnostics, operation leases, benchmark lifecycle, cancellation, artifact
  projection, and download packaging.
- `modules/rag_graph.py` owns the bounded adaptive query workflow.
- `modules/evaluation.py`, `modules/evaluation_metrics.py`, and the evaluation
  models own benchmark execution, measurements, classifications, and artifact
  serialization.
- `modules/retrieval.py` and `modules/llm.py` adapt Chroma and Ollama without
  making those infrastructure details browser concerns.

## Workspace query flow

1. Source PDF and TXT files are stored under `sources/`.
2. Import writes the workspace manifest at `data/manifest.json` and vectors at
   `data/chroma/`.
3. Diagnostics validate source, manifest, index, embedding, and model readiness
   before query execution.
4. The graph combines semantic and BM25 retrieval with stable fusion, then may
   decompose the question, grade evidence, retry retrieval once, synthesize an
   answer, validate citations, or abstain.
5. The API returns the answer, citations, evidence payloads, and a compact
   execution trace for inspection in the React workspace.

The graph is bounded to at most four subqueries and one retry. These limits make
termination observable and keep the evaluation contract reproducible.

## Benchmark flow

1. The preparation script selects the canonical development cases and builds a
   separate benchmark index at `data/multihop_chroma/`.
2. Benchmark diagnostics confirm that dataset, manifest, index, models, and
   configuration are ready.
3. The benchmark manager executes requested case-system pairs, emits progress
   events, and persists cases incrementally.
4. Presentation services aggregate applicable metrics, project case traces,
   classify failures, and expose Summary, Retrieval, Grounding, Execution,
   Cases, and Failures views.
5. A valid complete canonical run may become the latest result. Every retained
   run can be reopened or downloaded independently.

## Storage boundaries

- `sources/` contains user-provided workspace documents.
- `data/manifest.json` and `data/chroma/` contain workspace index metadata and
  vectors.
- `data/multihop_eval/` and `data/multihop_chroma/` contain canonical benchmark
  inputs and vectors.
- `data/evaluation_results/` contains run directories with `run.json`,
  `summary.json`, `cases.jsonl`, and `events.jsonl`.

Workspace and benchmark indexes are deliberately separate. A benchmark result
therefore measures a prepared, fixed corpus rather than the user's current
workspace contents.

## Coordination, persistence, and cancellation

Document writes and benchmark runs acquire one process-local operation lease.
Conflicting operations are rejected with explicit state rather than running
concurrently against shared assets. Progress is emitted as events and persisted
alongside run state.

Cancellation is cooperative at case boundaries. Work already written remains
inspectable, while cancelled, failed, incomplete, or noncanonical runs cannot
replace the latest valid complete canonical result. Persistence lets the server
recover completed results across restarts; it does not attempt distributed
locking or multi-process coordination.

## Design decisions and limits

- One main workspace route keeps navigation stable; dense tools use overlays,
  drawers, and sheets rather than separate page hierarchies.
- Evidence identifiers and readable source text cross the API boundary so the
  UI can verify claims instead of displaying opaque scores.
- Corrupt index metadata is a first-class diagnostic state, not a missing-index
  fallback.
- Fixed graph bounds, explicit timeouts, and cooperative cancellation favour
  debuggability over open-ended agent behaviour.
- Local filesystem state, process-local coordination, and the absence of
  authentication are intentional single-user boundaries, not a production
  multi-tenant architecture.

The benchmark contract and metric applicability rules are defined separately
in the [benchmark methodology](benchmark-methodology.md).
