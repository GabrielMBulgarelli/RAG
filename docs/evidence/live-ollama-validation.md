# Live Ollama workflow validation

The live workflow was validated against the production FastAPI application and
its served React build on 2026-08-01/02. This record contains scoped live Ollama
evidence; no model responses or benchmark results were simulated.

## Environment

- Tested commit: `bcd11d827c1fb6868e7242d39cec9434abce9727`
- Ollama: `0.31.2`
- Chat model: `qwen3.5:9b`, digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`, GGUF `Q4_K_M`, 9.7B parameters
- Embedding model: `nomic-embed-text:latest`, digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`, GGUF `F16`, 137M parameters
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- Dataset: `yixuantt/MultiHopRAG`, development split, dataset hash `5b7e367051b92d73df56a07bb04182daf649325639e601bb85eea57bd155d07c`

## Preparation and grounded query

`scripts/prepare_multihop_eval.py --index` resolved all 62 gold-evidence records and produced a benchmark manifest with 609 documents plus a Chroma index with 12,743 chunks. Preparation took 255.61 seconds. The frontend dependency install, TypeScript check, and Vite production build completed successfully.

The production application loaded `qwen3.5:9b` through the React model control. A real TXT fixture was uploaded and indexed as one document, one page, and one chunk. The UI accepted grounded questions, retrieved the exact fixture chunk, and exposed its retrieval score and complete execution trace in the Details inspector.

The first query coincided with a cold qwen load and returned HTTP 503 after about 60 seconds. Runtime diagnostics remained ready, Ollama completed the load, and an identical retry completed. The resident model then answered subsequent requests without that initialization failure. Although retrieval selected the exact chunk at essentially 1.0 score, qwen's structured evidence grade was incomplete and the application conservatively classified it as insufficient, so the visible answer abstained and had no cited Sources. This is a model-output/runtime deviation, not a workflow blocker.

## Complete benchmark

Completed run: `e9279a38-ef5b-416b-9c4c-5791a1c18cae`

- Started: `2026-08-01T23:29:25.241164Z`
- Completed: `2026-08-02T00:09:04.696325Z`
- Wall-clock duration: 2,379.455 seconds (39 minutes 39.455 seconds)
- Persisted results: exactly 140 unique case-system pairs: 20 cases for each of seven systems
- Aggregate sections: retrieval, grounding, and execution
- Clean successes: 42
- Expectation failures: 97
- Runtime failures: 1

Per-system duration is the sum of persisted case trace durations. The difference from wall time is benchmark setup, persistence, aggregation, and orchestration overhead.

| System | Results | Trace duration | Clean | Expectation failures | Runtime failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 20 | 3.045 s | 12 | 8 | 0 |
| BM25 | 20 | 5.235 s | 14 | 6 | 0 |
| Hybrid | 20 | 6.237 s | 12 | 8 | 0 |
| Dense RAG | 20 | 351.040 s (5m 51.040s) | 0 | 20 | 0 |
| BM25 RAG | 20 | 392.957 s (6m 32.957s) | 3 | 17 | 0 |
| Hybrid RAG | 20 | 511.186 s (8m 31.186s) | 1 | 19 | 0 |
| Full RAG | 20 | 977.920 s (16m 17.920s) | 0 | 19 | 1 |

The UI's Cases view distinguished and opened all three result classes. `multihop-2512` / Dense was inspected as a clean result, `multihop-0063` / Dense as an expectation failure (`retrieval_miss`), and `multihop-2512` / Full RAG as the runtime failure. The runtime failure recorded two LLM calls, zero retrieval rounds, 75.492 seconds, failed termination, and `route_error, strategy_error, retrieval_miss, citation_coverage_miss, non_termination, runtime_error`.

The React download control produced `benchmark-e9279a38-ef5b-416b-9c4c-5791a1c18cae.zip`. It contained exactly `run.json`, `summary.json`, `cases.jsonl`, and `events.jsonl`; the extracted files matched the persisted artifacts byte-for-byte. Persisted SHA-256 values were:

- `run.json`: `96d044f883e1a204889a0b2267bdf9e651006e242053a0dfafb53a7a31bbfc3d`
- `summary.json`: `2f0cdb51458ec5bb724a5bd29588406b10a6a8355310211755a876bfc35794fb`
- `cases.jsonl`: `517550072f0b80c1a29c3a38658e57370a46a05f3a820f4c6524fce7ba171f9d`
- `events.jsonl`: `c77ccd69c6806299b77783f3b59dbd95b2adba3a7764f860a09a9b6001f8db37`

After a clean application shutdown and production restart, `/api/benchmarks/latest` returned the same completed run and the React results dialogue reopened it. Diagnostics reported the benchmark files, 609-document manifest, 12,743-chunk benchmark index, and latest completed Full RAG artifact as ready. Model initialization correctly returned to `not_loaded` after restart and became ready after loading qwen again.

## Cancellation during an active model call

Cancellation run: `46da3944-d0b9-465f-883e-d67ca0116eaf`

- Run started at `2026-08-02T00:14:46.089549Z`.
- Dense RAG case `multihop-2512` started at `00:15:24.186602Z`.
- Cancellation was persisted at `00:15:27.611605Z` while its qwen call was active.
- The in-flight Ollama `/api/chat` call returned at approximately `00:15:32Z`; the case was persisted as a cancellation-induced runtime failure with one LLM call.
- Terminal `benchmark.cancelled` was persisted at `00:15:32.522663Z`, 4.911 seconds after the cancellation request.
- The final cancelled artifact contained 64 rows: 60 retrieval-only rows and four Dense RAG rows. No `case.started` or `case.completed` event occurred after the cancellation event. Only the active case's `case.failed`, a heartbeat, and `benchmark.cancelled` followed it.
- Ollama logged only the already-active chat completion after cancellation. No subsequent model request appeared through terminal state or during a five-second post-terminal observation.
- During `cancellation_requested`, the query textbox, upload, run, results, and cancel controls remained disabled. Querying re-enabled only after terminal `cancelled`.
- The cancelled run remained inspectable through its run endpoint. It did not replace the latest completed result, which remained `e9279a38-ef5b-416b-9c4c-5791a1c18cae`.

## Deviations and non-blocking warnings

- A cold qwen load outlasted the interactive query request and caused the initial 503 described above; retrying after residency succeeded.
- qwen returned incomplete structured grading data for the exact uploaded evidence, causing conservative abstention despite successful retrieval.
- The benchmark records `request_timeout_seconds: 30.0`, but observed model-backed case latency was not capped at 30 seconds. Full RAG p95 was 228.057 seconds, and the sole runtime-error case took 75.492 seconds. The full run nevertheless completed and persisted all expected results.
- Non-fatal warnings included unauthenticated Hugging Face access, Chroma telemetry capture errors, a LangGraph pending deprecation, and Ollama warnings for unsupported sampling options. No warning prevented the required workflow.

No source-code defect directly blocked this live-validation workflow, so no
implementation fix was required. The delivered outcomes and verification gates
are recorded in the [release report](../release-report.md).
