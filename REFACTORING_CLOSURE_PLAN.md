# React/FastAPI Workspace Closure Remaining Implementation Plan

> **Execution mode:** Implement inline in the primary Codex task. Do not use sub-agents. Track each checkbox in this file and stop on a genuine blocker or a failing baseline that cannot be attributed to the current task.

**Goal:** Finish the existing React/FastAPI workspace without another architectural rewrite, leaving one correct, persistent, inspectable, reproducible seven-system Full RAG Benchmark workflow.

**Architecture:** Keep the React single-workspace UI, FastAPI routes, application services, operation coordinator, `BenchmarkManager`, `FullRagBenchmarkExecutor`, and seven-system evaluator. Complete the remaining persistence, diagnostics, metrics, cancellation, presentation, evidence, reproducibility, onboarding, documentation, browser, screenshot, and live-Ollama work inside those boundaries.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, LangChain, LangGraph, Ollama, React 19, TypeScript, Vite, Vitest, pytest, and Playwright.

## Current Checkpoint

- Branch: `react-fastapi-workspace`
- Completed Task 1: benchmark format/version terminology removed.
  - `7b91efc refactor: remove benchmark format terminology`
  - `22f2124 fix: remove remaining benchmark format terminology`
- Completed Task 2: canonical completeness is validated from the exact 20 cases and 140 unique case-system results.
  - `cd150be fix: validate complete benchmark contents`
- Start with Task 3. No Task 3 production or test changes have been made.
- Baseline before Task 1:
  - Backend: `240 passed, 2 skipped`
  - Frontend: `73 passed`

## Global Constraints

- Maintain one current benchmark artifact structure.
- Do not add compatibility readers, migration code, format selectors, format-version fields, or numbered format terminology.
- Preserve the current React/FastAPI architecture and one continuous workspace route.
- Dense RAG, BM25 RAG, and Hybrid RAG may differ only in retrieval.
- Routing, strategy, retry, and explicit conflict metrics apply only to Full RAG.
- Cancellation is cooperative: the active model request may finish or time out, but no later case, system, or model request may start.
- Cancelled and failed runs remain inspectable but never become the latest completed result.
- Do not call the branch release-ready until offline tests, Playwright checks, and the real Ollama workflow pass.
- Follow TDD for every behavior change: add the test, observe the intended failure, implement the minimum change, and rerun the focused test.
- Keep each task in a separate verified commit. Do not stage unrelated files.
- Before each commit run `git diff --check` and inspect `git diff --stat`.

## Artifact Contract Used by All Remaining Tasks

Every embedded run directory must use exactly these public files:

```text
<benchmark_results_dir>/<run_id>/
├── run.json
├── summary.json
├── cases.jsonl
└── events.jsonl
```

A completed canonical artifact must expose:

```json
{
  "benchmark_name": "full_rag_benchmark",
  "result_kind": "standard_benchmark",
  "case_ids": ["the exact 20 canonical development case IDs"],
  "expected_result_count": 140,
  "completed_result_count": 140
}
```

Subset, diagnostic, failed, cancelled, incomplete, duplicate, or noncanonical runs use `custom_evaluation`.

---

## Task 3: Unify Embedded Persistence, Diagnostics, and Downloads

**Files**

- Modify `modules/application/benchmark_manager.py`
- Modify `modules/application/workspace_service.py`
- Modify `modules/bootstrap.py`
- Modify `modules/api/dependencies.py` only if the protocol needs the new read method
- Test `tests/application/test_benchmark_manager.py`
- Test `tests/application/test_workspace_service.py`
- Test `tests/test_bootstrap.py`

**Interfaces**

```python
class BenchmarkManager:
    async def has_completed_benchmark(self) -> bool:
        """Return True only when a valid completed embedded artifact exists."""
```

```python
CompletedBenchmarkProbe = Callable[[], Awaitable[bool]]
```

- [ ] Add failing manager tests proving:
  - queued, running, failed, and cancelled runs return `False`;
  - a completed valid run returns `True`;
  - the newest completed run survives manager restart;
  - corrupt run directories and corrupt companion files are skipped;
  - `summary.json` is generated without a test manually creating it;
  - every archive contains exactly `run.json`, `summary.json`, `cases.jsonl`, and `events.jsonl`;
  - a symlinked or missing required artifact makes the download fail safely instead of returning an incomplete archive.

- [ ] Run the focused red tests:

```powershell
uv run pytest tests/application/test_benchmark_manager.py -k "completed_benchmark or download or restart or corrupt" -q --no-cov
```

Expected: failures for the missing method, missing generated summary, or incomplete archive behavior.

- [ ] Make `BenchmarkManager` write `summary.json` from the authoritative `BenchmarkRun` plus persisted case records. Update it whenever the durable run snapshot changes, including terminal failed/cancelled states, so every stored run remains downloadable and inspectable.

- [ ] Validate `summary.json` when reopening a run. The stored `run_id`, status, result counts, case IDs, sections, and failure aggregates must agree with `run.json` and `cases.jsonl`.

- [ ] Implement `has_completed_benchmark()` by reusing the same safe completed-run scan as `latest_benchmark()`. Do not catch errors other than “no valid completed run.”

- [ ] Remove `WorkspaceService._latest_evaluation_exists()` and all scanning of `evals/results/multihop`.

- [ ] Split evaluation diagnostics into:
  - a synchronous dataset/index readiness probe;
  - the injected async `CompletedBenchmarkProbe`.

- [ ] Compose dependencies in `modules/bootstrap.py` so `WorkspaceService` receives `BenchmarkManager.has_completed_benchmark`.

- [ ] Run:

```powershell
uv run pytest tests/application/test_benchmark_manager.py tests/application/test_workspace_service.py tests/test_bootstrap.py -q --no-cov
uv run ruff check modules/application/benchmark_manager.py modules/application/workspace_service.py modules/bootstrap.py tests/application/test_benchmark_manager.py tests/application/test_workspace_service.py tests/test_bootstrap.py
uv run pyright modules/application/benchmark_manager.py modules/application/workspace_service.py modules/bootstrap.py
```

- [ ] Commit:

```powershell
git add modules/application/benchmark_manager.py modules/application/workspace_service.py modules/bootstrap.py modules/api/dependencies.py tests/application/test_benchmark_manager.py tests/application/test_workspace_service.py tests/test_bootstrap.py
git commit -m "fix: unify benchmark persistence and diagnostics"
```

---

## Task 4: Reopen the Latest Stored Result

**Files**

- Modify `frontend/src/benchmark/useBenchmark.ts`
- Modify `frontend/src/workspace/Sidebar.tsx`
- Modify `frontend/src/App.tsx`
- Test `frontend/src/benchmark/useBenchmark.test.tsx`
- Test `frontend/src/App.test.tsx`

**Controller contract**

```ts
type BenchmarkController = {
  loadLatest(): Promise<boolean>;
  latestLoading: boolean;
  latestError: string | null;
};
```

- [ ] Add failing controller tests for success, 404/no-result, server failure, loading reset, and retaining the loaded run for case inspection/download.
- [ ] Add failing app tests proving “View latest results” opens the result dialog, shows an explicit empty state, is disabled during another workspace operation, and works after a fresh app mount.
- [ ] Implement `loadLatest()` using the existing latest-benchmark API client. Return `true` only when a result was loaded.
- [ ] Add **View latest results** directly below **Run benchmark**.
- [ ] On success store the run and open the existing result overlay. Do not create a second results component.
- [ ] Use explicit copy for the empty state: `No completed Full RAG Benchmark result is available.`
- [ ] Run:

```powershell
npm --prefix frontend exec -- vitest run frontend/src/benchmark/useBenchmark.test.tsx frontend/src/App.test.tsx
npm --prefix frontend run typecheck
```

- [ ] Commit:

```powershell
git add frontend/src/benchmark/useBenchmark.ts frontend/src/workspace/Sidebar.tsx frontend/src/App.tsx frontend/src/benchmark/useBenchmark.test.tsx frontend/src/App.test.tsx
git commit -m "feat: reopen latest benchmark results"
```

---

## Task 5: Correct Metric Semantics and Failure Aggregation

**Files**

- Modify `modules/evaluation_metrics.py`
- Modify `modules/application/full_rag_benchmark.py`
- Modify `modules/application/models.py` if aggregate failure counts need a response type
- Test `tests/test_evaluation.py`
- Test `tests/application/test_full_rag_benchmark.py`

**Required semantics**

```python
citation_precision(cited_chunk_ids, relevant_chunk_ids)
```

```text
runtime_error_count
runtime_error_rate
```

- [ ] Add failing tests proving citation precision uses expected relevant evidence, while nonexistent/unretrieved citation labels remain validation failures.
- [ ] Add failing tests proving runtime errors:
  - count as failed responses;
  - do not improve abstention accuracy;
  - do not improve answerable response rate;
  - receive zero answer exact match and token F1 when an answer is expected;
  - remain in detailed failures.
- [ ] Add failing denominator tests proving missing Full RAG route/strategy outputs count as incorrect across all eligible cases.
- [ ] Prove fixed RAG workflow observations are `not_applicable`.
- [ ] Prove retrieval-only answer and workflow observations are `not_applicable`.
- [ ] Add aggregate classification counts for runtime error, retrieval miss, citation failure, over-abstention, failed abstention, non-termination, route, strategy, retry, and conflict failures.
- [ ] Implement only the metric/failure changes needed by those tests.
- [ ] Run:

```powershell
uv run pytest tests/test_evaluation.py tests/application/test_full_rag_benchmark.py -q --no-cov
uv run ruff check modules/evaluation_metrics.py modules/application/full_rag_benchmark.py modules/application/models.py tests/test_evaluation.py tests/application/test_full_rag_benchmark.py
uv run pyright modules/evaluation_metrics.py modules/application/full_rag_benchmark.py modules/application/models.py
```

- [ ] Commit:

```powershell
git add modules/evaluation_metrics.py modules/application/full_rag_benchmark.py modules/application/models.py tests/test_evaluation.py tests/application/test_full_rag_benchmark.py
git commit -m "fix: correct benchmark metric semantics"
```

---

## Task 6: Standardize Timeout and Cooperative Cancellation

**Files**

- Modify `modules/evaluation.py`
- Modify `modules/application/full_rag_benchmark.py`
- Modify `modules/application/benchmark_manager.py`
- Modify `modules/application/models.py`
- Modify `frontend/src/benchmark/BenchmarkProgress.tsx`
- Modify related backend/frontend tests

**Required UI copy**

```text
Cancellation requested. The active model request may finish; no additional cases will start.
```

- [ ] Add backend failing tests for cancellation:
  - before the first case;
  - while one case is active;
  - before and after every model invocation;
  - before the next system;
  - no second case starts;
  - no second model call starts;
  - completed case records remain persisted;
  - cancelled runs have no complete-run aggregate sections;
  - cancelled runs are not returned by `latest_benchmark()`.
- [ ] Add frontend failing tests proving the composer, uploads, deletion, and model loading stay disabled until terminal cancellation.
- [ ] Remove the nested daemon-thread deadline fallback from `modules/evaluation.py`.
- [ ] Use the single canonical request timeout for Dense RAG, BM25 RAG, Hybrid RAG, and Full RAG model calls.
- [ ] Keep the operation-coordinator lease until the active request exits or times out.
- [ ] Render the required cancellation copy verbatim.
- [ ] Run:

```powershell
uv run pytest tests/test_evaluation.py tests/application/test_full_rag_benchmark.py tests/application/test_benchmark_manager.py -q --no-cov
npm --prefix frontend exec -- vitest run
npm --prefix frontend run typecheck
```

- [ ] Commit:

```powershell
git add modules/evaluation.py modules/application/full_rag_benchmark.py modules/application/benchmark_manager.py modules/application/models.py frontend/src/benchmark/BenchmarkProgress.tsx tests frontend/src
git commit -m "fix: standardize benchmark cancellation and timeouts"
```

---

## Task 7: Complete Result Presentation and Case Listing

**Files**

- Modify `frontend/src/benchmark/BenchmarkResults.tsx`
- Modify `frontend/src/api/types.ts`
- Modify `frontend/src/api/client.ts`
- Modify `modules/api/routes.py`
- Modify `modules/api/dependencies.py`
- Modify `modules/application/benchmark_manager.py`
- Modify `modules/application/models.py`
- Modify corresponding backend/frontend tests

**Endpoint**

```http
GET /api/benchmarks/{run_id}/cases
```

**Tabs**

```text
Summary
Retrieval
Grounding
Execution
Cases
Failures
```

- [ ] Add failing API/manager tests for lightweight summaries of every successful, expectation-failure, and runtime-failure case-system result.
- [ ] Add failing frontend tests for all six tabs, stable section-ID selection, section `system_ids`, successful/failed case opening, detailed failure rows, aggregate counts, and focus restoration.
- [ ] Add a typed case-summary model and manager method that reads `cases.jsonl` once and returns lightweight rows.
- [ ] Add the route and client method.
- [ ] Remove regular-expression section/metric classification. Use exact stable IDs.
- [ ] Render only systems in each section’s `system_ids`; retrieval-only systems must not appear as “not reported” in answer sections.
- [ ] Summary must show principal retrieval metrics, answer token F1, citation precision, abstention accuracy, P95 latency, runtime-error count, and runtime-error rate.
- [ ] Keep the existing case drawer nested inside the results dialog.
- [ ] Run:

```powershell
uv run pytest tests/application/test_benchmark_manager.py tests/api -q --no-cov
npm --prefix frontend exec -- vitest run
npm --prefix frontend run typecheck
```

- [ ] Commit:

```powershell
git add modules/application modules/api frontend/src tests/application tests/api
git commit -m "feat: complete benchmark result inspection"
```

---

## Task 8: Preserve Readable Evidence

**Files**

- Modify `modules/evaluation.py`
- Modify `modules/application/full_rag_benchmark.py`
- Modify `frontend/src/benchmark/BenchmarkResults.tsx`
- Modify related tests

**Shared mapper rule**

```python
text = hit.get("excerpt") or hit.get("content") or ""
excerpt = " ".join(str(text).split())[:300]
```

- [ ] Add failing tests for all seven systems proving every evidence item contains chunk ID, document ID, filename, page, and a normalized readable excerpt.
- [ ] Implement one mapper reused by retrieval, fixed RAG, and Full RAG.
- [ ] Render structured evidence cards with filename, page, chunk ID, and excerpt.
- [ ] Keep raw evidence only inside an expandable diagnostic control.
- [ ] Run backend and frontend focused tests, Ruff, Pyright, and typecheck.
- [ ] Commit:

```powershell
git add modules/evaluation.py modules/application/full_rag_benchmark.py frontend/src/benchmark/BenchmarkResults.tsx tests frontend/src
git commit -m "fix: preserve readable benchmark evidence"
```

---

## Task 9: Complete Reproducibility Metadata

**Files**

- Modify `modules/application/full_rag_benchmark.py`
- Modify `modules/application/models.py`
- Modify `modules/evaluation.py`
- Modify related tests

**Record exactly**

```text
git commit
dataset identifier
dataset content hash
exact case IDs
chat model
embedding model
temperature
fixed-RAG prompt identifier
Full RAG graph configuration
chunk size and overlap
retrieval limit
semantic and sparse candidate limits
maximum context chunks
retry and subquery limits
request timeout
benchmark start and completion timestamps
```

- [ ] Add failing serialization tests for every field above and for preservation across restart/download.
- [ ] Compute the dataset hash from the actual benchmark case/source content, not filenames or timestamps.
- [ ] Record graph configuration as explicit JSON values rather than a repr string.
- [ ] Remove `case_limit` unless it is derived from the exact selected canonical 20 IDs; prefer `case_ids` and result counts.
- [ ] Run focused backend tests, Ruff, and Pyright.
- [ ] Commit:

```powershell
git add modules/application/full_rag_benchmark.py modules/application/models.py modules/evaluation.py tests
git commit -m "feat: record benchmark reproducibility metadata"
```

---

## Task 10: Report Corrupt Index State Honestly

**Files**

- Modify `modules/application/workspace_service.py`
- Modify `modules/application/errors.py`
- Test `tests/application/test_workspace_service.py`
- Test `tests/api/test_api_errors.py`

- [ ] Add failing tests for:
  - missing manifest → empty corpus;
  - malformed/unreadable existing manifest → diagnostic error;
  - missing Chroma collection → actionable error;
  - missing indexed chunks → actionable error;
  - incompatible index settings → actionable error.
- [ ] Stop swallowing manifest parse/read failures.
- [ ] Preserve the missing-manifest empty-corpus behavior.
- [ ] Map each corrupt state to public, actionable diagnostics without exposing filesystem paths.
- [ ] Run:

```powershell
uv run pytest tests/application/test_workspace_service.py tests/api/test_api_errors.py -q --no-cov
```

- [ ] Commit:

```powershell
git add modules/application/workspace_service.py modules/application/errors.py tests/application/test_workspace_service.py tests/api/test_api_errors.py
git commit -m "fix: report corrupt index diagnostics"
```

---

## Task 11: Add Benchmark Preparation Onboarding

**Files**

- Modify `README.md`
- Modify `modules/application/workspace_service.py`
- Modify diagnostics UI/tests

**Required preparation command**

```bash
uv run python scripts/prepare_multihop_eval.py --index
```

- [ ] Add failing diagnostics tests for missing case/source files, missing/population-empty benchmark manifest, and missing Chroma benchmark index.
- [ ] Require both benchmark files and a populated benchmark manifest/collection before enabling **Run benchmark**.
- [ ] Show the exact preparation command in the blocked diagnostic message.
- [ ] Document that user documents and benchmark data are separate, preparation does not alter uploaded documents, and preparation is required once unless benchmark indexing settings change.
- [ ] Do not add in-app dataset downloading/indexing.
- [ ] Run backend diagnostics tests, frontend diagnostics tests, and typecheck.
- [ ] Commit:

```powershell
git add README.md modules/application/workspace_service.py frontend/src tests
git commit -m "feat: add benchmark preparation onboarding"
```

---

## Task 12: Remove the Obsolete Benchmark Adapter

**Files**

- Delete `modules/application/unavailable_benchmarks.py`
- Delete `tests/application/test_unavailable_benchmarks.py`
- Modify imports/protocols only where the search proves they remain

- [ ] Confirm only the obsolete adapter/tests remain:

```powershell
rg -n "UnavailableBenchmarkManager|unavailable_benchmarks" .
```

- [ ] Delete both files with a recoverable, explicit patch and remove remaining accommodations.
- [ ] Run backend tests that compose the real manager and API.
- [ ] Confirm the search returns no matches.
- [ ] Commit:

```powershell
git add -A modules/application/unavailable_benchmarks.py tests/application/test_unavailable_benchmarks.py modules tests
git commit -m "refactor: remove unavailable benchmark adapter"
```

---

## Task 13: Rewrite Documentation for the Current Application

**Files**

- Modify `README.md`
- Modify `ACCEPTANCE_CRITERIA.md`
- Modify `docs/ui-refactor-parity.md` or delete it if it has no current maintenance value
- Modify other tracked architecture/interface docs found by the terminology search

- [ ] Document one React/FastAPI workspace and one continuous route.
- [ ] Document sidebar, conversation, inspector, and overlays.
- [ ] Document seven systems and the Retrieval, Grounding, Execution, Cases, and Failures views.
- [ ] Document persistence/reopening, cooperative cancellation, preparation, current endpoints, and quality checks.
- [ ] Remove Gradio, multipage/route-based UI, four-system, old-format, migration/compatibility, and local Windows-path language.
- [ ] Run:

```powershell
rg -n -i "gradio|schema[_ -]?version|schema[ -]?v[0-9]|legacy schema|UnavailableBenchmarkManager|C:\\\\Users\\\\" README.md modules tests docs ACCEPTANCE_CRITERIA.md
```

Expected: no obsolete UI/benchmark-format/adapter/local-path references. Unrelated vector-index manifest fields must be evaluated separately and retained when they are not benchmark-format terminology.

- [ ] Commit:

```powershell
git add README.md ACCEPTANCE_CRITERIA.md docs
git commit -m "docs: describe the current React FastAPI workspace"
```

---

## Task 14: Add Browser and Responsive Verification

**Files**

- Modify `frontend/package.json`
- Create or modify `frontend/playwright.config.ts`
- Create or modify `frontend/e2e/workspace.spec.ts`
- Modify `.github/workflows/quality.yml`
- Modify `scripts/verify.sh`

**Viewports**

```text
Desktop: 1440 × 1000
Tablet: 1024 × 768
Mobile: 390 × 844
```

- [ ] Add Playwright fixtures that use deterministic API responses and the production Vite/FastAPI build.
- [ ] Cover production load, mobile sidebar, Escape/focus restoration, inspector collapse, document details/delete, benchmark progress, cancellation-requested copy, disabled composer, stored-result reopening, all result tabs, successful/failed cases, download, diagnostics, and overflow.
- [ ] Add `test:e2e` to `frontend/package.json`.
- [ ] Add Playwright installation/execution to CI and `scripts/verify.sh`.
- [ ] Run:

```powershell
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

- [ ] Commit:

```powershell
git add frontend/package.json frontend/playwright.config.ts frontend/e2e/workspace.spec.ts .github/workflows/quality.yml scripts/verify.sh
git commit -m "test: add responsive workspace browser coverage"
```

---

## Task 15: Replace Obsolete Screenshots

**Files**

- Replace `docs/assets/dashboard/ask-documents.png`
- Add supporting screenshots under `docs/assets/workspace/`
- Modify `README.md` and supporting docs

**Required images**

```text
desktop workspace
tablet layout
mobile workspace/sidebar
benchmark progress
benchmark results
case drawer
diagnostics
```

- [ ] Build and serve the production application from the verified code state.
- [ ] Capture deterministic states at the Task 14 viewports.
- [ ] Confirm no secrets, usernames, absolute paths, or private document content appear.
- [ ] Keep only the strongest workspace image in README; place supporting images in docs.
- [ ] Verify image links and dimensions.
- [ ] Commit:

```powershell
git add docs/assets README.md docs
git commit -m "docs: replace workspace screenshots"
```

---

## Task 16: Run and Record the Live Ollama Workflow

**Preparation**

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
uv run python scripts/prepare_multihop_eval.py --index
npm --prefix frontend ci
npm --prefix frontend run build
```

- [ ] Start the production FastAPI application and open the served React build.
- [ ] Load `qwen3.5:9b`.
- [ ] Upload and index a real PDF or TXT document.
- [ ] Ask a grounded question and inspect sources and trace.
- [ ] Run all seven systems over the exact 20 canonical cases.
- [ ] Confirm exactly 140 persisted case-system rows and all aggregate sections.
- [ ] Inspect one successful case and one expectation failure.
- [ ] Inspect one runtime failure if the run produces one; record “none produced” otherwise.
- [ ] Download and inspect all four artifact files.
- [ ] Restart the application and reopen the completed result.
- [ ] Confirm diagnostics see that same completed result.
- [ ] Start another run, request cancellation during a model call, and verify:
  - the active request finishes or times out;
  - no additional case/model call starts;
  - querying stays disabled until terminal cancellation;
  - the cancelled run remains inspectable;
  - the cancelled run is not latest completed.
- [ ] Record in a tracked validation document:
  - total duration;
  - per-system duration;
  - runtime-error count;
  - cancellation timing/behavior;
  - Ollama version;
  - model versions/digests;
  - tested git commit.
- [ ] Commit the validation record:

```powershell
git add docs
git commit -m "docs: record live Ollama validation"
```

If Ollama, model downloads, or the full benchmark cannot run, stop and report the exact blocker. Do not mark Task 16 complete with simulated evidence.

---

## Task 17: Final Release Verification

- [ ] Run the full project gate:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/verify.sh
```

The script must cover locked Python installation, locked frontend installation, Vitest, TypeScript, production build, dependency audit, Ruff, Pyright, Lanorme, offline pytest, Playwright, and offline diagnostics.

- [ ] Run the obsolete-term gate:

```powershell
rg -n -i "gradio|schema[_ -]?version|schema[ -]?v[0-9]|legacy schema|UnavailableBenchmarkManager" README.md modules tests docs ACCEPTANCE_CRITERIA.md
```

- [ ] Inspect repository state:

```powershell
git status --short
git log --oneline --decorate -20
```

- [ ] Confirm:
  - React/FastAPI is the only interface;
  - canonical completeness uses actual contents;
  - runtime/missing outputs cannot improve metrics;
  - cooperative cancellation matches the UI copy;
  - completed results persist/reopen and diagnostics/downloads use the same manager artifacts;
  - runtime failure aggregates appear in Summary;
  - evidence is readable for all systems;
  - every result is inspectable by case;
  - preparation is documented and blocks execution until ready;
  - obsolete adapter/docs are gone;
  - browser checks and screenshots are current;
  - Task 16 live evidence exists.

- [ ] Push only when explicitly authorized. A successful GitHub workflow for the final commit is external release evidence and cannot be claimed from local checks alone.

- [ ] Do not create an empty “verification” commit. If the final gate requires fixes, apply them surgically, rerun the affected check and the full gate, then commit the fixes.

## Completion Report Template

Use this exact structure when the remaining plan is finished:

```text
Implemented:
- persistence/diagnostics/download unification
- latest-result reopening
- corrected metrics and failure aggregation
- cooperative cancellation and shared timeout
- complete results/case/failure presentation
- readable evidence and reproducibility metadata
- corrupt-index diagnostics and benchmark onboarding
- adapter/documentation/browser/screenshot cleanup

Validation:
- backend: <exact passed/skipped counts>
- frontend: <exact passed counts>
- typecheck/build/lint/static analysis: <exact results>
- Playwright: <exact passed counts and viewports>
- live Ollama: <commit, versions, duration, result count, cancellation outcome>

Unresolved:
- <only genuine remaining blockers; write “none” when there are none>
```
