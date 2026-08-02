# Release Report

## Completion record

This is the sole comprehensive completion record for the React/FastAPI release
validated on August 1, 2026. It consolidates the final implementation outcomes,
release gates, responsive evidence, and local-runtime qualification. Current
behavioural requirements remain authoritative in
[acceptance criteria](acceptance-criteria.md).

## Delivered outcomes

- The React workspace and FastAPI application services own the complete
  document, query, diagnostics, and benchmark workflows.
- Document import, manifest handling, workspace persistence, downloads, and
  latest-result reopening use one coordinated lifecycle.
- The query graph provides bounded decomposition, hybrid retrieval, evidence
  grading, one retry, citation validation, abstention, and inspectable traces.
- Benchmark preparation and execution cover 20 canonical MultiHopRAG
  development cases across seven systems with explicit metric applicability.
- Benchmark results persist summaries, cases, events, failures, and
  reproducibility metadata; completed runs can be reopened or downloaded.
- Cooperative cancellation, timeouts, operation leases, corrupt-index
  diagnostics, and preparation guidance make long-running operations explicit.
- Summary, Retrieval, Grounding, Execution, Cases, and Failures views expose
  both aggregate and case-level evidence.
- Responsive browser verification covers desktop, tablet, and mobile workspace
  behaviour, including overlays and benchmark inspection.

## Verification evidence

The final local release gate was `./scripts/verify.sh`. On August 1, 2026, it
completed successfully with:

| Gate | Result |
| --- | --- |
| Backend tests | 261 passed |
| Backend coverage | 87.58% |
| Frontend tests | 79 passed |
| Playwright | 5 scenarios covering desktop, tablet, and mobile |
| Frontend production build | passed |
| Ruff, Pyright, and Lanorme | passed |
| Offline runtime diagnostics | passed |
| Locked dependency installation checks | passed |
| Frontend dependency audit | zero known vulnerabilities reported |

The release review also checked README commands and links, obsolete terminology,
tracked generated assets, and whitespace errors. The dependency-audit result is
a dated observation, not a guarantee about future advisories.

## Visual evidence

The tracked screenshots are a responsive inventory of the validated release:

| View | Viewport | Evidence |
| --- | --- | --- |
| Ask documents | 1440 x 1000 | [desktop ask view](assets/dashboard/ask-documents.png) |
| Workspace | 1024 x 768 | [tablet workspace](assets/workspace/tablet-workspace.png) |
| Workspace navigation | 390 x 844 | [mobile sidebar](assets/workspace/mobile-workspace-sidebar.png) |
| Benchmark progress | 1440 x 1000 | [progress view](assets/workspace/benchmark-progress.png) |
| Benchmark results | 1440 x 1000 | [results view](assets/workspace/benchmark-results.png) |
| Case inspection | 1440 x 1000 | [case inspector](assets/workspace/case-inspection.png) |
| Runtime diagnostics | 390 x 844 | [mobile diagnostics](assets/workspace/diagnostics.png) |

These assets are evidence of the recorded layouts, not visual-regression
baselines enforced by the test suite.

## Live local-model evidence

The scoped [live Ollama validation](live-ollama-validation.md) records the
real local environment and a complete 140-pair benchmark execution. It used
Ollama 0.31.2 and `qwen3.5:9b`, retained persisted result evidence, and found no
source-code defect blocking that workflow. The live record complements this
release report; it is not a second repository-wide completion record.

## Qualification

All evidence above was produced locally. It establishes the repository state
and local runtime behaviour observed on August 1, 2026; it does not claim that a
remote CI service re-ran the final documentation commits. Model-dependent
results remain subject to the hardware and runtime limits described in
[benchmark methodology](benchmark-methodology.md).
