# Product Definition

## Purpose

This project is a local, desktop-first Retrieval-Augmented Generation workbench
for importing PDF and TXT documents, asking questions with cited evidence,
comparing retrieval and answer-generation systems, and inspecting how each
result was produced. The product exposes operational state, evaluation
evidence, and deliberate limits instead of presenting only a polished answer
surface.

## Users and workflows

The primary user is a technical evaluator, engineer, or reviewer who needs to
understand both what a RAG system returns and how it reached that result. The
core workflows are:

1. Import source documents and observe indexing progress.
2. Ask a question, inspect the cited answer, and open the supporting evidence.
3. Prepare and run the canonical benchmark or a clearly labeled custom run.
4. Compare systems through summary, retrieval, grounding, execution, case, and
   failure views.
5. Reopen completed results and export their persisted artifacts for review.

## Workspace surfaces

The product uses one persistent application shell rather than disconnected demo
pages. Its primary surfaces are the workspace dashboard, document library, ask
view, and benchmark view. The navigation rail and document sidebar maintain
context while evidence inspectors and operation details open as overlays.
Overlays trap focus while open and restore it to the invoking control when
closed.

## Evidence inspection

Answers and benchmark cases make source material inspectable. Evidence cards
include the chunk identifier, document identifier, filename, page when
available, and readable source text. Benchmark cases also retain execution
metadata and traces so an aggregate score can be followed back to the case that
produced it.

## Readiness states

The interface reports distinct, actionable states for:

- Ollama unavailable;
- a missing index manifest;
- a corrupt or incomplete index;
- missing source or evaluation assets; and
- an operation already in progress.

Each state explains the next supported action, such as starting Ollama or
running the documented index-preparation command.

## Coordinated operations

Document ingestion and benchmark execution share a single operation lease, so
long-running writes cannot collide silently. Cancellation is cooperative:
completed cases remain inspectable, cancelled or failed runs retain their
status, and partial work cannot replace the latest valid complete benchmark.

## Benchmark experience

The benchmark compares seven systems: three retrieval-only baselines, three
fixed single-call RAG systems, and the bounded adaptive full-RAG system. The
results workspace separates Summary, Retrieval, Grounding, Execution, Cases,
and Failures so metric applicability and failure evidence stay visible. A
standard result must satisfy the canonical completeness contract; subsets,
smoke runs, and noncanonical configurations are labeled as custom evaluation.

## Persistence

Completed benchmark runs are stored on disk with summary, case, event, and
reproducibility metadata. The application can reopen results after a server
restart and download a run as a ZIP archive. The latest-result pointer advances
only for a valid complete canonical run.

## Experience and accessibility principles

- Desktop, tablet, and mobile layouts remain contained without horizontal
  overflow; dense tools become drawers or bottom sheets on smaller viewports.
- Answers, cases, and failures expose evidence rather than relying only on
  aggregate scores.
- Semantic landmarks, keyboard operation, visible focus, labeled controls,
  status announcements, reduced-motion support, and practical touch targets
  support a WCAG 2.2 AA-aligned experience.
- A restrained indigo-and-slate visual system keeps the conversation dominant
  while preserving dense technical information.

## Deliberate boundaries

- The application is local-first and single-user; it has no authentication or
  multi-tenant isolation.
- Ollama provides generation and embeddings, so model availability and hardware
  determine runtime behaviour.
- Workspace and benchmark vectors are stored in separate Chroma locations and
  are prepared through separate workflows.
- The UI does not download the MultiHopRAG dataset or build its evaluation
  index; the documented preparation command does that explicitly.
- The canonical benchmark is fixed at 20 MultiHopRAG development cases across
  seven systems. It is a controlled regression comparison, not a claim of broad
  model quality.
- PDF and TXT are the supported import formats.
- Diagnostics report missing or corrupt assets; the application does not
  silently repair them.
