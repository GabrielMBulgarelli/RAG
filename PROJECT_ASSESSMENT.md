# RAG Project Assessment

## Overall status

The project is a solid local RAG evaluation workbench and functional MVP. Its
planned architecture is largely present: deterministic ingestion, dense, BM25,
and hybrid retrieval, agentic routing, retries, abstention, citations, tracing,
evaluation, and a Gradio interface.

It is not yet a high-quality RAG application. The principal limitation is now
measured effectiveness rather than the absence of major architectural
components.

## Current quality evidence

The held-out MultiHopRAG run produced the following results:

| Metric | Agentic RAG | Best baseline | Project target |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.375 | Hybrid: 0.469 | >= 0.85 |
| MRR@5 | 0.292 | Hybrid: 0.588 | Not defined |
| Citation precision | 0.908 | Not applicable | 1.00 |
| Gold-evidence citation coverage | 0.425 | Not applicable | Not defined |
| Abstention accuracy | 0.700 | Not applicable | >= 0.85 |
| Retry precision | 0.400 | Not applicable | Not defined |
| Answer token F1 | 0.0086 | Not applicable | Not defined |
| Mean latency | 136.8 seconds | Hybrid: 4.47 seconds | Not defined |

The most important result is that the agentic system currently ranks evidence
worse than the simpler hybrid retriever while taking approximately 30 times
longer. The architecture works, but the agentic workflow has not yet
demonstrated a quality advantage that justifies its cost.

The source results are stored in
`evals/results/multihop/20260712T093825Z-test/summary.json`.

## Missing capabilities and quality improvements

### 1. Retrieval quality

Retrieval quality is the primary blocker. Recall@5 is less than half the
required threshold. The next retrieval iteration should evaluate:

- A second-stage reranker after hybrid retrieval.
- Parent-document or hierarchical retrieval.
- Adaptive candidate counts instead of always limiting final retrieval to five
  chunks.
- Chunk-size and overlap experiments by document type.
- Better indexing of titles, sources, dates, and document metadata.
- Query-decomposition checks to ensure every subquery contributes evidence
  required by the original question.
- Multi-query fusion that prevents weak rewritten queries from displacing
  strong original-query results.
- Retrieval and relevance thresholds calibrated on development data.

The agentic retrieval path should not be treated as an improvement until it
beats the hybrid baseline or provides a measurable answer-quality advantage.

### 2. Answer-generation correctness

The agentic answer token F1 score of 0.0086 is effectively zero. Recorded
answers are often verbose, heavily hedged, and fail to provide the requested
entity or comparison directly.

Two issues must be distinguished:

1. Verify that the evaluator compares the intended answer fields and performs
   appropriate normalization.
2. If the evaluator is correct, constrain generation to a structured response
   containing a direct answer, a short explanation, citations, and an explicit
   abstention status.

Retrieval-only baselines naturally have no meaningful answer-generation score,
but the agentic system must demonstrate useful answer correctness.

### 3. Evidence and citation enforcement

Citation precision is high but incomplete, and gold-evidence coverage remains
low. Some held-out cases contain `invalid_citation` and
`citation_coverage_miss` failure labels.

The system still needs:

- Claim-by-claim support validation.
- Rejection or regeneration when a cited chunk does not support its associated
  claim.
- Coverage checks for every material claim in an answer.
- A strict distinction between retrieved, relevant, and cited evidence.
- Clickable citations that open the exact source passage in the interface.

### 4. Abstention and retry calibration

Abstention accuracy is 0.70, below the required 0.85. Retry recall is perfect,
but retry precision is only 0.40, indicating that retries are frequently used
for cases where they do not help.

The evidence-confidence logic must distinguish among:

- The answer is absent from the indexed corpus.
- Retrieval failed and another retrieval strategy may help.
- Retrieved evidence is conflicting.
- Sufficient evidence exists but the language model remains uncertain.

Retries should be triggered by a specific recoverable retrieval failure rather
than by general model uncertainty.

### 5. Interactive performance

A mean latency of approximately 137 seconds is not suitable for an interactive
application. Improvements should include:

- Keeping simple questions on a fast, non-agentic path.
- Reducing the current average of 4.5 language-model calls per query.
- Running independent retrieval subqueries concurrently.
- Caching query rewrites, embeddings, and retrieval results.
- Using a smaller model for routing and query rewriting.
- Warming the Ollama model during application startup.
- Streaming progress and partial output to the interface.
- Adding per-stage timeouts and user cancellation.
- Avoiding retries when the initial evidence is already sufficient.

A practical initial objective is under 10 seconds for simple retrieval and
under 30 seconds for genuinely complex local queries.

### 6. Operational and document-ingestion polish

The remaining local-application gaps include:

- Implementing `modules.diagnostics`; `scripts/verify.sh` currently skips the
  check when the module is absent.
- Eliminating or correctly disabling the broken Chroma telemetry events.
- Background ingestion with visible progress, cancellation, and recovery.
- Index-integrity checks plus explicit backup, restore, and rebuild commands.
- Persistent structured traces and a failed-query inspection view.
- Clear Ollama, model, embedding, index, and evaluation health indicators.

Document support should eventually include scanned-PDF OCR, Markdown, HTML,
DOCX, tables, and image-aware documents in addition to the currently supported
formats.

## Recommended completion criteria

The local application can be considered high quality when:

1. Recall@5 reaches at least 0.85.
2. Agentic retrieval beats the hybrid baseline or demonstrates a clear and
   repeatable answer-quality advantage.
3. Citation precision reaches 1.00 with materially higher evidence coverage.
4. Abstention accuracy reaches at least 0.85.
5. Answer F1 and direct-answer correctness become meaningful.
6. Interactive latency is reduced by several times.
7. Diagnostics, ingestion recovery, and index-management workflows are
   complete.

The immediate priority should therefore be retrieval and answer-quality
diagnosis, followed by citation enforcement and latency reduction. Additional
interface work should follow once the core RAG quality is demonstrably sound.
