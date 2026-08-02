# Benchmark Methodology

## Purpose and dataset

The benchmark is a controlled comparison of retrieval and answer workflows,
not a broad claim about model quality. Its canonical dataset is a deterministic
selection of 20 cases from the MultiHopRAG development split. Case identifiers
and the dataset hash are persisted so a result can be tied to the exact
selection that produced it.

## Systems under comparison

Every standard run evaluates seven systems:

| Class | Systems | Output |
| --- | --- | --- |
| Retrieval-only | `dense`, `bm25`, `hybrid` | ranked evidence |
| Fixed single-call RAG | `dense-rag`, `bm25-rag`, `hybrid-rag` | answer and citations |
| Bounded adaptive RAG | `full-rag` | answer, citations, route, retries, and trace |

Retrieval-only baselines do not receive answer-quality metrics. Metrics that
have no eligible cases are reported explicitly rather than converted to zero.

## Preparation and execution

Prepare the fixed selection and benchmark index before running an evaluation:

```bash
uv run python scripts/prepare_multihop_eval.py --index
```

Run all canonical systems against the development selection:

```bash
uv run python -m modules.evaluation \
  --systems all \
  --split development \
  --dataset multihop \
  --model qwen3.5:9b
```

Preparation records the selected cases, expected source documents, benchmark
manifest, and index metadata. Execution requires ready diagnostics, evaluates
each requested case-system pair once, respects configured per-case timeouts,
and writes case evidence incrementally. Evaluation is intentionally serial so
local model contention does not become an uncontrolled comparison variable.

## Completeness classes

A **standard benchmark** contains exactly the 20 canonical cases across all
seven systems: 140 unique case-system results, with the canonical dataset,
split, system set, and benchmark configuration. Duplicate, missing, failed, or
cancelled pairs make a run incomplete.

A subset, smoke run, diagnostic run, noncanonical model or configuration, or
otherwise incomplete execution is labeled **custom evaluation**. Custom runs
remain inspectable but cannot replace the latest valid complete standard result
or support a claim about standard-benchmark improvement.

## Metrics and applicability

Metric observations carry a value, applicability status, and sample count.
Statuses distinguish measured values, metrics not applicable to a system, and
metrics with no eligible cases.

### Retrieval

- recall at 5;
- document recall at 5;
- mean reciprocal rank at 5; and
- normalized discounted cumulative gain at 5.

### Grounding and answer behaviour

- citation precision and gold-evidence citation coverage;
- abstention accuracy and unanswerable abstention recall;
- answerable response rate;
- normalized answer exact match; and
- answer token F1.

### Execution and failures

- termination rate, mean latency, and p95 latency;
- mean model calls and retrieval rounds per query;
- route and strategy accuracy;
- retry precision and recall;
- conflict recall and false-positive rate;
- runtime error count and rate; and
- counts for retrieval misses, citation failures, over-abstention, failed
  abstention, non-termination, route failures, strategy failures, retry
  failures, and conflict failures.

Runtime failures do not receive invented zero-valued quality scores. They are
classified and counted while remaining in the completeness assessment.

## Artifacts and reproducibility

Each run directory under `data/evaluation_results/` contains:

- `run.json` for configuration, status, timestamps, and reproducibility data;
- `summary.json` for aggregate observations and failure counts;
- `cases.jsonl` for per-case outputs, evidence, metrics, and traces; and
- `events.jsonl` for progress and lifecycle events.

Reproducibility metadata includes the code commit when available, dataset hash,
case identifiers, model and embedding identifiers, temperature, prompt and
graph versions, chunking and retrieval settings, context and subquery limits,
retry policy, timeout, and run timestamps. Downloading a run packages these
persisted files; it does not recompute the result.

## Cancellation and failure semantics

Cancellation is cooperative. The runner completes or stops at a safe boundary,
persists the terminal status, and keeps completed case evidence available for
inspection. Failed and cancelled runs are historical evidence, not valid
complete baselines. A retained result must distinguish operational errors from
retrieval, citation, abstention, termination, route, strategy, retry, and
conflict failures.

## Evidence required for an improvement claim

An improvement claim requires:

1. a complete 20-case by seven-system standard result;
2. an explicit complete baseline evaluated under the same canonical contract;
3. the commit, dataset hash, case identifiers, model, and configuration used;
4. metric values with applicability states and sample counts;
5. failure aggregates with inspectable contributing cases; and
6. preserved evidence identifiers, readable source text, and execution traces.

A smoke run, partial run, selected example, or aggregate without provenance is
useful diagnostic evidence but is not proof of improvement.

## Validity limits

The fixed 20-case development selection supports regression testing and
side-by-side workflow comparison. It does not estimate performance over the
full MultiHopRAG distribution, other domains, other model families, or hosted
production traffic. Results remain sensitive to local hardware, Ollama model
builds, embedding state, and runtime contention. Those variables must be
reported when interpreting or comparing local runs.
