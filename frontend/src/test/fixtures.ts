import { vi } from "vitest";

import type { WorkspaceApi } from "../api/client";
import type {
  BenchmarkCaseDetail,
  BenchmarkRun,
  BenchmarkStartResponse,
  DiagnosticsSnapshot,
  DocumentList,
  QueryResponse,
  RuntimeSnapshot,
} from "../api/types";

export const benchmarkRunId = "4cbdbcb9-5a57-4514-a392-2dce907456d5";

const benchmarkSystems = [
  { id: "dense", label: "Dense" },
  { id: "bm25", label: "BM25" },
  { id: "hybrid", label: "Hybrid" },
  { id: "dense-rag", label: "Dense RAG" },
  { id: "bm25-rag", label: "BM25 RAG" },
  { id: "hybrid-rag", label: "Hybrid RAG" },
  { id: "full-rag", label: "Full RAG" },
];

export const benchmarkStart: BenchmarkStartResponse = {
  run_id: benchmarkRunId,
  status: "queued",
  links: {
    run: `/api/benchmarks/${benchmarkRunId}`,
    events: `/api/benchmarks/${benchmarkRunId}/events`,
    download: `/api/benchmarks/${benchmarkRunId}/download`,
  },
};

export const benchmarkRun: BenchmarkRun = {
  ...benchmarkStart,
  status: "completed",
  progress: {
    completed_cases: 20,
    total_cases: 20,
    current_system: "full-rag",
    current_system_index: 6,
    total_systems: 7,
    current_case_id: "case / 1",
    current_case_index: 19,
  },
  metadata: {
    dataset: "MultiHopRAG",
    split: "development",
    systems: benchmarkSystems,
    chat_model: "qwen3:8b",
    embedding_model: "nomic-embed-text",
    started_at: "2026-07-29T15:30:00Z",
    completed_at: "2026-07-29T15:34:00Z",
    reproducibility: {
      seed: 42,
      package_versions: { rag: "0.1.0" },
    },
  },
  sections: [{
    id: "retrieval",
    title: "Retrieval",
    system_ids: ["dense", "bm25", "hybrid"],
    detail: "Retrieval quality at five.",
    metrics: [{
      name: "recall_at_5",
      label: "Recall at 5",
      observations: [
        { system: "dense", value: 0, status: "measured", sample_count: 20, note: null },
        {
          system: "bm25",
          value: null,
          status: "not_applicable",
          sample_count: 0,
          note: "Not scored for this fixture.",
        },
        {
          system: "hybrid",
          value: null,
          status: "no_eligible_cases",
          sample_count: 0,
          note: null,
        },
      ],
    }],
  }, {
    id: "grounding",
    title: "Grounding",
    system_ids: ["dense-rag", "bm25-rag", "hybrid-rag", "full-rag"],
    detail: null,
    metrics: [{
      name: "answer_token_f1",
      label: "Answer token F1",
      observations: benchmarkSystems.map((system, index) => ({
        system: system.id,
        value: index < 3 ? null : 0.71 + index / 100,
        status: index < 3 ? "not_applicable" as const : "measured" as const,
        sample_count: index < 3 ? 0 : 20,
        note: index < 3 ? "Retrieval-only system." : null,
      })),
    }, {
      name: "latency_p95_ms",
      label: "Latency P95",
      observations: [{
        system: "full-rag",
        value: 1250.25,
        status: "measured",
        sample_count: 20,
        note: null,
      }],
    }],
  }],
  failures: [{
    case_id: "case / 1",
    system: "full-rag",
    classification: "citation_mismatch",
    detail: "The cited evidence did not support the answer.",
  }],
  error: null,
};

export const benchmarkCase: BenchmarkCaseDetail = {
  case_id: "case / 1",
  system: "full-rag",
  question: "Which policy changed?",
  expected_answer: "The July policy.",
  generated_answer: "The July policy changed.",
  expected_evidence: [
    { document_id: "expected-1", text: "Expected first." },
    { document_id: "expected-2", text: "Expected second." },
  ],
  retrieved_evidence: [
    { chunk_id: "chunk-2", text: "Retrieved first." },
    { chunk_id: "chunk-1", text: "Retrieved second." },
  ],
  metric_observations: [
    {
      name: "recall_at_5",
      label: "Recall at 5",
      system: "full-rag",
      value: 0,
      status: "measured",
      sample_count: 1,
      note: null,
    },
    {
      name: "answer_token_f1",
      label: "Answer token F1",
      system: "dense",
      value: null,
      status: "not_applicable",
      sample_count: 0,
      note: "Retrieval only.",
    },
  ],
  failure_classification: "citation_mismatch",
  public_trace: [{
    stage: "retrieve",
    decision: "selected evidence",
    retrieved_count: 4,
    fused_count: 3,
    selected_count: 2,
    retry_count: 0,
    llm_calls: 1,
    termination: "complete",
    duration_ms: 31.25,
  }],
  sanitized_raw_result: { route: "full-rag", private_prompt: null },
};

export const runtimeReady: RuntimeSnapshot = {
  state: "ready",
  configured_chat_model: "qwen3:8b",
  active_chat_model: "qwen3:8b",
  embedding_model: "nomic-embed-text",
  available_chat_models: ["qwen3:8b", "gemma3:4b"],
  detail: "Models are ready.",
  capabilities: {
    can_query: true,
    can_load_models: true,
    can_upload: true,
    can_run_benchmark: true,
  },
  active_operation: null,
  corpus: {
    document_count: 0,
    page_count: 0,
    chunk_count: 0,
    status: "empty",
  },
};

export const documentList: DocumentList = {
  documents: [{
    id: "doc/alpha",
    filename: "alpha.pdf",
    state: "ready",
    size_bytes: 2048,
    page_count: 3,
    chunk_count: 7,
    indexed_at: "2026-07-29T10:00:00Z",
    updated_at: "2026-07-29T10:00:00Z",
  }],
  corpus: {
    document_count: 1,
    page_count: 3,
    chunk_count: 7,
    status: "ready",
  },
  active_operation: null,
};

export const queryResponse: QueryResponse = {
  session_id: "9a1c4aaa-432c-4bf9-84a2-93afaf2e2c10",
  message: {
    id: "6514a7f8-afc7-4571-9745-4dc95bea47f1",
    role: "assistant",
    content: "The policy changed in July.",
    created_at: "2026-07-29T10:01:00Z",
  },
  answer_state: "supported",
  sources: [{
    label: "1",
    filename: "alpha.pdf",
    page: 2,
    excerpt: "The revised policy takes effect in July.",
  }],
  retrieval_hits: [{
    chunk_id: "alpha-2-1",
    filename: "alpha.pdf",
    page: 2,
    semantic_score: 0.91,
    sparse_score: 0.64,
    fused_score: 0.83,
    selection_score: 0.88,
    matched_subqueries: ["policy changes"],
  }],
  trace: [{
    stage: "retrieve",
    decision: "selected evidence",
    retrieved_count: 8,
    fused_count: 5,
    selected_count: 1,
    retry_count: 0,
    llm_calls: 1,
    termination: "complete",
    duration_ms: 24.6,
  }],
  diagnostics: {
    route: "retrieval",
    retrieval_strategy: "hybrid",
    subqueries: ["policy changes"],
    retry_count: 0,
    evidence_state: "supported",
    conflict_state: "none",
    citation_validation: "valid",
  },
};

export const diagnostics: DiagnosticsSnapshot = {
  state: "ready",
  title: "System ready",
  detail: "All local services are available.",
  runtime_checks: [{
    area: "runtime",
    name: "Chat model",
    state: "ready",
    detail: "qwen3:8b loaded",
  }],
  index_checks: [{
    area: "index",
    name: "Vector index",
    state: "ready",
    detail: "7 chunks available",
  }],
  evaluation_checks: [{
    area: "evaluation",
    name: "Evaluation corpus",
    state: "review",
    detail: "Optional benchmark data not loaded",
  }],
  active_operation: null,
  stale: false,
};

export function createMockApi(
  overrides: Partial<WorkspaceApi> = {},
): WorkspaceApi & Record<keyof WorkspaceApi, ReturnType<typeof vi.fn>> {
  return {
    getRuntime: vi.fn().mockResolvedValue(runtimeReady),
    loadModel: vi.fn().mockResolvedValue(runtimeReady),
    getDiagnostics: vi.fn().mockResolvedValue(diagnostics),
    getDocuments: vi.fn().mockResolvedValue({
      documents: [],
      corpus: runtimeReady.corpus,
      active_operation: null,
    }),
    uploadDocuments: vi.fn().mockResolvedValue({
      accepted: [],
      documents: documentList.documents,
      corpus: documentList.corpus,
    }),
    deleteDocument: vi.fn().mockResolvedValue(documentList),
    query: vi.fn().mockResolvedValue(queryResponse),
    clearConversation: vi.fn().mockResolvedValue(undefined),
    exportConversation: vi.fn().mockResolvedValue({
      blob: new Blob(['{"messages":[]}'], { type: "application/json" }),
      filename: "conversation-session.json",
    }),
    startBenchmark: vi.fn().mockResolvedValue(benchmarkStart),
    getBenchmark: vi.fn().mockResolvedValue(benchmarkRun),
    getLatestBenchmark: vi.fn().mockResolvedValue(benchmarkRun),
    getBenchmarkCase: vi.fn().mockResolvedValue(benchmarkCase),
    cancelBenchmark: vi.fn().mockResolvedValue({
      ...benchmarkRun,
      status: "cancellation_requested",
    }),
    downloadBenchmark: vi.fn().mockResolvedValue({
      blob: new Blob(['{"run_id":"fixture"}'], { type: "application/json" }),
      filename: "benchmark-fixture.json",
    }),
    streamBenchmarkEvents: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as WorkspaceApi & Record<keyof WorkspaceApi, ReturnType<typeof vi.fn>>;
}
