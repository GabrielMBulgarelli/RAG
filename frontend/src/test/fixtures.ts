import { vi } from "vitest";

import type { WorkspaceApi } from "../api/client";
import type {
  DiagnosticsSnapshot,
  DocumentList,
  QueryResponse,
  RuntimeSnapshot,
} from "../api/types";

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
    ...overrides,
  } as WorkspaceApi & Record<keyof WorkspaceApi, ReturnType<typeof vi.fn>>;
}
