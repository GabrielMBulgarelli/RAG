export type RuntimeState = "ready" | "not_loaded" | "blocked" | "error";
export type CheckState = "ready" | "review" | "blocked" | "not_loaded" | "error";
export type CorpusState = "ready" | "empty" | "review" | "error";
export type DiagnosticsState = "ready" | "review" | "blocked" | "error";
export type AnswerState =
  | "supported"
  | "limited"
  | "abstention"
  | "unavailable"
  | "completed";

export type OperationKind =
  | "index_documents"
  | "delete_document"
  | "load_model"
  | "query"
  | "benchmark";

export interface ActiveOperation {
  operation_id: string;
  kind: OperationKind;
  started_at: string;
  benchmark_run_id: string | null;
  cancellation_requested: boolean;
}

export interface CapabilitySnapshot {
  can_query: boolean;
  can_load_models: boolean;
  can_upload: boolean;
  can_run_benchmark: boolean;
}

export interface CorpusSnapshot {
  document_count: number;
  page_count: number;
  chunk_count: number;
  status: CorpusState;
}

export interface RuntimeSnapshot {
  state: RuntimeState;
  configured_chat_model: string;
  active_chat_model: string | null;
  embedding_model: string;
  available_chat_models: string[];
  detail: string;
  capabilities: CapabilitySnapshot;
  active_operation: ActiveOperation | null;
  corpus: CorpusSnapshot;
}

export interface DiagnosticCheck {
  area: string;
  name: string;
  state: CheckState;
  detail: string;
}

export interface DiagnosticsSnapshot {
  state: DiagnosticsState;
  title: string;
  detail: string;
  runtime_checks: DiagnosticCheck[];
  index_checks: DiagnosticCheck[];
  evaluation_checks: DiagnosticCheck[];
  active_operation: ActiveOperation | null;
  stale: boolean;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  state: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  indexed_at: string | null;
  updated_at: string;
}

export interface DocumentList {
  documents: DocumentRecord[];
  corpus: CorpusSnapshot;
  active_operation: ActiveOperation | null;
}

export interface UploadAccepted {
  filename: string;
  document_id: string;
}

export interface UploadBatchResult {
  accepted: UploadAccepted[];
  documents: DocumentRecord[];
  corpus: CorpusSnapshot;
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface Source {
  label: string;
  filename: string;
  page: number | null;
  excerpt: string;
}

export interface RetrievalHit {
  chunk_id: string;
  filename: string;
  page: number | null;
  semantic_score: number | null;
  sparse_score: number | null;
  fused_score: number | null;
  selection_score: number | null;
  matched_subqueries: string[];
}

export interface TraceEvent {
  stage: string;
  decision: string;
  retrieved_count: number | null;
  fused_count: number | null;
  selected_count: number | null;
  retry_count: number;
  llm_calls: number;
  termination: string;
  duration_ms: number | null;
}

export interface QueryDiagnostics {
  route: string;
  retrieval_strategy: string;
  subqueries: string[];
  retry_count: number;
  evidence_state: string;
  conflict_state: string;
  citation_validation: string;
}

export interface QueryResponse {
  session_id: string;
  message: ConversationMessage;
  answer_state: AnswerState;
  sources: Source[];
  retrieval_hits: RetrievalHit[];
  trace: TraceEvent[];
  diagnostics: QueryDiagnostics;
}

export interface ApiProblem {
  code: string;
  message: string;
  details: Record<string, unknown>;
}
