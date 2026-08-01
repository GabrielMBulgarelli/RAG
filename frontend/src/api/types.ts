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

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export type BenchmarkRunStatus =
  | "queued"
  | "running"
  | "cancellation_requested"
  | "cancelled"
  | "completed"
  | "failed";

export interface BenchmarkLinks {
  run: string;
  events: string;
  download: string;
}

export interface BenchmarkStartResponse {
  run_id: string;
  status: BenchmarkRunStatus;
  links: BenchmarkLinks;
}

export type BenchmarkMetricStatus =
  | "measured"
  | "not_applicable"
  | "no_eligible_cases";

export interface BenchmarkMetricObservation {
  system: string;
  value: number | null;
  status: BenchmarkMetricStatus;
  sample_count: number;
  note: string | null;
}

export interface BenchmarkCaseMetricObservation extends BenchmarkMetricObservation {
  name: string;
  label: string;
}

export interface BenchmarkMetric {
  name: string;
  label: string;
  observations: BenchmarkMetricObservation[];
}

export interface BenchmarkSystem {
  id: string;
  label: string;
}

export interface BenchmarkSection {
  id: string;
  title: string;
  system_ids: string[];
  metrics: BenchmarkMetric[];
  detail: string | null;
}

export interface BenchmarkFailure {
  case_id: string;
  system: string;
  classification: string;
  detail: string;
}

export interface BenchmarkProgress {
  completed_cases: number;
  total_cases: number;
  current_system: string | null;
  current_system_index: number | null;
  total_systems: number;
  current_case_id: string | null;
  current_case_index: number | null;
}

export interface BenchmarkMetadata {
  dataset: string;
  split: string;
  systems: BenchmarkSystem[];
  chat_model: string;
  embedding_model: string;
  started_at: string | null;
  completed_at: string | null;
  reproducibility: Record<string, JsonValue>;
}

export interface BenchmarkRun {
  run_id: string;
  status: BenchmarkRunStatus;
  progress: BenchmarkProgress;
  metadata: BenchmarkMetadata;
  sections: BenchmarkSection[];
  failures: BenchmarkFailure[];
  links: BenchmarkLinks;
  error: ApiProblem | null;
}

export type BenchmarkEventType =
  | "benchmark.started"
  | "system.started"
  | "case.started"
  | "case.completed"
  | "case.failed"
  | "system.completed"
  | "benchmark.cancellation_requested"
  | "benchmark.cancelled"
  | "benchmark.completed"
  | "benchmark.failed"
  | "heartbeat";

export interface BenchmarkEvent {
  event_id: number;
  run_id: string;
  type: BenchmarkEventType;
  timestamp: string;
  data: Record<string, JsonValue>;
}

export interface BenchmarkCaseDetail {
  case_id: string;
  system: string;
  question: string;
  expected_answer: string | null;
  generated_answer: string | null;
  expected_evidence: Array<Record<string, JsonValue>>;
  retrieved_evidence: Array<Record<string, JsonValue>>;
  metric_observations: BenchmarkCaseMetricObservation[];
  failure_classification: string | null;
  public_trace: TraceEvent[];
  sanitized_raw_result: Record<string, JsonValue> | null;
}

export type BenchmarkCaseOutcome =
  | "successful"
  | "expectation_failure"
  | "runtime_failure";

export interface BenchmarkCaseSummary {
  case_id: string;
  system: string;
  question: string;
  outcome: BenchmarkCaseOutcome;
  failure_classification: string | null;
}
