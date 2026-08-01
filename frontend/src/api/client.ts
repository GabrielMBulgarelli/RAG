import type {
  ApiProblem,
  BenchmarkCaseDetail,
  BenchmarkCaseSummary,
  BenchmarkEvent,
  BenchmarkEventType,
  BenchmarkRun,
  BenchmarkStartResponse,
  DiagnosticsSnapshot,
  DocumentList,
  QueryResponse,
  RuntimeSnapshot,
  UploadBatchResult,
} from "./types";

export interface WorkspaceApi {
  getRuntime(): Promise<RuntimeSnapshot>;
  loadModel(chatModel: string): Promise<RuntimeSnapshot>;
  getDiagnostics(): Promise<DiagnosticsSnapshot>;
  getDocuments(): Promise<DocumentList>;
  uploadDocuments(files: File[]): Promise<UploadBatchResult>;
  deleteDocument(documentId: string): Promise<DocumentList>;
  query(sessionId: string, question: string): Promise<QueryResponse>;
  clearConversation(sessionId: string): Promise<void>;
  exportConversation(sessionId: string): Promise<DownloadFile>;
  startBenchmark(): Promise<BenchmarkStartResponse>;
  getBenchmark(runId: string): Promise<BenchmarkRun>;
  getLatestBenchmark(): Promise<BenchmarkRun>;
  getBenchmarkCases(runId: string): Promise<BenchmarkCaseSummary[]>;
  getBenchmarkCase(
    runId: string,
    caseId: string,
    systemId: string,
  ): Promise<BenchmarkCaseDetail>;
  cancelBenchmark(runId: string): Promise<BenchmarkRun>;
  downloadBenchmark(runId: string): Promise<DownloadFile>;
  streamBenchmarkEvents(
    runId: string,
    lastEventId: number,
    signal: AbortSignal,
    onEvent: (event: BenchmarkEvent) => void,
  ): Promise<void>;
}

export interface DownloadFile {
  blob: Blob;
  filename: string;
}

export class ApiClientError extends Error {
  readonly problem: ApiProblem;
  readonly status: number;

  constructor(problem: ApiProblem, status: number) {
    super(problem.message);
    this.name = "ApiClientError";
    this.problem = problem;
    this.status = status;
  }
}

function isApiProblem(value: unknown): value is ApiProblem {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ApiProblem>;
  return (
    typeof candidate.code === "string"
    && typeof candidate.message === "string"
    && typeof candidate.details === "object"
    && candidate.details !== null
    && !Array.isArray(candidate.details)
  );
}

async function problemFrom(response: Response): Promise<ApiProblem> {
  try {
    const body: unknown = await response.clone().json();
    if (isApiProblem(body)) {
      return body;
    }
  } catch {
    // The normalized fallback below avoids leaking HTML or server internals.
  }
  return {
    code: "request_failed",
    message: `The workspace request failed (${response.status}).`,
    details: {},
  };
}

async function ensureSuccess(response: Response): Promise<Response> {
  if (!response.ok) {
    throw new ApiClientError(await problemFrom(response), response.status);
  }
  return response;
}

function jsonInit(method: "POST", body: unknown): RequestInit {
  return {
    method,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  };
}

function exportFilename(disposition: string | null, fallback: string): string {
  const encoded = disposition?.match(/filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i)?.[1];
  const regular = disposition?.match(/filename\s*=\s*(?:"([^"]+)"|([^;]+))/i);
  let candidate = encoded ?? regular?.[1] ?? regular?.[2] ?? "";
  candidate = candidate.trim().replace(/^["']|["']$/g, "");
  if (encoded) {
    try {
      candidate = decodeURIComponent(candidate);
    } catch {
      candidate = "";
    }
  }
  const leaf = candidate.split(/[\\/]/).at(-1)?.replace(/[\u0000-\u001f\u007f]/g, "").trim();
  return leaf && leaf !== "." && leaf !== ".." ? leaf : fallback;
}

const BENCHMARK_EVENT_TYPES = new Set<BenchmarkEventType>([
  "benchmark.started",
  "system.started",
  "case.started",
  "case.completed",
  "case.failed",
  "system.completed",
  "benchmark.cancellation_requested",
  "benchmark.cancelled",
  "benchmark.completed",
  "benchmark.failed",
  "heartbeat",
]);
const UUID_PATTERN = (
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function benchmarkEvent(
  value: unknown,
  sseId: string,
  sseType: string,
): BenchmarkEvent {
  if (!isRecord(value)) {
    throw new Error("The benchmark event stream returned a malformed benchmark event.");
  }
  const type = value.type;
  const valid = (
    Number.isInteger(value.event_id)
    && Number(value.event_id) >= 1
    && typeof value.run_id === "string"
    && UUID_PATTERN.test(value.run_id)
    && typeof type === "string"
    && BENCHMARK_EVENT_TYPES.has(type as BenchmarkEventType)
    && typeof value.timestamp === "string"
    && isRecord(value.data)
    && sseId === String(value.event_id)
    && sseType === type
  );
  if (!valid) {
    throw new Error("The benchmark event stream returned a malformed benchmark event.");
  }
  return value as unknown as BenchmarkEvent;
}

function parseEventBlock(block: string): BenchmarkEvent | null {
  let id = "";
  let type = "";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let fieldValue = separator < 0 ? "" : line.slice(separator + 1);
    if (fieldValue.startsWith(" ")) {
      fieldValue = fieldValue.slice(1);
    }
    if (field === "id") {
      id = fieldValue;
    } else if (field === "event") {
      type = fieldValue;
    } else if (field === "data") {
      data.push(fieldValue);
    }
  }
  if (data.length === 0) {
    return null;
  }
  try {
    return benchmarkEvent(JSON.parse(data.join("\n")), id, type);
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(
        "The benchmark event stream returned a malformed benchmark event.",
        { cause: error },
      );
    }
    throw error;
  }
}

async function readBenchmarkEvents(
  response: Response,
  signal: AbortSignal,
  onEvent: (event: BenchmarkEvent) => void,
): Promise<void> {
  if (!response.body) {
    throw new Error("The benchmark event stream returned no response body.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  if (signal.aborted) {
    await reader.cancel(signal.reason);
    throw new DOMException("The benchmark event stream was aborted.", "AbortError");
  }
  const cancel = () => {
    void reader.cancel(signal.reason);
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary = buffer.match(/\r?\n\r?\n/);
      while (boundary?.index !== undefined) {
        const event = parseEventBlock(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary[0].length);
        if (event) {
          onEvent(event);
        }
        boundary = buffer.match(/\r?\n\r?\n/);
      }
      if (done) {
        if (buffer.trim()) {
          const event = parseEventBlock(buffer);
          if (event) {
            onEvent(event);
          }
        }
        break;
      }
    }
    if (signal.aborted) {
      throw new DOMException("The benchmark event stream was aborted.", "AbortError");
    }
  } finally {
    signal.removeEventListener("abort", cancel);
    reader.releaseLock();
  }
}

export function createApiClient(baseUrl = ""): WorkspaceApi {
  const endpoint = (path: string) => `${baseUrl.replace(/\/$/, "")}${path}`;
  const json = async <T>(path: string, init: RequestInit = {}): Promise<T> => {
    const response = await ensureSuccess(await fetch(endpoint(path), {
      ...init,
      headers: {
        Accept: "application/json",
        ...init.headers,
      },
    }));
    return response.json() as Promise<T>;
  };

  return {
    getRuntime: () => json<RuntimeSnapshot>("/api/runtime"),
    loadModel: (chatModel) => json<RuntimeSnapshot>(
      "/api/runtime/models",
      jsonInit("POST", { chat_model: chatModel }),
    ),
    getDiagnostics: () => json<DiagnosticsSnapshot>("/api/diagnostics"),
    getDocuments: () => json<DocumentList>("/api/documents"),
    uploadDocuments: async (files) => {
      const body = new FormData();
      files.forEach((file) => body.append("files", file));
      return json<UploadBatchResult>("/api/documents", { method: "POST", body });
    },
    deleteDocument: (documentId) => json<DocumentList>(
      `/api/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" },
    ),
    query: (sessionId, question) => json<QueryResponse>(
      "/api/query",
      jsonInit("POST", { session_id: sessionId, question }),
    ),
    clearConversation: async (sessionId) => {
      await ensureSuccess(await fetch(
        endpoint(`/api/conversations/${encodeURIComponent(sessionId)}`),
        { method: "DELETE", headers: { Accept: "application/json" } },
      ));
    },
    exportConversation: async (sessionId) => {
      const response = await ensureSuccess(await fetch(
        endpoint("/api/conversations/export"),
        jsonInit("POST", { session_id: sessionId }),
      ));
      return {
        blob: await response.blob(),
        filename: exportFilename(
          response.headers.get("Content-Disposition"),
          "conversation.json",
        ),
      };
    },
    startBenchmark: () => json<BenchmarkStartResponse>(
      "/api/benchmarks",
      { method: "POST" },
    ),
    getBenchmark: (runId) => json<BenchmarkRun>(
      `/api/benchmarks/${encodeURIComponent(runId)}`,
    ),
    getLatestBenchmark: () => json<BenchmarkRun>("/api/benchmarks/latest"),
    getBenchmarkCases: (runId) => json<BenchmarkCaseSummary[]>(
      `/api/benchmarks/${encodeURIComponent(runId)}/cases`,
    ),
    getBenchmarkCase: (runId, caseId, systemId) => json<BenchmarkCaseDetail>(
      `/api/benchmarks/${encodeURIComponent(runId)}`
      + `/cases/${encodeURIComponent(caseId)}/systems/${encodeURIComponent(systemId)}`,
    ),
    cancelBenchmark: (runId) => json<BenchmarkRun>(
      `/api/benchmarks/${encodeURIComponent(runId)}/cancel`,
      { method: "POST" },
    ),
    downloadBenchmark: async (runId) => {
      const response = await ensureSuccess(await fetch(
        endpoint(`/api/benchmarks/${encodeURIComponent(runId)}/download`),
        { headers: { Accept: "application/zip" } },
      ));
      return {
        blob: await response.blob(),
        filename: exportFilename(
          response.headers.get("Content-Disposition"),
          "benchmark.zip",
        ),
      };
    },
    streamBenchmarkEvents: async (runId, lastEventId, signal, onEvent) => {
      const response = await ensureSuccess(await fetch(
        endpoint(`/api/benchmarks/${encodeURIComponent(runId)}/events`),
        {
          headers: {
            Accept: "text/event-stream",
            "Last-Event-ID": String(lastEventId),
          },
          signal,
        },
      ));
      await readBenchmarkEvents(response, signal, onEvent);
    },
  };
}
