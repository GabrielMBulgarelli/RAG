import type {
  ApiProblem,
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

function exportFilename(disposition: string | null): string {
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
  return leaf && leaf !== "." && leaf !== ".." ? leaf : "conversation.json";
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
        filename: exportFilename(response.headers.get("Content-Disposition")),
      };
    },
  };
}
