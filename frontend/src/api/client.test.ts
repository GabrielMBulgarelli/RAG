import { beforeEach, describe, expect, it, vi } from "vitest";

import { createApiClient } from "./client";

const runtime = {
  state: "ready" as const,
  configured_chat_model: "chat:configured",
  active_chat_model: "chat:active",
  embedding_model: "embed:active",
  available_chat_models: ["chat:active"],
  detail: "Ready.",
  capabilities: {
    can_query: true,
    can_load_models: true,
    can_upload: true,
    can_run_benchmark: false,
  },
  active_operation: null,
  corpus: {
    document_count: 0,
    page_count: 0,
    chunk_count: 0,
    status: "empty" as const,
  },
};

describe("workspace API client", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  it("uses the exact runtime and model-loading routes", async () => {
    fetchMock
      .mockResolvedValueOnce(Response.json(runtime))
      .mockResolvedValueOnce(Response.json(runtime));
    const api = createApiClient();

    await api.getRuntime();
    await api.loadModel("chat:active");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/runtime", expect.objectContaining({
      headers: expect.objectContaining({ Accept: "application/json" }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/runtime/models", expect.objectContaining({
      body: JSON.stringify({ chat_model: "chat:active" }),
      method: "POST",
    }));
  });

  it("sends every selected file under the files multipart field", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({
      accepted: [],
      documents: [],
      corpus: runtime.corpus,
    }));
    const first = new File(["a"], "alpha.pdf", { type: "application/pdf" });
    const second = new File(["b"], "beta.pdf", { type: "application/pdf" });

    await createApiClient().uploadDocuments([first, second]);

    const init = fetchMock.mock.calls[0]?.[1];
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/documents");
    expect(init?.method).toBe("POST");
    expect(Array.from((init?.body as FormData).getAll("files"))).toEqual([first, second]);
  });

  it("encodes document ids and sends the query session contract", async () => {
    fetchMock
      .mockResolvedValueOnce(Response.json({
        documents: [],
        corpus: runtime.corpus,
        active_operation: null,
      }))
      .mockResolvedValueOnce(Response.json({
        session_id: "9a1c4aaa-432c-4bf9-84a2-93afaf2e2c10",
        message: {
          id: "6514a7f8-afc7-4571-9745-4dc95bea47f1",
          role: "assistant",
          content: "Answer",
          created_at: "2026-07-29T10:00:00Z",
        },
        answer_state: "supported",
        sources: [],
        retrieval_hits: [],
        trace: [],
        diagnostics: {
          route: "direct",
          retrieval_strategy: "hybrid",
          subqueries: [],
          retry_count: 0,
          evidence_state: "supported",
          conflict_state: "none",
          citation_validation: "valid",
        },
      }));
    const api = createApiClient();

    await api.deleteDocument("folder/a b");
    await api.query("9a1c4aaa-432c-4bf9-84a2-93afaf2e2c10", "What changed?");

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/documents/folder%2Fa%20b");
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/query");
    expect(fetchMock.mock.calls[1]?.[1]).toEqual(expect.objectContaining({
      body: JSON.stringify({
        session_id: "9a1c4aaa-432c-4bf9-84a2-93afaf2e2c10",
        question: "What changed?",
      }),
      method: "POST",
    }));
  });

  it("returns the conversation export as a blob", async () => {
    fetchMock.mockResolvedValueOnce(new Response("# Conversation", {
      headers: { "Content-Type": "text/markdown" },
    }));

    const result = await createApiClient().exportConversation("session-id");

    expect(await result.text()).toBe("# Conversation");
    expect(fetchMock).toHaveBeenCalledWith("/api/conversations/export", expect.objectContaining({
      body: JSON.stringify({ session_id: "session-id" }),
      method: "POST",
    }));
  });

  it("throws a normalized ApiProblem for non-success responses", async () => {
    fetchMock.mockResolvedValueOnce(Response.json({
      code: "operation_busy",
      message: "Another operation is active.",
      details: { operation: "query" },
    }, { status: 409 }));

    const request = createApiClient().getDocuments();

    await expect(request).rejects.toMatchObject({
      name: "ApiClientError",
      message: "Another operation is active.",
      problem: {
        code: "operation_busy",
        details: { operation: "query" },
      },
      status: 409,
    });
  });

  it("does not expose malformed server errors", async () => {
    fetchMock.mockResolvedValueOnce(new Response("<html>failure</html>", {
      status: 500,
      headers: { "Content-Type": "text/html" },
    }));

    await expect(createApiClient().getDiagnostics()).rejects.toMatchObject({
      message: "The workspace request failed (500).",
      problem: {
        code: "request_failed",
        details: {},
      },
    });
  });
});
