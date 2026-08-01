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

  it("returns the backend JSON export with its server filename", async () => {
    fetchMock.mockResolvedValueOnce(new Response('{"messages":[]}', {
      headers: {
        "Content-Type": "application/json",
        "Content-Disposition": (
          'attachment; filename="conversation-9a1c4aaa-432c-4bf9-84a2-93afaf2e2c10.json"'
        ),
      },
    }));

    const result = await createApiClient().exportConversation("session-id");

    expect(result.filename).toBe(
      "conversation-9a1c4aaa-432c-4bf9-84a2-93afaf2e2c10.json",
    );
    expect(result.blob.type).toBe("application/json");
    expect(await result.blob.text()).toBe('{"messages":[]}');
    expect(fetchMock).toHaveBeenCalledWith("/api/conversations/export", expect.objectContaining({
      body: JSON.stringify({ session_id: "session-id" }),
      method: "POST",
    }));
  });

  it("uses a safe JSON filename when export disposition is absent or unsafe", async () => {
    fetchMock
      .mockResolvedValueOnce(new Response("{}", {
        headers: {
          "Content-Type": "application/json",
          "Content-Disposition": 'attachment; filename="../../unsafe.json"',
        },
      }))
      .mockResolvedValueOnce(new Response("{}", {
        headers: { "Content-Type": "application/json" },
      }));
    const api = createApiClient();

    expect((await api.exportConversation("session-id")).filename).toBe("unsafe.json");
    expect((await api.exportConversation("session-id")).filename).toBe("conversation.json");
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

  it("uses exact benchmark resource paths, encoding, cancel, and download filename", async () => {
    const runId = "4cbdbcb9-5a57-4514-a392-2dce907456d5";
    const start = {
      run_id: runId,
      status: "queued",
      links: {
        run: `/api/benchmarks/${runId}`,
        events: `/api/benchmarks/${runId}/events`,
        download: `/api/benchmarks/${runId}/download`,
      },
    };
    const run = {
      ...start,
      progress: { completed_cases: 0, total_cases: 0, total_systems: 0 },
      metadata: {
        dataset: "multihop",
        split: "development",
        systems: [],
        chat_model: "qwen3:8b",
        embedding_model: "nomic-embed-text",
        started_at: null,
        completed_at: null,
        reproducibility: {},
      },
      sections: [],
      failures: [],
      error: null,
    };
    fetchMock
      .mockResolvedValueOnce(Response.json(start, { status: 202 }))
      .mockResolvedValueOnce(Response.json(run))
      .mockResolvedValueOnce(Response.json(run))
      .mockResolvedValueOnce(Response.json([]))
      .mockResolvedValueOnce(Response.json({
        case_id: "case / 1",
        system: "dense + rerank",
        question: "Question",
        expected_answer: null,
        generated_answer: null,
        expected_evidence: [],
        retrieved_evidence: [],
        metric_observations: [],
        failure_classification: null,
        public_trace: [],
        sanitized_raw_result: null,
      }))
      .mockResolvedValueOnce(Response.json({ ...run, status: "cancellation_requested" }, {
        status: 202,
      }))
      .mockResolvedValueOnce(new Response('{"run_id":"download"}', {
        headers: {
          "Content-Type": "application/zip",
          "Content-Disposition": 'attachment; filename="../benchmark-safe.json"',
        },
      }))
      .mockResolvedValueOnce(new Response("zip bytes", {
        headers: { "Content-Type": "application/zip" },
      }));
    const api = createApiClient();

    await api.startBenchmark();
    await api.getBenchmark(runId);
    await api.getLatestBenchmark();
    await api.getBenchmarkCases(runId);
    await api.getBenchmarkCase(runId, "case / 1", "dense + rerank");
    await api.cancelBenchmark(runId);
    const download = await api.downloadBenchmark(runId);
    const fallbackDownload = await api.downloadBenchmark(runId);

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/benchmarks",
      `/api/benchmarks/${runId}`,
      "/api/benchmarks/latest",
      `/api/benchmarks/${runId}/cases`,
      `/api/benchmarks/${runId}/cases/case%20%2F%201/systems/dense%20%2B%20rerank`,
      `/api/benchmarks/${runId}/cancel`,
      `/api/benchmarks/${runId}/download`,
      `/api/benchmarks/${runId}/download`,
    ]);
    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({ method: "POST" }));
    expect(fetchMock.mock.calls[5]?.[1]).toEqual(expect.objectContaining({ method: "POST" }));
    expect(download.filename).toBe("benchmark-safe.json");
    expect(await download.blob.text()).toBe('{"run_id":"download"}');
    expect(fallbackDownload.filename).toBe("benchmark.zip");
    expect(fetchMock.mock.calls[6]?.[1]).toEqual(expect.objectContaining({
      headers: { Accept: "application/zip" },
    }));
  });
});
