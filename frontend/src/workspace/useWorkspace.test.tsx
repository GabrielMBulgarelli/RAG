import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { WorkspaceApi } from "../api/client";
import type { QueryResponse } from "../api/types";
import {
  createMockApi,
  diagnostics,
  documentList,
  queryResponse,
  runtimeReady,
} from "../test/fixtures";
import { useWorkspace } from "./useWorkspace";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function readyWorkspace(api: WorkspaceApi) {
  const hook = renderHook(() => useWorkspace(api));
  await waitFor(() => expect(hook.result.current.loadingWorkspace).toBe(false));
  return hook;
}

describe("workspace operation coordination", () => {
  beforeEach(() => {
    vi.mocked(URL.createObjectURL).mockClear();
  });

  it("atomically rejects every overlapping mutation while a query is in flight", async () => {
    const query = deferred<QueryResponse>();
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
      query: vi.fn().mockReturnValue(query.promise),
    });
    const hook = await readyWorkspace(api);

    act(() => {
      void hook.result.current.submitQuestion("First question");
    });
    await waitFor(() => expect(api.query).toHaveBeenCalledTimes(1));
    const exchangeId = hook.result.current.exchanges[0]?.id;

    act(() => {
      void hook.result.current.submitQuestion("Second question");
      if (exchangeId) {
        void hook.result.current.retryQuestion(exchangeId);
      }
      void hook.result.current.uploadDocuments([
        new File(["text"], "notes.txt", { type: "text/plain" }),
      ]);
      void hook.result.current.deleteDocument("doc/alpha");
      void hook.result.current.loadModel("gemma3:4b");
      void hook.result.current.clearConversation();
      void hook.result.current.exportConversation();
    });

    expect(hook.result.current.busy).toBe(true);
    expect(hook.result.current.busyKind).toBe("query");
    expect(api.query).toHaveBeenCalledTimes(1);
    expect(api.uploadDocuments).not.toHaveBeenCalled();
    expect(api.deleteDocument).not.toHaveBeenCalled();
    expect(api.loadModel).not.toHaveBeenCalled();
    expect(api.clearConversation).not.toHaveBeenCalled();
    expect(api.exportConversation).not.toHaveBeenCalled();

    await act(async () => {
      query.resolve(queryResponse);
      await query.promise;
    });
    await waitFor(() => expect(hook.result.current.busy).toBe(false));
  });

  it("blocks callbacks against a server-reported active operation", async () => {
    const busyRuntime = {
      ...runtimeReady,
      active_operation: {
        operation_id: "ed4d9182-3242-450d-ab84-7d2c93266140",
        kind: "index_documents" as const,
        started_at: "2026-07-29T10:00:00Z",
        benchmark_run_id: null,
        cancellation_requested: false,
      },
    };
    const api = createMockApi({
      getRuntime: vi.fn().mockResolvedValue(busyRuntime),
    });
    const hook = await readyWorkspace(api);

    await act(async () => {
      await hook.result.current.submitQuestion("Blocked question");
      await hook.result.current.uploadDocuments([
        new File(["text"], "notes.txt", { type: "text/plain" }),
      ]);
    });

    expect(hook.result.current.busy).toBe(true);
    expect(hook.result.current.busyKind).toBe("index_documents");
    expect(api.query).not.toHaveBeenCalled();
    expect(api.uploadDocuments).not.toHaveBeenCalled();
  });

  it("does not refresh or download after unmount", async () => {
    const exported = deferred<Awaited<ReturnType<WorkspaceApi["exportConversation"]>>>();
    const api = createMockApi({
      exportConversation: vi.fn().mockReturnValue(exported.promise),
    });
    const hook = await readyWorkspace(api);

    act(() => {
      void hook.result.current.exportConversation();
    });
    await waitFor(() => expect(api.exportConversation).toHaveBeenCalledTimes(1));
    hook.unmount();

    exported.resolve({
      blob: new Blob(["{}"], { type: "application/json" }),
      filename: "conversation-safe.json",
    });
    await exported.promise;
    await Promise.resolve();

    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it("ignores an old API mutation continuation after the API prop changes", async () => {
    const upload = deferred<Awaited<ReturnType<WorkspaceApi["uploadDocuments"]>>>();
    const firstApi = createMockApi({
      uploadDocuments: vi.fn().mockReturnValue(upload.promise),
    });
    const secondRuntime = { ...runtimeReady, active_chat_model: "gemma3:4b" };
    const secondApi = createMockApi({
      getRuntime: vi.fn().mockResolvedValue(secondRuntime),
    });
    const hook = renderHook(
      ({ api }: { api: WorkspaceApi }) => useWorkspace(api),
      { initialProps: { api: firstApi } },
    );
    await waitFor(() => expect(hook.result.current.loadingWorkspace).toBe(false));

    act(() => {
      void hook.result.current.uploadDocuments([
        new File(["text"], "notes.txt", { type: "text/plain" }),
      ]);
    });
    await waitFor(() => expect(firstApi.uploadDocuments).toHaveBeenCalledTimes(1));
    hook.rerender({ api: secondApi });
    await waitFor(() => {
      expect(hook.result.current.runtime?.active_chat_model).toBe("gemma3:4b");
    });

    upload.resolve({
      accepted: [],
      documents: documentList.documents,
      corpus: documentList.corpus,
    });
    await upload.promise;
    await Promise.resolve();

    expect(firstApi.getRuntime).toHaveBeenCalledTimes(1);
    expect(hook.result.current.runtime?.active_chat_model).toBe("gemma3:4b");
  });

  it("deduplicates concurrent diagnostics reads", async () => {
    const diagnosticsRequest = deferred<typeof diagnostics>();
    const api = createMockApi({
      getDiagnostics: vi.fn().mockReturnValue(diagnosticsRequest.promise),
    });
    const hook = await readyWorkspace(api);

    act(() => {
      void hook.result.current.refreshDiagnostics();
      void hook.result.current.refreshDiagnostics();
    });

    expect(api.getDiagnostics).toHaveBeenCalledTimes(1);
    diagnosticsRequest.resolve(diagnostics);
    await diagnosticsRequest.promise;
  });

  it("refreshes diagnostics after a successful model load", async () => {
    const refreshedDiagnostics = {
      ...diagnostics,
      title: "Models loaded",
      runtime_checks: [{
        area: "runtime",
        name: "Chat model",
        state: "ready" as const,
        detail: "gemma3:4b loaded",
      }],
    };
    const api = createMockApi({
      getDiagnostics: vi.fn()
        .mockResolvedValueOnce(diagnostics)
        .mockResolvedValueOnce(refreshedDiagnostics),
    });
    const hook = await readyWorkspace(api);
    await act(async () => {
      await hook.result.current.refreshDiagnostics();
      await hook.result.current.loadModel("gemma3:4b");
    });

    expect(api.getDiagnostics).toHaveBeenCalledTimes(2);
    expect(hook.result.current.diagnostics?.title).toBe("Models loaded");
  });

  it("removes stale diagnostics when their post-load refresh fails", async () => {
    const api = createMockApi({
      getDiagnostics: vi.fn()
        .mockResolvedValueOnce(diagnostics)
        .mockRejectedValueOnce(new Error("Fresh diagnostics unavailable.")),
    });
    const hook = await readyWorkspace(api);
    await act(async () => {
      await hook.result.current.refreshDiagnostics();
      await hook.result.current.loadModel("gemma3:4b");
    });

    expect(hook.result.current.diagnostics).toBeNull();
    expect(hook.result.current.actionError).toBe("Fresh diagnostics unavailable.");
  });
});
