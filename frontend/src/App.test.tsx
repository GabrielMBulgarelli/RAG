import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { ApiClientError } from "./api/client";
import type { QueryResponse } from "./api/types";
import {
  createMockApi,
  diagnostics,
  documentList,
  queryResponse,
  runtimeReady,
} from "./test/fixtures";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("single workspace", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it("loads runtime and documents into one permanent workspace", async () => {
    const api = createMockApi();
    render(<App api={api} />);

    expect(await screen.findByRole("heading", { name: "Local Document RAG" })).toBeVisible();
    expect(api.getRuntime).toHaveBeenCalledTimes(1);
    expect(api.getDocuments).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("navigation")).not.toBeInTheDocument();
    expect(screen.getByText(/index local documents to begin/i)).toBeVisible();
  });

  it("renders a ready corpus and its compact document list", async () => {
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
      getRuntime: vi.fn().mockResolvedValue({
        ...runtimeReady,
        corpus: documentList.corpus,
      }),
    });
    render(<App api={api} />);

    expect(await screen.findByText("alpha.pdf")).toBeVisible();
    expect(screen.getByText("1 indexed document")).toBeVisible();
    expect(screen.getByText("3 pages · 7 chunks")).toBeVisible();
  });

  it("submits a trimmed question with a stable session and renders evidence", async () => {
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
      query: vi.fn().mockImplementation(async (sessionId: string) => ({
        ...queryResponse,
        session_id: sessionId,
      })),
    });
    const user = userEvent.setup();
    render(<App api={api} />);
    await screen.findByText("alpha.pdf");

    const composer = screen.getByLabelText("Ask about your documents");
    await user.type(composer, "  What changed?  ");
    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(api.query).toHaveBeenCalledWith(expect.stringMatching(/^[0-9a-f-]{36}$/), "What changed?");
    expect(await screen.findByText("The policy changed in July.")).toBeVisible();
    expect(screen.getByText("What changed?")).toBeVisible();
    const citation = screen.getByRole("button", { name: "Source 1" });
    expect(citation).toBeVisible();
    await user.click(citation);
    expect(screen.getByRole("button", { name: "Source 1, selected" })).toHaveAttribute(
      "aria-current",
      "true",
    );
    expect(screen.getByText("The revised policy takes effect in July.")).toBeVisible();
  });

  it("selects an older source-less answer through an explicit keyboard control", async () => {
    const olderResponse = {
      ...queryResponse,
      message: {
        ...queryResponse.message,
        id: "40a69776-90a7-4a1e-a674-0202462e9d24",
        content: "An older answer without sources.",
      },
      sources: [],
      diagnostics: {
        ...queryResponse.diagnostics,
        route: "older-answer",
      },
    };
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
      query: vi.fn()
        .mockResolvedValueOnce(olderResponse)
        .mockResolvedValueOnce(queryResponse),
    });
    const user = userEvent.setup();
    render(<App api={api} />);
    await screen.findByText("alpha.pdf");

    const composer = screen.getByLabelText("Ask about your documents");
    await user.type(composer, "First");
    await user.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText("An older answer without sources.");
    await user.type(composer, "Second");
    await user.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText("The policy changed in July.");

    const inspectOlder = screen.getAllByRole("button", { name: /inspect answer/i })[0];
    inspectOlder.focus();
    await user.keyboard("{Enter}");

    expect(inspectOlder).toHaveFocus();
    expect(screen.getByText("No sources accompanied this answer.")).toBeVisible();
  });

  it("keeps the user message and offers retry without inventing an answer", async () => {
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
      query: vi.fn()
        .mockRejectedValueOnce(new ApiClientError({
          code: "runtime_unavailable",
          message: "The model is not available.",
          details: {},
        }, 503))
        .mockResolvedValueOnce(queryResponse),
    });
    const user = userEvent.setup();
    render(<App api={api} />);
    await screen.findByText("alpha.pdf");

    await user.type(screen.getByLabelText("Ask about your documents"), "Explain the policy");
    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The model is not available.");
    expect(screen.getByText("Explain the policy")).toBeVisible();
    expect(screen.queryByLabelText("Assistant response")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry question" }));
    expect(api.query).toHaveBeenCalledTimes(2);
    expect(await screen.findByText("The policy changed in July.")).toBeVisible();
  });

  it("switches and collapses the evidence inspector with accessible accordions", async () => {
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
    });
    const user = userEvent.setup();
    render(<App api={api} />);
    await screen.findByText("alpha.pdf");
    await user.type(screen.getByLabelText("Ask about your documents"), "What changed?");
    await user.keyboard("{Control>}{Enter}{/Control}");
    await screen.findByText("The policy changed in July.");

    await user.click(screen.getByRole("tab", { name: "Details" }));
    const detailsTab = screen.getByRole("tab", { name: "Details" });
    const sourcesTab = screen.getByRole("tab", { name: "Sources" });
    expect(detailsTab).toHaveAttribute("id", "inspector-tab-details");
    expect(detailsTab).toHaveAttribute("aria-controls", "inspector-panel-details");
    expect(screen.getByRole("tabpanel", { name: "Details" })).toHaveAttribute(
      "aria-labelledby",
      "inspector-tab-details",
    );
    expect(screen.getByText("Retrieval")).toBeVisible();
    expect(screen.getByText("Execution trace")).toBeVisible();
    expect(screen.getByText("Query diagnostics")).toBeVisible();
    expect(screen.getByText("Raw trace")).toBeVisible();

    detailsTab.focus();
    await user.keyboard("{ArrowLeft}");
    expect(sourcesTab).toHaveFocus();
    expect(sourcesTab).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{End}");
    expect(detailsTab).toHaveFocus();
    await user.keyboard("{Home}");
    expect(sourcesTab).toHaveFocus();

    await user.click(screen.getByRole("button", { name: "Collapse inspector" }));
    expect(screen.queryByRole("tab", { name: "Sources" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Expand inspector" }));
    expect(screen.getByRole("tab", { name: "Sources" })).toBeVisible();
  });

  it("refreshes runtime and documents after upload and exposes upload errors", async () => {
    const api = createMockApi();
    const user = userEvent.setup();
    const { rerender } = render(<App api={api} />);
    await screen.findByRole("heading", { name: "Local Document RAG" });

    const file = new File(["pdf"], "new.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText("Upload documents"), file);

    await waitFor(() => expect(api.uploadDocuments).toHaveBeenCalledWith([file]));
    expect(api.getRuntime).toHaveBeenCalledTimes(2);
    expect(api.getDocuments).toHaveBeenCalledTimes(2);

    const failingApi = createMockApi({
      uploadDocuments: vi.fn().mockRejectedValue(new Error("Upload failed safely.")),
    });
    rerender(<App api={failingApi} />);
    await user.upload(screen.getByLabelText("Upload documents"), file);
    expect(await screen.findByRole("alert")).toHaveTextContent("Upload failed safely.");
  });

  it("owns document details and exact delete confirmation centrally", async () => {
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
    });
    const user = userEvent.setup();
    render(<App api={api} />);

    await user.click(await screen.findByRole("button", { name: /alpha\.pdf details/i }));
    const detailsDialog = screen.getByRole("dialog", { name: "Document details" });
    expect(within(detailsDialog).getByText("3 pages")).toBeVisible();
    await user.click(within(detailsDialog).getByRole("button", { name: "Delete document" }));

    const confirmDialog = screen.getByRole("dialog", { name: "Delete alpha.pdf?" });
    expect(confirmDialog).toHaveTextContent("alpha.pdf");
    await user.click(within(confirmDialog).getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(api.deleteDocument).toHaveBeenCalledWith("doc/alpha"));
    expect(api.getRuntime).toHaveBeenCalledTimes(2);
    expect(api.getDocuments).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("groups, refreshes, and safely reports diagnostics while supporting model load", async () => {
    const notLoaded = {
      ...runtimeReady,
      state: "not_loaded" as const,
      active_chat_model: null,
      detail: "Load a chat model.",
      capabilities: {
        ...runtimeReady.capabilities,
        can_query: false,
      },
    };
    const api = createMockApi({
      getRuntime: vi.fn()
        .mockResolvedValueOnce(notLoaded)
        .mockResolvedValue(runtimeReady),
      getDiagnostics: vi.fn()
        .mockResolvedValueOnce(diagnostics)
        .mockResolvedValueOnce({
          ...diagnostics,
          title: "Models loaded",
        })
        .mockRejectedValueOnce(new Error("Diagnostics unavailable.")),
      loadModel: vi.fn().mockResolvedValue(runtimeReady),
    });
    const user = userEvent.setup();
    render(<App api={api} />);

    await user.click(await screen.findByRole("button", { name: "System diagnostics" }));
    const dialog = await screen.findByRole("dialog", { name: "System diagnostics" });
    expect(within(dialog).getByRole("heading", { name: "Runtime" })).toBeVisible();
    expect(within(dialog).getByRole("heading", { name: "Index" })).toBeVisible();
    expect(within(dialog).getByRole("heading", { name: "Evaluation" })).toBeVisible();

    await user.click(within(dialog).getByRole("button", { name: "Load models" }));
    await waitFor(() => expect(api.loadModel).toHaveBeenCalledWith("qwen3:8b"));
    expect(api.getRuntime).toHaveBeenCalledTimes(2);
    expect(api.getDocuments).toHaveBeenCalledTimes(2);
    expect(api.getDiagnostics).toHaveBeenCalledTimes(2);
    expect(within(dialog).getByText("Models loaded")).toBeVisible();

    await user.click(within(dialog).getByRole("button", { name: "Refresh diagnostics" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("Diagnostics unavailable.");
    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog", { name: "System diagnostics" })).not.toBeInTheDocument();
  });

  it("clears only after success and downloads exports without navigation", async () => {
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
    });
    const user = userEvent.setup();
    let downloadName = "";
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      downloadName = this.download;
    });
    render(<App api={api} />);
    await screen.findByText("alpha.pdf");
    await user.type(screen.getByLabelText("Ask about your documents"), "What changed?");
    await user.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText("The policy changed in July.");

    await user.click(screen.getByRole("button", { name: "Export conversation" }));
    await waitFor(() => expect(api.exportConversation).toHaveBeenCalledTimes(1));
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
    expect(downloadName).toBe("conversation-session.json");

    await user.click(screen.getByRole("button", { name: "Clear conversation" }));
    await waitFor(() => expect(api.clearConversation).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("The policy changed in July.")).not.toBeInTheDocument();
  });

  it("disables conflicting controls and explains the unconnected benchmark boundary", async () => {
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
      getDocuments: vi.fn().mockResolvedValue({
        ...documentList,
        active_operation: busyRuntime.active_operation,
      }),
    });
    render(<App api={api} />);

    expect(await screen.findByRole("button", { name: "Send question" })).toBeDisabled();
    expect(screen.getByLabelText("Upload documents")).toBeDisabled();
    const benchmark = screen.getByRole("button", { name: "Run benchmark" });
    expect(benchmark).toBeDisabled();
    expect(screen.getByText(/benchmark workflow is not connected/i)).toBeVisible();
  });

  it("disables conflicting controls immediately while a local query is in flight", async () => {
    const pendingQuery = deferred<QueryResponse>();
    const api = createMockApi({
      getDocuments: vi.fn().mockResolvedValue(documentList),
      query: vi.fn().mockReturnValue(pendingQuery.promise),
    });
    const user = userEvent.setup();
    render(<App api={api} />);
    await screen.findByText("alpha.pdf");

    await user.type(screen.getByLabelText("Ask about your documents"), "Hold this query");
    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(screen.getByRole("button", { name: "Send question" })).toBeDisabled();
    expect(screen.getByLabelText("Upload documents")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Clear conversation" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Export conversation" })).toBeDisabled();

    await act(async () => {
      pendingQuery.resolve(queryResponse);
      await pendingQuery.promise;
    });
  });

  it("invokes the optional benchmark boundary only when capability allows it", async () => {
    const onRunBenchmark = vi.fn();
    const user = userEvent.setup();
    render(<App api={createMockApi()} onRunBenchmark={onRunBenchmark} />);

    await user.click(await screen.findByRole("button", { name: "Run benchmark" }));

    expect(onRunBenchmark).toHaveBeenCalledTimes(1);
  });

  it("explains when runtime capability disables a connected benchmark boundary", async () => {
    const api = createMockApi({
      getRuntime: vi.fn().mockResolvedValue({
        ...runtimeReady,
        capabilities: {
          ...runtimeReady.capabilities,
          can_run_benchmark: false,
        },
      }),
    });
    render(<App api={api} onRunBenchmark={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "Run benchmark" })).toBeDisabled();
    expect(screen.getByText(/runtime does not currently allow a benchmark/i)).toBeVisible();
  });

  it("closes a dialog on cancel and backdrop interaction", async () => {
    const user = userEvent.setup();
    render(<App api={createMockApi()} />);
    await user.click(await screen.findByRole("button", { name: "System diagnostics" }));
    const dialog = await screen.findByRole("dialog", { name: "System diagnostics" });

    fireEvent(dialog, new Event("cancel", { cancelable: true }));
    expect(screen.queryByRole("dialog", { name: "System diagnostics" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "System diagnostics" }));
    const reopened = await screen.findByRole("dialog", { name: "System diagnostics" });
    fireEvent.click(reopened);
    expect(screen.queryByRole("dialog", { name: "System diagnostics" })).not.toBeInTheDocument();
  });

  it("gives the mobile drawer focus, Escape, and covered-content semantics", async () => {
    const user = userEvent.setup();
    render(<App api={createMockApi()} />);
    const opener = await screen.findByRole("button", { name: "Open workspace controls" });
    opener.focus();

    await user.click(opener);

    const close = screen.getByRole("button", { name: "Close workspace controls" });
    expect(close).toHaveFocus();
    expect(screen.getByRole("main", { hidden: true })).toHaveAttribute("inert");
    expect(screen.getByRole("main", { hidden: true })).toHaveAttribute("aria-hidden", "true");

    await user.keyboard("{Escape}");

    expect(screen.getByRole("complementary", { name: "Workspace controls" })).not.toHaveClass(
      "sidebar--open",
    );
    expect(screen.getByRole("main")).not.toHaveAttribute("inert");
    expect(screen.getByRole("main")).not.toHaveAttribute("aria-hidden");
    expect(opener).toHaveFocus();
  });
});
