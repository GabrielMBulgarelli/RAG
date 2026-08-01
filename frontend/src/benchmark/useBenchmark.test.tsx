import { StrictMode, type PropsWithChildren } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClientError, type WorkspaceApi } from "../api/client";
import type {
  BenchmarkCaseDetail,
  BenchmarkEvent,
  BenchmarkRun,
  BenchmarkStartResponse,
} from "../api/types";
import {
  benchmarkCase,
  benchmarkRun,
  benchmarkRunId,
  benchmarkStart,
  createMockApi,
} from "../test/fixtures";
import { useBenchmark } from "./useBenchmark";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function runningRun(overrides: Partial<BenchmarkRun> = {}): BenchmarkRun {
  return {
    ...benchmarkRun,
    status: "running",
    metadata: {
      ...benchmarkRun.metadata,
      completed_at: null,
    },
    ...overrides,
  };
}

function event(eventId: number, type = "case.completed"): BenchmarkEvent {
  return {
    event_id: eventId,
    run_id: benchmarkRunId,
    type: type as BenchmarkEvent["type"],
    timestamp: "2026-07-29T15:31:00Z",
    data: {},
  };
}

function waitForAbort(signal: AbortSignal): Promise<void> {
  return new Promise((_, reject) => {
    signal.addEventListener("abort", () => {
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

describe("benchmark controller", () => {
  beforeEach(() => {
    vi.mocked(URL.createObjectURL).mockClear();
  });

  it("loads the latest completed run and resets loading after success", async () => {
    const latest = deferred<BenchmarkRun>();
    const api = createMockApi({
      getLatestBenchmark: vi.fn().mockReturnValue(latest.promise),
    });
    const hook = renderHook(() => useBenchmark(api));
    let loaded = false;

    act(() => {
      void hook.result.current.loadLatest().then((result) => {
        loaded = result;
      });
    });
    expect(hook.result.current.latestLoading).toBe(true);

    await act(async () => {
      latest.resolve(benchmarkRun);
      await latest.promise;
    });

    expect(loaded).toBe(true);
    expect(api.getLatestBenchmark).toHaveBeenCalledTimes(1);
    expect(hook.result.current.run).toEqual(benchmarkRun);
    expect(hook.result.current.latestLoading).toBe(false);
    expect(hook.result.current.latestError).toBeNull();
  });

  it("treats a missing latest run as an empty result", async () => {
    const api = createMockApi({
      getLatestBenchmark: vi.fn().mockRejectedValue(new ApiClientError({
        code: "benchmark_not_found",
        message: "Benchmark not found.",
        details: {},
      }, 404)),
    });
    const hook = renderHook(() => useBenchmark(api));

    let loaded = true;
    await act(async () => {
      loaded = await hook.result.current.loadLatest();
    });

    expect(loaded).toBe(false);
    expect(hook.result.current.run).toBeNull();
    expect(hook.result.current.latestLoading).toBe(false);
    expect(hook.result.current.latestError).toBeNull();
  });

  it("reports a latest-run server failure and resets loading", async () => {
    const api = createMockApi({
      getLatestBenchmark: vi.fn().mockRejectedValue(new ApiClientError({
        code: "runtime_unavailable",
        message: "Stored benchmark results are unavailable.",
        details: {},
      }, 503)),
    });
    const hook = renderHook(() => useBenchmark(api));

    let loaded = true;
    await act(async () => {
      loaded = await hook.result.current.loadLatest();
    });

    expect(loaded).toBe(false);
    expect(hook.result.current.latestLoading).toBe(false);
    expect(hook.result.current.latestError).toBe(
      "Stored benchmark results are unavailable.",
    );
  });

  it("retains the latest run for case inspection and download", async () => {
    const api = createMockApi();
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      expect(await hook.result.current.loadLatest()).toBe(true);
    });
    await act(async () => {
      expect(await hook.result.current.openCase(
        benchmarkRun.run_id,
        benchmarkCase.case_id,
        benchmarkCase.system,
      )).toBe(true);
      expect(await hook.result.current.download()).toBe(true);
    });

    expect(api.getBenchmarkCase).toHaveBeenCalledWith(
      benchmarkRun.run_id,
      benchmarkCase.case_id,
      benchmarkCase.system,
    );
    expect(hook.result.current.caseDetail).toEqual(benchmarkCase);
    expect(api.downloadBenchmark).toHaveBeenCalledWith(benchmarkRun.run_id);
  });

  it("starts atomically, publishes durable progress, and begins replay at zero", async () => {
    const start = deferred<BenchmarkStartResponse>();
    const active = runningRun();
    const api = createMockApi({
      startBenchmark: vi.fn().mockReturnValue(start.promise),
      getBenchmark: vi.fn().mockResolvedValue(active),
      streamBenchmarkEvents: vi.fn((
        _runId: string,
        _lastId: number,
        signal: AbortSignal,
      ) => waitForAbort(signal)),
    });
    const hook = renderHook(() => useBenchmark(api));

    act(() => {
      void hook.result.current.start();
      void hook.result.current.start();
    });

    expect(api.startBenchmark).toHaveBeenCalledTimes(1);
    expect(hook.result.current.startInFlight).toBe(true);
    expect(hook.result.current.busy).toBe(true);

    await act(async () => {
      start.resolve(benchmarkStart);
      await start.promise;
    });
    await waitFor(() => expect(hook.result.current.run?.status).toBe("running"));

    expect(api.getBenchmark).toHaveBeenCalledWith(benchmarkRunId);
    expect(api.streamBenchmarkEvents).toHaveBeenCalledWith(
      benchmarkRunId,
      0,
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(hook.result.current.busy).toBe(true);
    hook.unmount();
  });

  it("coalesces an event burst into serialized durable refreshes", async () => {
    const firstRefresh = deferred<BenchmarkRun>();
    let activeGets = 0;
    let maxActiveGets = 0;
    const getBenchmark = vi.fn()
      .mockResolvedValueOnce(runningRun())
      .mockImplementationOnce(async () => {
        activeGets += 1;
        maxActiveGets = Math.max(maxActiveGets, activeGets);
        const value = await firstRefresh.promise;
        activeGets -= 1;
        return value;
      })
      .mockImplementationOnce(async () => {
        activeGets += 1;
        maxActiveGets = Math.max(maxActiveGets, activeGets);
        activeGets -= 1;
        return benchmarkRun;
      });
    const api = createMockApi({
      getBenchmark,
      streamBenchmarkEvents: vi.fn((
        _runId: string,
        _lastId: number,
        signal: AbortSignal,
        onEvent: (next: BenchmarkEvent) => void,
      ) => {
        onEvent(event(1));
        onEvent(event(2));
        onEvent(event(3));
        return waitForAbort(signal);
      }),
    });
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      await hook.result.current.start();
    });
    await waitFor(() => expect(getBenchmark).toHaveBeenCalledTimes(2));
    expect(activeGets).toBe(1);

    await act(async () => {
      firstRefresh.resolve(runningRun({
        progress: { ...benchmarkRun.progress, completed_cases: 3 },
      }));
      await firstRefresh.promise;
    });

    await waitFor(() => expect(hook.result.current.run?.status).toBe("completed"));
    expect(getBenchmark).toHaveBeenCalledTimes(3);
    expect(maxActiveGets).toBe(1);
    expect(hook.result.current.lastEventId).toBe(3);
    expect(hook.result.current.busy).toBe(false);
  });

  it("reconnects a clean nonterminal close and stops after a terminal snapshot", async () => {
    const stream = vi.fn()
      .mockResolvedValueOnce(undefined)
      .mockImplementationOnce(async (
        _runId: string,
        _lastId: number,
        _signal: AbortSignal,
        onEvent: (next: BenchmarkEvent) => void,
      ) => {
        onEvent(event(1, "benchmark.completed"));
      });
    const api = createMockApi({
      getBenchmark: vi.fn()
        .mockResolvedValueOnce(runningRun())
        .mockResolvedValueOnce(runningRun())
        .mockResolvedValueOnce(benchmarkRun),
      streamBenchmarkEvents: stream,
    });
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      await hook.result.current.start();
    });

    await waitFor(() => expect(hook.result.current.run?.status).toBe("completed"), {
      timeout: 2_000,
    });
    expect(stream).toHaveBeenCalledTimes(2);
    expect(stream.mock.calls[0]?.[1]).toBe(0);
    expect(stream.mock.calls[1]?.[1]).toBe(0);
    expect(hook.result.current.connectionError).toBeNull();
  });

  it("stops retrying after repeated clean nonterminal stream closures", async () => {
    const api = createMockApi({
      getBenchmark: vi.fn().mockResolvedValue(runningRun()),
      streamBenchmarkEvents: vi.fn().mockResolvedValue(undefined),
    });
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      await hook.result.current.start();
    });
    await waitFor(() => {
      expect(hook.result.current.connectionError).toBe(
        "The benchmark event stream closed repeatedly.",
      );
    }, { timeout: 2_000 });

    expect(api.streamBenchmarkEvents).toHaveBeenCalledTimes(3);
    expect(hook.result.current.busy).toBe(true);
  });

  it("resets the clean-close retry budget after an explicit retry", async () => {
    const api = createMockApi({
      getBenchmark: vi.fn().mockResolvedValue(runningRun()),
      streamBenchmarkEvents: vi.fn().mockResolvedValue(undefined),
    });
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      await hook.result.current.start();
    });
    await waitFor(() => {
      expect(hook.result.current.connectionError).toBe(
        "The benchmark event stream closed repeatedly.",
      );
    }, { timeout: 2_000 });
    expect(api.streamBenchmarkEvents).toHaveBeenCalledTimes(3);

    act(() => hook.result.current.retryConnection());

    await waitFor(() => expect(api.streamBenchmarkEvents).toHaveBeenCalledTimes(6), {
      timeout: 2_000,
    });
    expect(hook.result.current.connectionError).toBe(
      "The benchmark event stream closed repeatedly.",
    );
  });

  it("retains the durable run on stream error and exposes an explicit retry", async () => {
    const api = createMockApi({
      getBenchmark: vi.fn().mockResolvedValue(runningRun()),
      streamBenchmarkEvents: vi.fn()
        .mockRejectedValueOnce(new Error("Connection lost."))
        .mockImplementationOnce((
          _runId: string,
          _lastId: number,
          signal: AbortSignal,
        ) => waitForAbort(signal)),
    });
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      await hook.result.current.start();
    });
    await waitFor(() => expect(hook.result.current.connectionError).toBe("Connection lost."));

    expect(hook.result.current.run?.status).toBe("running");
    expect(hook.result.current.busy).toBe(true);
    act(() => hook.result.current.retryConnection());
    await waitFor(() => expect(api.streamBenchmarkEvents).toHaveBeenCalledTimes(2));
    expect(hook.result.current.connectionError).toBeNull();
    hook.unmount();
  });

  it("invalidates old case and API continuations", async () => {
    const oldCase = deferred<BenchmarkCaseDetail>();
    const firstApi = createMockApi({
      getBenchmarkCase: vi.fn().mockReturnValue(oldCase.promise),
    });
    const secondCase = { ...benchmarkCase, case_id: "case-2" };
    const secondApi = createMockApi({
      getBenchmarkCase: vi.fn().mockResolvedValue(secondCase),
    });
    const hook = renderHook(
      ({ api }: { api: WorkspaceApi }) => useBenchmark(api),
      { initialProps: { api: firstApi } },
    );

    act(() => {
      void hook.result.current.openCase(benchmarkRunId, "case-1", "dense");
    });
    hook.rerender({ api: secondApi });
    await act(async () => {
      await hook.result.current.openCase(benchmarkRunId, "case-2", "full-rag");
    });
    await act(async () => {
      oldCase.resolve(benchmarkCase);
      await oldCase.promise;
    });

    expect(hook.result.current.caseDetail?.case_id).toBe("case-2");
    expect(hook.result.current.caseError).toBeNull();
  });

  it("invalidates an in-flight case request when a new run starts", async () => {
    const oldCase = deferred<BenchmarkCaseDetail>();
    const api = createMockApi({
      getBenchmarkCase: vi.fn().mockReturnValue(oldCase.promise),
    });
    const hook = renderHook(() => useBenchmark(api));

    act(() => {
      void hook.result.current.openCase(benchmarkRunId, "case-1", "dense");
    });
    expect(hook.result.current.caseLoading).toBe(true);

    await act(async () => {
      await hook.result.current.start();
    });
    expect(hook.result.current.caseLoading).toBe(false);

    await act(async () => {
      oldCase.resolve(benchmarkCase);
      await oldCase.promise;
    });
    expect(hook.result.current.caseDetail).toBeNull();
    expect(hook.result.current.caseError).toBeNull();
  });

  it("publishes cancellation_requested once and exposes cancel failures", async () => {
    const cancellation = deferred<BenchmarkRun>();
    const active = runningRun();
    const api = createMockApi({
      getBenchmark: vi.fn().mockResolvedValue(active),
      cancelBenchmark: vi.fn().mockReturnValueOnce(cancellation.promise)
        .mockRejectedValueOnce(new Error("Cancel refused.")),
      streamBenchmarkEvents: vi.fn((
        _runId: string,
        _lastId: number,
        signal: AbortSignal,
      ) => waitForAbort(signal)),
    });
    const hook = renderHook(() => useBenchmark(api));
    await act(async () => hook.result.current.start());

    act(() => {
      void hook.result.current.cancel();
      void hook.result.current.cancel();
    });
    expect(api.cancelBenchmark).toHaveBeenCalledTimes(1);
    expect(hook.result.current.cancelInFlight).toBe(true);

    await act(async () => {
      cancellation.resolve({ ...active, status: "cancellation_requested" });
      await cancellation.promise;
    });
    expect(hook.result.current.run?.status).toBe("cancellation_requested");
    expect(hook.result.current.cancelInFlight).toBe(false);

    await act(async () => {
      await hook.result.current.cancel();
    });
    expect(hook.result.current.cancelError).toBe("Cancel refused.");
    expect(hook.result.current.run?.status).toBe("cancellation_requested");
    hook.unmount();
  });

  it("loads and clears case success and error states", async () => {
    const api = createMockApi({
      getBenchmarkCase: vi.fn()
        .mockResolvedValueOnce(benchmarkCase)
        .mockRejectedValueOnce(new Error("Case unavailable.")),
    });
    const hook = renderHook(() => useBenchmark(api));

    await act(async () => {
      await hook.result.current.openCase(benchmarkRunId, "case-1", "dense");
    });
    expect(hook.result.current.caseDetail).toEqual(benchmarkCase);
    expect(hook.result.current.caseLoading).toBe(false);

    await act(async () => {
      await hook.result.current.openCase(benchmarkRunId, "missing", "bm25");
    });
    expect(hook.result.current.caseDetail).toBeNull();
    expect(hook.result.current.caseError).toBe("Case unavailable.");
    act(() => hook.result.current.closeCase());
    expect(hook.result.current.caseError).toBeNull();
  });

  it("downloads the durable run with a safe filename and retains a safe error", async () => {
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const api = createMockApi({
      downloadBenchmark: vi.fn()
        .mockResolvedValueOnce({
          blob: new Blob(["result"], { type: "application/json" }),
          filename: "safe-results.json",
        })
        .mockRejectedValueOnce(new Error("Download unavailable.")),
    });
    const hook = renderHook(() => useBenchmark(api));
    await act(async () => hook.result.current.start());

    await act(async () => {
      await hook.result.current.download();
    });
    expect(api.downloadBenchmark).toHaveBeenCalledWith(benchmarkRunId);
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(anchorClick).toHaveBeenCalledTimes(1);

    await act(async () => {
      await hook.result.current.download();
    });
    expect(hook.result.current.downloadError).toBe("Download unavailable.");
    anchorClick.mockRestore();
  });

  it("makes download atomic while a request is in flight", async () => {
    const pendingDownload = deferred<{ blob: Blob; filename: string }>();
    const api = createMockApi({
      downloadBenchmark: vi.fn().mockReturnValue(pendingDownload.promise),
    });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const hook = renderHook(() => useBenchmark(api));
    await act(async () => hook.result.current.start());

    act(() => {
      void hook.result.current.download();
      void hook.result.current.download();
    });
    expect(api.downloadBenchmark).toHaveBeenCalledTimes(1);
    expect(hook.result.current.downloadInFlight).toBe(true);

    await act(async () => {
      pendingDownload.resolve({
        blob: new Blob(["result"]),
        filename: "benchmark.zip",
      });
      await pendingDownload.promise;
    });
    expect(hook.result.current.downloadInFlight).toBe(false);
    click.mockRestore();
  });

  it("aborts the active stream on a StrictMode unmount", async () => {
    let signal: AbortSignal | undefined;
    const api = createMockApi({
      getBenchmark: vi.fn().mockResolvedValue(runningRun()),
      streamBenchmarkEvents: vi.fn((
        _runId: string,
        _lastId: number,
        nextSignal: AbortSignal,
      ) => {
        signal = nextSignal;
        return waitForAbort(nextSignal);
      }),
    });
    const wrapper = ({ children }: PropsWithChildren) => (
      <StrictMode>{children}</StrictMode>
    );
    const hook = renderHook(() => useBenchmark(api), { wrapper });
    await act(async () => hook.result.current.start());
    expect(signal?.aborted).toBe(false);

    hook.unmount();
    expect(signal?.aborted).toBe(true);
  });

  it("aborts and ignores stale stream and cancel work on API replacement", async () => {
    const signals: AbortSignal[] = [];
    const oldCancel = deferred<BenchmarkRun>();
    const firstApi = createMockApi({
      getBenchmark: vi.fn().mockResolvedValue(runningRun()),
      cancelBenchmark: vi.fn().mockReturnValue(oldCancel.promise),
      streamBenchmarkEvents: vi.fn((
        _runId: string,
        _lastId: number,
        signal: AbortSignal,
      ) => {
        signals.push(signal);
        return waitForAbort(signal);
      }),
    });
    const secondApi = createMockApi();
    const hook = renderHook(
      ({ api }: { api: WorkspaceApi }) => useBenchmark(api),
      { initialProps: { api: firstApi } },
    );
    await act(async () => hook.result.current.start());
    expect(signals).toHaveLength(1);

    act(() => {
      void hook.result.current.cancel();
    });
    hook.rerender({ api: secondApi });
    await waitFor(() => expect(signals[0]?.aborted).toBe(true));
    await act(async () => {
      oldCancel.resolve({ ...benchmarkRun, status: "cancelled" });
      await oldCancel.promise;
    });
    expect(hook.result.current.run).toBeNull();
    expect(hook.result.current.cancelError).toBeNull();
    expect(hook.result.current.busy).toBe(false);
    hook.unmount();
  });

  it("ignores a stale download continuation after run supersession", async () => {
    const oldDownload = deferred<{ blob: Blob; filename: string }>();
    const secondRunId = "5ba22a5a-71c8-4f02-a07e-9b0cd92bb7cf";
    const firstCompleted = { ...benchmarkRun };
    const secondCompleted = { ...benchmarkRun, run_id: secondRunId };
    const api = createMockApi({
      startBenchmark: vi.fn()
        .mockResolvedValueOnce(benchmarkStart)
        .mockResolvedValueOnce({ ...benchmarkStart, run_id: secondRunId }),
      getBenchmark: vi.fn()
        .mockResolvedValueOnce(runningRun())
        .mockResolvedValueOnce(firstCompleted)
        .mockResolvedValueOnce(secondCompleted),
      downloadBenchmark: vi.fn().mockReturnValue(oldDownload.promise),
      streamBenchmarkEvents: vi.fn((
        _runId: string,
        _lastId: number,
        _signal: AbortSignal,
        onEvent: (next: BenchmarkEvent) => void,
      ) => {
        onEvent(event(1, "benchmark.completed"));
        return Promise.resolve();
      }),
    });
    const hook = renderHook(() => useBenchmark(api));
    await act(async () => hook.result.current.start());
    await waitFor(() => expect(hook.result.current.run?.status).toBe("completed"));

    act(() => {
      void hook.result.current.download();
    });
    await act(async () => hook.result.current.start());
    expect(hook.result.current.run?.run_id).toBe(secondRunId);

    await act(async () => {
      oldDownload.resolve({
        blob: new Blob(["old"]),
        filename: "old.json",
      });
      await oldDownload.promise;
    });
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(hook.result.current.run?.run_id).toBe(secondRunId);
  });
});
