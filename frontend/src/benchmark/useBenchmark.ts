import { useCallback, useEffect, useRef, useState } from "react";

import { ApiClientError, type DownloadFile, type WorkspaceApi } from "../api/client";
import type {
  BenchmarkCaseDetail,
  BenchmarkEvent,
  BenchmarkRun,
} from "../api/types";

type ConnectionState = "idle" | "connecting" | "connected" | "error";

interface RunToken {
  id: symbol;
  generation: number;
  api: WorkspaceApi;
  runId: string;
  consecutiveCleanCloses: number;
}

interface RefreshState {
  tokenId: symbol;
  queued: boolean;
  promise: Promise<BenchmarkRun | null> | null;
}

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const RECONNECT_DELAY_MS = 50;
const MAX_CONSECUTIVE_CLEAN_CLOSES = 3;

function isTerminal(run: BenchmarkRun | null): boolean {
  return Boolean(run && TERMINAL_STATUSES.has(run.status));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The benchmark request failed.";
}

function saveDownload({ blob, filename }: DownloadFile): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function reconnectDelay(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(resolve, RECONNECT_DELAY_MS);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timeout);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

export function useBenchmark(api: WorkspaceApi) {
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [startInFlight, setStartInFlight] = useState(false);
  const [cancelInFlight, setCancelInFlight] = useState(false);
  const [downloadInFlight, setDownloadInFlight] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [lastEventId, setLastEventId] = useState(0);
  const [startError, setStartError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [caseDetail, setCaseDetail] = useState<BenchmarkCaseDetail | null>(null);
  const [caseLoading, setCaseLoading] = useState(false);
  const [caseError, setCaseError] = useState<string | null>(null);
  const [latestLoading, setLatestLoading] = useState(false);
  const [latestError, setLatestError] = useState<string | null>(null);

  const mountedRef = useRef(false);
  const apiRef = useRef(api);
  const generationRef = useRef(0);
  const runRef = useRef<BenchmarkRun | null>(null);
  const tokenRef = useRef<RunToken | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const startRef = useRef(false);
  const cancelRef = useRef(false);
  const lastEventIdRef = useRef(0);
  const refreshRef = useRef<RefreshState | null>(null);
  const caseTokenRef = useRef<symbol | null>(null);
  const downloadTokenRef = useRef<symbol | null>(null);
  const downloadInFlightRef = useRef(false);
  const latestTokenRef = useRef<symbol | null>(null);
  const connectRef = useRef<(token: RunToken) => void>(() => undefined);

  const isCurrent = useCallback((token: RunToken): boolean => (
    mountedRef.current
    && apiRef.current === token.api
    && generationRef.current === token.generation
    && tokenRef.current?.id === token.id
  ), []);

  const publishRun = useCallback((token: RunToken, nextRun: BenchmarkRun) => {
    if (!isCurrent(token)) {
      return false;
    }
    runRef.current = nextRun;
    setRun(nextRun);
    if (isTerminal(nextRun)) {
      streamAbortRef.current?.abort();
      streamAbortRef.current = null;
      setConnectionState("idle");
      setConnectionError(null);
    }
    return true;
  }, [isCurrent]);

  const failConnection = useCallback((token: RunToken, error: unknown) => {
    if (!isCurrent(token)) {
      return;
    }
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    setConnectionState("error");
    setConnectionError(errorMessage(error));
  }, [isCurrent]);

  const refreshDurable = useCallback((token: RunToken): Promise<BenchmarkRun | null> => {
    if (!isCurrent(token)) {
      return Promise.resolve(null);
    }
    let state = refreshRef.current;
    if (!state || state.tokenId !== token.id) {
      state = { tokenId: token.id, queued: false, promise: null };
      refreshRef.current = state;
    }
    if (state.promise) {
      state.queued = true;
      return state.promise;
    }
    const execute = async (): Promise<BenchmarkRun | null> => {
      let latest: BenchmarkRun | null = null;
      do {
        state.queued = false;
        const nextRun = await token.api.getBenchmark(token.runId);
        if (!publishRun(token, nextRun)) {
          return null;
        }
        latest = nextRun;
      } while (state.queued && !isTerminal(latest));
      return latest;
    };
    state.promise = execute()
      .catch((error: unknown) => {
        failConnection(token, error);
        return null;
      })
      .finally(() => {
        if (refreshRef.current === state) {
          state.promise = null;
        }
      });
    return state.promise;
  }, [failConnection, isCurrent, publishRun]);

  const connect = useCallback((token: RunToken) => {
    if (!isCurrent(token) || isTerminal(runRef.current)) {
      return;
    }
    streamAbortRef.current?.abort();
    const abort = new AbortController();
    streamAbortRef.current = abort;
    setConnectionState("connecting");
    setConnectionError(null);

    const accept = (event: BenchmarkEvent) => {
      if (
        !isCurrent(token)
        || event.run_id !== token.runId
        || event.event_id <= lastEventIdRef.current
      ) {
        return;
      }
      lastEventIdRef.current = event.event_id;
      token.consecutiveCleanCloses = 0;
      setLastEventId(event.event_id);
      setConnectionState("connected");
      void refreshDurable(token);
    };

    void token.api.streamBenchmarkEvents(
      token.runId,
      lastEventIdRef.current,
      abort.signal,
      accept,
    ).then(async () => {
      if (!isCurrent(token) || abort.signal.aborted) {
        return;
      }
      const latest = await refreshDurable(token);
      if (!latest || isTerminal(latest) || !isCurrent(token)) {
        return;
      }
      token.consecutiveCleanCloses += 1;
      if (token.consecutiveCleanCloses >= MAX_CONSECUTIVE_CLEAN_CLOSES) {
        failConnection(token, new Error("The benchmark event stream closed repeatedly."));
        return;
      }
      await reconnectDelay(abort.signal);
      if (isCurrent(token) && !abort.signal.aborted) {
        connectRef.current(token);
      }
    }).catch((error: unknown) => {
      if (
        error instanceof DOMException
        && error.name === "AbortError"
      ) {
        return;
      }
      failConnection(token, error);
    });
  }, [failConnection, isCurrent, refreshDurable]);
  connectRef.current = connect;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      streamAbortRef.current?.abort();
      caseTokenRef.current = null;
      downloadTokenRef.current = null;
      downloadInFlightRef.current = false;
      latestTokenRef.current = null;
    };
  }, []);

  useEffect(() => {
    apiRef.current = api;
    generationRef.current += 1;
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    tokenRef.current = null;
    refreshRef.current = null;
    startRef.current = false;
    cancelRef.current = false;
    lastEventIdRef.current = 0;
    runRef.current = null;
    caseTokenRef.current = null;
    downloadTokenRef.current = null;
    downloadInFlightRef.current = false;
    latestTokenRef.current = null;
    setRun(null);
    setStartInFlight(false);
    setCancelInFlight(false);
    setDownloadInFlight(false);
    setConnectionState("idle");
    setLastEventId(0);
    setStartError(null);
    setConnectionError(null);
    setCancelError(null);
    setDownloadError(null);
    setCaseDetail(null);
    setCaseLoading(false);
    setCaseError(null);
    setLatestLoading(false);
    setLatestError(null);
    return () => {
      streamAbortRef.current?.abort();
    };
  }, [api]);

  const start = useCallback(async (): Promise<boolean> => {
    if (
      startRef.current
      || latestTokenRef.current
      || (runRef.current && !isTerminal(runRef.current))
    ) {
      return false;
    }
    startRef.current = true;
    setStartInFlight(true);
    setStartError(null);
    setConnectionError(null);
    setCancelError(null);
    setDownloadError(null);
    setLatestError(null);
    tokenRef.current = null;
    refreshRef.current = null;
    caseTokenRef.current = null;
    setCaseDetail(null);
    setCaseLoading(false);
    setCaseError(null);
    downloadTokenRef.current = null;
    downloadInFlightRef.current = false;
    setDownloadInFlight(false);
    streamAbortRef.current?.abort();
    runRef.current = null;
    setRun(null);
    lastEventIdRef.current = 0;
    setLastEventId(0);

    const generation = generationRef.current;
    const requestApi = api;
    try {
      const started = await requestApi.startBenchmark();
      if (
        !mountedRef.current
        || generationRef.current !== generation
        || apiRef.current !== requestApi
      ) {
        return false;
      }
      const token = {
        id: Symbol(started.run_id),
        generation,
        api: requestApi,
        runId: started.run_id,
        consecutiveCleanCloses: 0,
      };
      tokenRef.current = token;
      const durable = await requestApi.getBenchmark(started.run_id);
      if (!publishRun(token, durable)) {
        return false;
      }
      if (!isTerminal(durable)) {
        connectRef.current(token);
      }
      return true;
    } catch (error) {
      if (
        mountedRef.current
        && generationRef.current === generation
        && apiRef.current === requestApi
      ) {
        setStartError(errorMessage(error));
      }
      return false;
    } finally {
      if (
        mountedRef.current
        && generationRef.current === generation
        && apiRef.current === requestApi
      ) {
        startRef.current = false;
        setStartInFlight(false);
      }
    }
  }, [api, publishRun]);

  const loadLatest = useCallback(async (): Promise<boolean> => {
    if (
      latestTokenRef.current
      || startRef.current
      || (runRef.current && !isTerminal(runRef.current))
    ) {
      return false;
    }
    const requestToken = Symbol("latest-benchmark");
    const generation = generationRef.current;
    const requestApi = api;
    latestTokenRef.current = requestToken;
    setLatestLoading(true);
    setLatestError(null);
    setStartError(null);
    setConnectionError(null);
    setCancelError(null);
    setDownloadError(null);
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    tokenRef.current = null;
    refreshRef.current = null;
    caseTokenRef.current = null;
    setCaseDetail(null);
    setCaseLoading(false);
    setCaseError(null);
    downloadTokenRef.current = null;
    downloadInFlightRef.current = false;
    setDownloadInFlight(false);
    runRef.current = null;
    setRun(null);

    const isLatestRequest = () => (
      mountedRef.current
      && generationRef.current === generation
      && apiRef.current === requestApi
      && latestTokenRef.current === requestToken
    );

    try {
      const latest = await requestApi.getLatestBenchmark();
      if (!isLatestRequest()) {
        return false;
      }
      runRef.current = latest;
      setRun(latest);
      return true;
    } catch (error) {
      if (isLatestRequest() && !(error instanceof ApiClientError && error.status === 404)) {
        setLatestError(errorMessage(error));
      }
      return false;
    } finally {
      if (isLatestRequest()) {
        latestTokenRef.current = null;
        setLatestLoading(false);
      }
    }
  }, [api]);

  const retryConnection = useCallback(() => {
    const token = tokenRef.current;
    if (!token || !isCurrent(token) || isTerminal(runRef.current)) {
      return;
    }
    token.consecutiveCleanCloses = 0;
    setConnectionError(null);
    connectRef.current(token);
  }, [isCurrent]);

  const cancel = useCallback(async (): Promise<boolean> => {
    const token = tokenRef.current;
    if (!token || !isCurrent(token) || isTerminal(runRef.current) || cancelRef.current) {
      return false;
    }
    cancelRef.current = true;
    setCancelInFlight(true);
    setCancelError(null);
    try {
      const nextRun = await token.api.cancelBenchmark(token.runId);
      return publishRun(token, nextRun);
    } catch (error) {
      if (isCurrent(token)) {
        setCancelError(errorMessage(error));
      }
      return false;
    } finally {
      if (isCurrent(token)) {
        cancelRef.current = false;
        setCancelInFlight(false);
      }
    }
  }, [isCurrent, publishRun]);

  const openCase = useCallback(async (
    runId: string,
    caseId: string,
    systemId: string,
  ): Promise<boolean> => {
    const token = Symbol(`${caseId}:${systemId}`);
    const generation = generationRef.current;
    const requestApi = api;
    caseTokenRef.current = token;
    setCaseDetail(null);
    setCaseError(null);
    setCaseLoading(true);
    try {
      const detail = await requestApi.getBenchmarkCase(runId, caseId, systemId);
      if (
        !mountedRef.current
        || generationRef.current !== generation
        || apiRef.current !== requestApi
        || caseTokenRef.current !== token
      ) {
        return false;
      }
      setCaseDetail(detail);
      return true;
    } catch (error) {
      if (
        mountedRef.current
        && generationRef.current === generation
        && apiRef.current === requestApi
        && caseTokenRef.current === token
      ) {
        setCaseError(errorMessage(error));
      }
      return false;
    } finally {
      if (
        mountedRef.current
        && generationRef.current === generation
        && apiRef.current === requestApi
        && caseTokenRef.current === token
      ) {
        setCaseLoading(false);
      }
    }
  }, [api]);

  const closeCase = useCallback(() => {
    caseTokenRef.current = null;
    setCaseDetail(null);
    setCaseError(null);
    setCaseLoading(false);
  }, []);

  const download = useCallback(async (): Promise<boolean> => {
    const currentRun = runRef.current;
    if (!currentRun || downloadInFlightRef.current) {
      return false;
    }
    const token = Symbol("download");
    const generation = generationRef.current;
    const requestApi = api;
    const runId = currentRun.run_id;
    downloadTokenRef.current = token;
    downloadInFlightRef.current = true;
    setDownloadInFlight(true);
    setDownloadError(null);
    try {
      const file = await requestApi.downloadBenchmark(runId);
      if (
        !mountedRef.current
        || generationRef.current !== generation
        || apiRef.current !== requestApi
        || downloadTokenRef.current !== token
        || runRef.current?.run_id !== runId
      ) {
        return false;
      }
      saveDownload(file);
      return true;
    } catch (error) {
      if (
        mountedRef.current
        && generationRef.current === generation
        && apiRef.current === requestApi
        && downloadTokenRef.current === token
      ) {
        setDownloadError(errorMessage(error));
      }
      return false;
    } finally {
      if (
        mountedRef.current
        && generationRef.current === generation
        && apiRef.current === requestApi
        && downloadTokenRef.current === token
      ) {
        downloadInFlightRef.current = false;
        setDownloadInFlight(false);
      }
    }
  }, [api]);

  const active = Boolean(run && !isTerminal(run));
  const busy = startInFlight || latestLoading || active;

  return {
    run,
    startInFlight,
    cancelInFlight,
    connectionState,
    lastEventId,
    startError,
    connectionError,
    cancelError,
    downloadError,
    downloadInFlight,
    caseDetail,
    caseLoading,
    caseError,
    latestLoading,
    latestError,
    busy,
    start,
    loadLatest,
    retryConnection,
    cancel,
    openCase,
    closeCase,
    download,
  };
}

export type BenchmarkController = Omit<
  ReturnType<typeof useBenchmark>,
  "latestLoading" | "latestError" | "loadLatest"
>;
