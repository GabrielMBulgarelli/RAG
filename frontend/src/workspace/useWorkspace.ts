import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { DownloadFile, WorkspaceApi } from "../api/client";
import type {
  ActiveOperation,
  DiagnosticsSnapshot,
  DocumentList,
  OperationKind,
  QueryResponse,
  RuntimeSnapshot,
} from "../api/types";
import { getSessionId } from "../session";

export interface Exchange {
  id: string;
  question: string;
  createdAt: string;
  response: QueryResponse | null;
  error: string | null;
  pending: boolean;
}

export type WorkspaceOperationKind =
  | OperationKind
  | "clear_conversation"
  | "export_conversation";

interface OperationToken {
  id: symbol;
  kind: WorkspaceOperationKind;
  generation: number;
  api: WorkspaceApi;
}

interface DiagnosticsToken {
  id: symbol;
  generation: number;
  api: WorkspaceApi;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The workspace request failed.";
}

function downloadFile({ blob, filename }: DownloadFile): void {
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

export function useWorkspace(api: WorkspaceApi) {
  const [runtime, setRuntime] = useState<RuntimeSnapshot | null>(null);
  const [documentList, setDocumentList] = useState<DocumentList | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsSnapshot | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [selectedExchangeId, setSelectedExchangeId] = useState<string | null>(null);
  const [selectedSourceLabel, setSelectedSourceLabel] = useState<string | null>(null);
  const [loadingWorkspace, setLoadingWorkspace] = useState(true);
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [localOperation, setLocalOperation] = useState<WorkspaceOperationKind | null>(null);
  const sessionId = useMemo(() => getSessionId(), []);

  const mountedRef = useRef(false);
  const apiRef = useRef(api);
  const generationRef = useRef(0);
  const operationRef = useRef<OperationToken | null>(null);
  const diagnosticsRef = useRef<DiagnosticsToken | null>(null);
  const serverOperationRef = useRef<ActiveOperation | null>(null);

  if (apiRef.current !== api) {
    apiRef.current = api;
    generationRef.current += 1;
    operationRef.current = null;
    diagnosticsRef.current = null;
  }

  const serverOperation = runtime?.active_operation ?? documentList?.active_operation ?? null;
  serverOperationRef.current = serverOperation;

  const isCurrent = useCallback((generation: number, requestApi: WorkspaceApi) => (
    mountedRef.current
    && generationRef.current === generation
    && apiRef.current === requestApi
  ), []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      operationRef.current = null;
      diagnosticsRef.current = null;
    };
  }, []);

  useEffect(() => {
    setRuntime(null);
    setDocumentList(null);
    setDiagnostics(null);
    setExchanges([]);
    setSelectedExchangeId(null);
    setSelectedSourceLabel(null);
    setLoadingWorkspace(true);
    setLocalOperation(null);
    setDiagnosticsLoading(false);
    setWorkspaceError(null);
    setActionError(null);
    serverOperationRef.current = null;
  }, [api]);

  const refreshWorkspace = useCallback(async (
    generation: number,
    requestApi: WorkspaceApi,
  ): Promise<boolean> => {
    const [nextRuntime, nextDocuments] = await Promise.all([
      requestApi.getRuntime(),
      requestApi.getDocuments(),
    ]);
    if (!isCurrent(generation, requestApi)) {
      return false;
    }
    setRuntime(nextRuntime);
    setDocumentList(nextDocuments);
    return true;
  }, [isCurrent]);

  useEffect(() => {
    let cancelled = false;
    const generation = generationRef.current;
    const requestApi = api;
    setLoadingWorkspace(true);
    setWorkspaceError(null);
    Promise.all([requestApi.getRuntime(), requestApi.getDocuments()])
      .then(([nextRuntime, nextDocuments]) => {
        if (!cancelled && isCurrent(generation, requestApi)) {
          setRuntime(nextRuntime);
          setDocumentList(nextDocuments);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled && isCurrent(generation, requestApi)) {
          setWorkspaceError(errorMessage(error));
        }
      })
      .finally(() => {
        if (!cancelled && isCurrent(generation, requestApi)) {
          setLoadingWorkspace(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [api, isCurrent]);

  const beginOperation = useCallback((kind: WorkspaceOperationKind): OperationToken | null => {
    if (!mountedRef.current || operationRef.current || serverOperationRef.current) {
      return null;
    }
    const token = {
      id: Symbol(kind),
      kind,
      generation: generationRef.current,
      api,
    };
    operationRef.current = token;
    setLocalOperation(kind);
    setActionError(null);
    return token;
  }, [api]);

  const finishOperation = useCallback((token: OperationToken) => {
    if (operationRef.current?.id !== token.id) {
      return;
    }
    operationRef.current = null;
    if (isCurrent(token.generation, token.api)) {
      setLocalOperation(null);
    }
  }, [isCurrent]);

  const readDiagnostics = useCallback(async (
    force: boolean,
    clearStaleOnError: boolean,
  ): Promise<boolean> => {
    if (!mountedRef.current) {
      return false;
    }
    if (diagnosticsRef.current && !force) {
      return false;
    }
    const token = {
      id: Symbol("diagnostics"),
      generation: generationRef.current,
      api,
    };
    diagnosticsRef.current = token;
    setDiagnosticsLoading(true);
    setActionError(null);
    try {
      const nextDiagnostics = await api.getDiagnostics();
      if (
        diagnosticsRef.current?.id !== token.id
        || !isCurrent(token.generation, token.api)
      ) {
        return false;
      }
      setDiagnostics(nextDiagnostics);
      return true;
    } catch (error) {
      if (
        diagnosticsRef.current?.id === token.id
        && isCurrent(token.generation, token.api)
      ) {
        if (clearStaleOnError) {
          setDiagnostics(null);
        }
        setActionError(errorMessage(error));
      }
      return false;
    } finally {
      if (diagnosticsRef.current?.id === token.id) {
        diagnosticsRef.current = null;
        if (isCurrent(token.generation, token.api)) {
          setDiagnosticsLoading(false);
        }
      }
    }
  }, [api, isCurrent]);

  const runQuery = useCallback(async (
    token: OperationToken,
    exchangeId: string,
    question: string,
  ) => {
    try {
      const response = await token.api.query(sessionId, question);
      if (!isCurrent(token.generation, token.api)) {
        return;
      }
      setExchanges((current) => current.map((exchange) => (
        exchange.id === exchangeId
          ? { ...exchange, pending: false, response, error: null }
          : exchange
      )));
      setSelectedExchangeId(exchangeId);
      setSelectedSourceLabel(response.sources[0]?.label ?? null);
    } catch (error) {
      if (isCurrent(token.generation, token.api)) {
        setExchanges((current) => current.map((exchange) => (
          exchange.id === exchangeId
            ? { ...exchange, pending: false, error: errorMessage(error) }
            : exchange
        )));
      }
    } finally {
      finishOperation(token);
    }
  }, [finishOperation, isCurrent, sessionId]);

  const submitQuestion = useCallback(async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) {
      return;
    }
    const token = beginOperation("query");
    if (!token) {
      return;
    }
    const exchangeId = crypto.randomUUID();
    setExchanges((current) => [...current, {
      id: exchangeId,
      question: trimmed,
      createdAt: new Date().toISOString(),
      response: null,
      error: null,
      pending: true,
    }]);
    await runQuery(token, exchangeId, trimmed);
  }, [beginOperation, runQuery]);

  const retryQuestion = useCallback(async (exchangeId: string) => {
    const exchange = exchanges.find((candidate) => candidate.id === exchangeId);
    if (!exchange) {
      return;
    }
    const token = beginOperation("query");
    if (!token) {
      return;
    }
    if (isCurrent(token.generation, token.api)) {
      setExchanges((current) => current.map((candidate) => (
        candidate.id === exchangeId
          ? { ...candidate, pending: true, error: null }
          : candidate
      )));
    }
    await runQuery(token, exchange.id, exchange.question);
  }, [beginOperation, exchanges, isCurrent, runQuery]);

  const runWorkspaceMutation = useCallback(async (
    kind: "index_documents" | "delete_document",
    mutate: (requestApi: WorkspaceApi) => Promise<unknown>,
  ) => {
    const token = beginOperation(kind);
    if (!token) {
      return false;
    }
    try {
      await mutate(token.api);
      if (!isCurrent(token.generation, token.api)) {
        return false;
      }
      return await refreshWorkspace(token.generation, token.api);
    } catch (error) {
      if (isCurrent(token.generation, token.api)) {
        setActionError(errorMessage(error));
      }
      return false;
    } finally {
      finishOperation(token);
    }
  }, [beginOperation, finishOperation, isCurrent, refreshWorkspace]);

  const uploadDocuments = useCallback(async (files: File[]) => {
    if (files.length === 0) {
      return false;
    }
    return runWorkspaceMutation(
      "index_documents",
      (requestApi) => requestApi.uploadDocuments(files),
    );
  }, [runWorkspaceMutation]);

  const deleteDocument = useCallback(async (documentId: string) => {
    return runWorkspaceMutation(
      "delete_document",
      (requestApi) => requestApi.deleteDocument(documentId),
    );
  }, [runWorkspaceMutation]);

  const loadModel = useCallback(async (chatModel: string) => {
    const token = beginOperation("load_model");
    if (!token) {
      return false;
    }
    try {
      await token.api.loadModel(chatModel);
      if (!isCurrent(token.generation, token.api)) {
        return false;
      }
      if (!await refreshWorkspace(token.generation, token.api)) {
        return false;
      }
      return await readDiagnostics(true, true);
    } catch (error) {
      if (isCurrent(token.generation, token.api)) {
        setDiagnostics(null);
        setActionError(errorMessage(error));
      }
      return false;
    } finally {
      finishOperation(token);
    }
  }, [
    beginOperation,
    finishOperation,
    isCurrent,
    readDiagnostics,
    refreshWorkspace,
  ]);

  const refreshDiagnostics = useCallback(
    () => readDiagnostics(false, false),
    [readDiagnostics],
  );

  const clearConversation = useCallback(async () => {
    const token = beginOperation("clear_conversation");
    if (!token) {
      return false;
    }
    try {
      await token.api.clearConversation(sessionId);
      if (!isCurrent(token.generation, token.api)) {
        return false;
      }
      setExchanges([]);
      setSelectedExchangeId(null);
      setSelectedSourceLabel(null);
      return true;
    } catch (error) {
      if (isCurrent(token.generation, token.api)) {
        setActionError(errorMessage(error));
      }
      return false;
    } finally {
      finishOperation(token);
    }
  }, [beginOperation, finishOperation, isCurrent, sessionId]);

  const exportConversation = useCallback(async () => {
    const token = beginOperation("export_conversation");
    if (!token) {
      return false;
    }
    try {
      const exported = await token.api.exportConversation(sessionId);
      if (!isCurrent(token.generation, token.api)) {
        return false;
      }
      downloadFile(exported);
      return true;
    } catch (error) {
      if (isCurrent(token.generation, token.api)) {
        setActionError(errorMessage(error));
      }
      return false;
    } finally {
      finishOperation(token);
    }
  }, [beginOperation, finishOperation, isCurrent, sessionId]);

  const selectedExchange = exchanges.find(
    (exchange) => exchange.id === selectedExchangeId,
  ) ?? [...exchanges].reverse().find((exchange) => exchange.response) ?? null;
  const busyKind = localOperation ?? serverOperation?.kind ?? null;

  return {
    runtime,
    documentList,
    diagnostics,
    exchanges,
    selectedExchange,
    selectedSourceLabel,
    loadingWorkspace,
    diagnosticsLoading,
    workspaceError,
    actionError,
    activeOperation: serverOperation,
    busy: busyKind !== null,
    busyKind,
    setSelectedExchangeId,
    setSelectedSourceLabel,
    submitQuestion,
    retryQuestion,
    uploadDocuments,
    deleteDocument,
    loadModel,
    refreshDiagnostics,
    clearConversation,
    exportConversation,
  };
}

export type WorkspaceController = ReturnType<typeof useWorkspace>;
