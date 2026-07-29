import { useCallback, useEffect, useMemo, useState } from "react";

import type { WorkspaceApi } from "../api/client";
import type {
  DiagnosticsSnapshot,
  DocumentList,
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The workspace request failed.";
}

function downloadBlob(blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "rag-conversation.md";
  anchor.click();
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
  const sessionId = useMemo(() => getSessionId(), []);

  const refreshWorkspace = useCallback(async () => {
    const [nextRuntime, nextDocuments] = await Promise.all([
      api.getRuntime(),
      api.getDocuments(),
    ]);
    setRuntime(nextRuntime);
    setDocumentList(nextDocuments);
  }, [api]);

  useEffect(() => {
    let active = true;
    setLoadingWorkspace(true);
    setWorkspaceError(null);
    Promise.all([api.getRuntime(), api.getDocuments()])
      .then(([nextRuntime, nextDocuments]) => {
        if (active) {
          setRuntime(nextRuntime);
          setDocumentList(nextDocuments);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setWorkspaceError(errorMessage(error));
        }
      })
      .finally(() => {
        if (active) {
          setLoadingWorkspace(false);
        }
      });
    return () => {
      active = false;
    };
  }, [api]);

  const runQuery = useCallback(async (exchangeId: string, question: string) => {
    setExchanges((current) => current.map((exchange) => (
      exchange.id === exchangeId
        ? { ...exchange, pending: true, error: null }
        : exchange
    )));
    try {
      const response = await api.query(sessionId, question);
      setExchanges((current) => current.map((exchange) => (
        exchange.id === exchangeId
          ? { ...exchange, pending: false, response, error: null }
          : exchange
      )));
      setSelectedExchangeId(exchangeId);
      setSelectedSourceLabel(response.sources[0]?.label ?? null);
    } catch (error) {
      setExchanges((current) => current.map((exchange) => (
        exchange.id === exchangeId
          ? { ...exchange, pending: false, error: errorMessage(error) }
          : exchange
      )));
    }
  }, [api, sessionId]);

  const submitQuestion = useCallback(async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) {
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
    await runQuery(exchangeId, trimmed);
  }, [runQuery]);

  const retryQuestion = useCallback(async (exchangeId: string) => {
    const exchange = exchanges.find((candidate) => candidate.id === exchangeId);
    if (exchange) {
      await runQuery(exchange.id, exchange.question);
    }
  }, [exchanges, runQuery]);

  const uploadDocuments = useCallback(async (files: File[]) => {
    if (files.length === 0) {
      return;
    }
    setActionError(null);
    try {
      await api.uploadDocuments(files);
      await refreshWorkspace();
    } catch (error) {
      setActionError(errorMessage(error));
    }
  }, [api, refreshWorkspace]);

  const deleteDocument = useCallback(async (documentId: string) => {
    setActionError(null);
    try {
      await api.deleteDocument(documentId);
      await refreshWorkspace();
      return true;
    } catch (error) {
      setActionError(errorMessage(error));
      return false;
    }
  }, [api, refreshWorkspace]);

  const loadModel = useCallback(async (chatModel: string) => {
    setActionError(null);
    try {
      await api.loadModel(chatModel);
      await refreshWorkspace();
      return true;
    } catch (error) {
      setActionError(errorMessage(error));
      return false;
    }
  }, [api, refreshWorkspace]);

  const refreshDiagnostics = useCallback(async () => {
    setDiagnosticsLoading(true);
    setActionError(null);
    try {
      setDiagnostics(await api.getDiagnostics());
      return true;
    } catch (error) {
      setActionError(errorMessage(error));
      return false;
    } finally {
      setDiagnosticsLoading(false);
    }
  }, [api]);

  const clearConversation = useCallback(async () => {
    setActionError(null);
    try {
      await api.clearConversation(sessionId);
      setExchanges([]);
      setSelectedExchangeId(null);
      setSelectedSourceLabel(null);
    } catch (error) {
      setActionError(errorMessage(error));
    }
  }, [api, sessionId]);

  const exportConversation = useCallback(async () => {
    setActionError(null);
    try {
      downloadBlob(await api.exportConversation(sessionId));
    } catch (error) {
      setActionError(errorMessage(error));
    }
  }, [api, sessionId]);

  const selectedExchange = exchanges.find(
    (exchange) => exchange.id === selectedExchangeId,
  ) ?? [...exchanges].reverse().find((exchange) => exchange.response) ?? null;
  const activeOperation = runtime?.active_operation ?? documentList?.active_operation ?? null;

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
    activeOperation,
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
