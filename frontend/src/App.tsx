import { useEffect, useRef, useState } from "react";

import { createApiClient, type WorkspaceApi } from "./api/client";
import type { DocumentRecord } from "./api/types";
import { useBenchmark } from "./benchmark/useBenchmark";
import { ConversationPanel } from "./workspace/ConversationPanel";
import { InspectorPanel } from "./workspace/InspectorPanel";
import {
  OverlayController,
  type OverlayState,
} from "./workspace/OverlayController";
import { Sidebar } from "./workspace/Sidebar";
import { useWorkspace } from "./workspace/useWorkspace";

const defaultApi = createApiClient();
const COMPACT_WORKSPACE_QUERY = "(max-width: 860px)";

interface AppProps {
  api?: WorkspaceApi;
  onRunBenchmark?: () => void;
}

function useCompactWorkspace(): boolean {
  const [compact, setCompact] = useState(
    () => window.matchMedia(COMPACT_WORKSPACE_QUERY).matches,
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia(COMPACT_WORKSPACE_QUERY);
    const update = (event: MediaQueryListEvent) => setCompact(event.matches);
    setCompact(mediaQuery.matches);
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);

  return compact;
}

export function App({ api = defaultApi, onRunBenchmark }: AppProps) {
  const benchmark = useBenchmark(api);
  const workspace = useWorkspace(api, benchmark.busy ? "benchmark" : null);
  const [overlay, setOverlay] = useState<OverlayState>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const overlayBeforeDiagnosticsRef = useRef<OverlayState>(null);
  const sidebarOpenerRef = useRef<HTMLButtonElement>(null);
  const sidebarCloseRef = useRef<HTMLButtonElement>(null);
  const sidebarWasOpenRef = useRef(false);
  const compactWorkspace = useCompactWorkspace();
  const drawerOpen = compactWorkspace && sidebarOpen;
  const runtimeState = workspace.runtime?.state ?? "not_loaded";
  const readiness = workspace.runtime?.active_chat_model ?? "Runtime not loaded";

  const openDiagnostics = () => {
    setOverlay((current) => {
      overlayBeforeDiagnosticsRef.current = current?.kind === "benchmark-progress"
        ? current
        : null;
      return { kind: "diagnostics" };
    });
    void workspace.refreshDiagnostics();
  };

  const closeOverlay = () => {
    setOverlay((current) => {
      const previous = current?.kind === "diagnostics"
        ? overlayBeforeDiagnosticsRef.current
        : null;
      overlayBeforeDiagnosticsRef.current = null;
      if (previous?.kind === "benchmark-progress") {
        return benchmark.run?.status === "completed"
          ? { kind: "benchmark-results" }
          : previous;
      }
      return null;
    });
  };

  const openDocumentDetails = (document: DocumentRecord) => {
    setOverlay({ kind: "document-details", document });
  };

  const runBenchmark = () => {
    if (onRunBenchmark) {
      onRunBenchmark();
      return;
    }
    setOverlay({ kind: "benchmark-progress" });
    void benchmark.start();
  };

  useEffect(() => {
    if (
      benchmark.run?.status === "completed"
      && overlay?.kind === "benchmark-progress"
    ) {
      setOverlay({ kind: "benchmark-results" });
    }
  }, [benchmark.run?.status, overlay?.kind]);

  useEffect(() => {
    if (!drawerOpen) {
      if (sidebarWasOpenRef.current) {
        sidebarWasOpenRef.current = false;
        if (compactWorkspace) {
          sidebarOpenerRef.current?.focus();
        }
      }
      return;
    }
    sidebarWasOpenRef.current = true;
    sidebarCloseRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setSidebarOpen(false);
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [compactWorkspace, drawerOpen]);

  useEffect(() => {
    if (!compactWorkspace) {
      setSidebarOpen(false);
    }
  }, [compactWorkspace]);

  return (
    <div className="app-shell">
      <header className="app-header">
        {compactWorkspace ? (
          <button
            ref={sidebarOpenerRef}
            className="icon-button mobile-menu"
            type="button"
            aria-label="Open workspace controls"
            aria-controls="workspace-sidebar"
            aria-expanded={drawerOpen}
            onClick={() => setSidebarOpen(true)}
          >
            ☰
          </button>
        ) : null}
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">R</span>
          <div>
            <p>Local evidence instrument</p>
            <h1>Local Document RAG</h1>
          </div>
        </div>
        <div className="readiness" aria-label={`Runtime ${runtimeState}`}>
          <span className={`state-dot state-dot--${runtimeState}`} aria-hidden="true" />
          <div>
            <strong>{runtimeState === "ready" ? "Ready" : "Needs attention"}</strong>
            <span>{readiness}</span>
          </div>
        </div>
      </header>

      <div className="workspace-layout">
        <Sidebar
          workspace={workspace}
          compact={compactWorkspace}
          open={drawerOpen}
          closeButtonRef={sidebarCloseRef}
          onClose={() => setSidebarOpen(false)}
          onDocumentDetails={openDocumentDetails}
          onDiagnostics={openDiagnostics}
          onRunBenchmark={runBenchmark}
        />
        {drawerOpen ? (
          <button
            className="sidebar-backdrop"
            type="button"
            aria-label="Dismiss workspace controls"
            onClick={() => setSidebarOpen(false)}
          />
        ) : null}
        <main
          className="workbench"
          inert={drawerOpen || undefined}
          aria-hidden={drawerOpen || undefined}
        >
          <ConversationPanel workspace={workspace} />
          <InspectorPanel
            workspace={workspace}
            collapsed={inspectorCollapsed}
            onCollapse={setInspectorCollapsed}
          />
        </main>
      </div>

      <OverlayController
        overlay={overlay}
        workspace={workspace}
        benchmark={benchmark}
        onClose={closeOverlay}
        onRequestDelete={(document) => setOverlay({ kind: "delete-confirm", document })}
        onCaseRefChange={(caseRef) => {
          setOverlay((current) => (
            current?.kind === "benchmark-results"
              ? { kind: "benchmark-results", caseRef }
              : current
          ));
        }}
      />
    </div>
  );
}
