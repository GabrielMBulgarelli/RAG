import { useState } from "react";

import { createApiClient, type WorkspaceApi } from "./api/client";
import type { DocumentRecord } from "./api/types";
import { ConversationPanel } from "./workspace/ConversationPanel";
import { InspectorPanel } from "./workspace/InspectorPanel";
import {
  OverlayController,
  type OverlayState,
} from "./workspace/OverlayController";
import { Sidebar } from "./workspace/Sidebar";
import { useWorkspace } from "./workspace/useWorkspace";

const defaultApi = createApiClient();

interface AppProps {
  api?: WorkspaceApi;
  onRunBenchmark?: () => void;
}

export function App({ api = defaultApi, onRunBenchmark }: AppProps) {
  const workspace = useWorkspace(api);
  const [overlay, setOverlay] = useState<OverlayState>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const runtimeState = workspace.runtime?.state ?? "not_loaded";
  const readiness = workspace.runtime?.active_chat_model ?? "Runtime not loaded";

  const openDiagnostics = () => {
    setOverlay({ kind: "diagnostics" });
    void workspace.refreshDiagnostics();
  };

  const openDocumentDetails = (document: DocumentRecord) => {
    setOverlay({ kind: "document-details", document });
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <button
          className="icon-button mobile-menu"
          type="button"
          aria-label="Open workspace controls"
          aria-controls="workspace-sidebar"
          aria-expanded={sidebarOpen}
          onClick={() => setSidebarOpen(true)}
        >
          ☰
        </button>
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
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onDocumentDetails={openDocumentDetails}
          onDiagnostics={openDiagnostics}
          onRunBenchmark={onRunBenchmark}
        />
        {sidebarOpen ? (
          <button
            className="sidebar-backdrop"
            type="button"
            aria-label="Close workspace controls"
            onClick={() => setSidebarOpen(false)}
          />
        ) : null}
        <main className="workbench">
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
        onClose={() => setOverlay(null)}
        onRequestDelete={(document) => setOverlay({ kind: "delete-confirm", document })}
      />
    </div>
  );
}
