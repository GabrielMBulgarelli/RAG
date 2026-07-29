import {
  useEffect,
  useState,
  type DragEvent,
  type RefObject,
} from "react";

import type { DocumentRecord } from "../api/types";
import type { WorkspaceController } from "./useWorkspace";

interface SidebarProps {
  workspace: WorkspaceController;
  open: boolean;
  closeButtonRef: RefObject<HTMLButtonElement | null>;
  onClose: () => void;
  onDocumentDetails: (document: DocumentRecord) => void;
  onDiagnostics: () => void;
  onRunBenchmark?: () => void;
}

export function Sidebar({
  workspace,
  open,
  closeButtonRef,
  onClose,
  onDocumentDetails,
  onDiagnostics,
  onRunBenchmark,
}: SidebarProps) {
  const runtime = workspace.runtime;
  const documents = workspace.documentList?.documents ?? [];
  const busy = workspace.busy;
  const uploadDisabled = busy || !runtime?.capabilities.can_upload;
  const [model, setModel] = useState("");

  useEffect(() => {
    if (runtime) {
      setModel(runtime.active_chat_model ?? runtime.configured_chat_model);
    }
  }, [runtime]);

  const upload = (files: FileList | null) => {
    if (!files || uploadDisabled) {
      return;
    }
    void workspace.uploadDocuments(Array.from(files));
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    upload(event.dataTransfer.files);
  };

  const benchmarkReason = !onRunBenchmark
    ? "Benchmark workflow is not connected in this workspace."
    : !runtime
      ? "Checking whether the runtime can run a benchmark."
      : !runtime.capabilities.can_run_benchmark
        ? "The runtime does not currently allow a benchmark."
        : busy
          ? "Wait for the active workspace operation before running a benchmark."
          : null;
  const benchmarkAvailable = benchmarkReason === null;
  const indexedLabel = `${documents.length} indexed document${documents.length === 1 ? "" : "s"}`;

  return (
    <aside
      id="workspace-sidebar"
      className={`sidebar${open ? " sidebar--open" : ""}`}
      aria-label="Workspace controls"
    >
      <div className="sidebar__mobile-heading">
        <strong>Workspace controls</strong>
            <button
              ref={closeButtonRef}
              className="icon-button"
              type="button"
              onClick={onClose}
              aria-label="Close workspace controls"
            >
          ×
        </button>
      </div>

      <section className="sidebar-section" aria-labelledby="documents-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Corpus</p>
            <h2 id="documents-heading">Documents</h2>
          </div>
          <span className="count-badge">{documents.length}</span>
        </div>

        <div
          className={`drop-zone${uploadDisabled ? " drop-zone--disabled" : ""}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
        >
          <label htmlFor="document-upload">Drop PDF or text files</label>
          <span>or choose from this device</span>
          <input
            id="document-upload"
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            multiple
            disabled={uploadDisabled}
            aria-label="Upload documents"
            onChange={(event) => {
              upload(event.currentTarget.files);
              event.currentTarget.value = "";
            }}
          />
        </div>

        <p className="compact-stat">{indexedLabel}</p>
        {workspace.documentList ? (
          <p className="compact-stat">
            {workspace.documentList.corpus.page_count} pages · {workspace.documentList.corpus.chunk_count} chunks
          </p>
        ) : null}
        <div className="document-list" aria-label="Indexed documents">
          {documents.length === 0 ? (
            <p className="muted-copy">No indexed files.</p>
          ) : documents.map((document) => (
            <button
              key={document.id}
              className="document-row"
              type="button"
              onClick={() => onDocumentDetails(document)}
              aria-label={`${document.filename} details`}
            >
              <span className="document-row__name">{document.filename}</span>
              <span>{document.page_count}p · {document.chunk_count}c</span>
            </button>
          ))}
        </div>
      </section>

      <section className="sidebar-section" aria-labelledby="runtime-heading">
        <p className="eyebrow">Local runtime</p>
        <h2 id="runtime-heading">Models & index</h2>
        <dl className="runtime-list">
          <div>
            <dt>Chat</dt>
            <dd>{runtime?.active_chat_model ?? "Not loaded"}</dd>
          </div>
          <div>
            <dt>Embedding</dt>
            <dd>{runtime?.embedding_model ?? "Checking…"}</dd>
          </div>
          <div>
            <dt>Corpus</dt>
            <dd>{workspace.documentList?.corpus.status ?? "Checking…"}</dd>
          </div>
        </dl>

        {runtime && runtime.state !== "ready" && runtime.capabilities.can_load_models ? (
          <div className="model-loader">
            <label htmlFor="model-choice">Chat model</label>
            <select
              id="model-choice"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              disabled={busy}
            >
              {runtime.available_chat_models.map((availableModel) => (
                <option key={availableModel} value={availableModel}>{availableModel}</option>
              ))}
            </select>
            <button
              className="button button--quiet"
              type="button"
              disabled={busy || !model}
              onClick={() => void workspace.loadModel(model)}
            >
              Load models
            </button>
          </div>
        ) : null}
      </section>

      <section className="sidebar-section sidebar-section--actions" aria-labelledby="actions-heading">
        <p className="eyebrow">Actions</p>
        <h2 id="actions-heading">Workspace</h2>
        <button
          className="button button--secondary"
          type="button"
          disabled={!benchmarkAvailable}
          aria-describedby={benchmarkReason ? "benchmark-reason" : undefined}
          onClick={onRunBenchmark}
        >
          Run benchmark
        </button>
        {benchmarkReason ? (
          <p id="benchmark-reason" className="control-note">{benchmarkReason}</p>
        ) : null}
        <button className="button button--quiet" type="button" onClick={onDiagnostics}>
          System diagnostics
        </button>
      </section>

      {workspace.actionError ? (
        <p className="inline-error" role="alert">{workspace.actionError}</p>
      ) : null}
    </aside>
  );
}
