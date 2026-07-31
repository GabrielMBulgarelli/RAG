import { useEffect, useRef, type ReactNode } from "react";

import type { DiagnosticCheck, DocumentRecord } from "../api/types";
import { BenchmarkProgress } from "../benchmark/BenchmarkProgress";
import {
  BenchmarkResults,
  type BenchmarkCaseRef,
} from "../benchmark/BenchmarkResults";
import type { BenchmarkController } from "../benchmark/useBenchmark";
import type { WorkspaceController } from "./useWorkspace";

export type OverlayState =
  | { kind: "diagnostics" }
  | { kind: "document-details"; document: DocumentRecord }
  | { kind: "delete-confirm"; document: DocumentRecord }
  | { kind: "benchmark-progress" }
  | { kind: "benchmark-results"; caseRef?: BenchmarkCaseRef }
  | null;

interface DialogFrameProps {
  title: string;
  children: ReactNode;
  onClose: () => void;
  className?: string;
}

function DialogFrame({
  title,
  children,
  onClose,
  className = "",
}: DialogFrameProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) {
      dialog.showModal();
    }
    return () => {
      if (dialog?.open) {
        dialog.close();
      }
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className={`dialog ${className}`}
      aria-label={title}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      {children}
    </dialog>
  );
}

interface OverlayControllerProps {
  overlay: OverlayState;
  workspace: WorkspaceController;
  benchmark: BenchmarkController;
  onClose: () => void;
  onRequestDelete: (document: DocumentRecord) => void;
  onCaseRefChange: (caseRef: BenchmarkCaseRef | undefined) => void;
}

function CheckGroup({
  title,
  checks,
}: {
  title: string;
  checks: DiagnosticCheck[];
}) {
  return (
    <section className="check-group">
      <h3>{title}</h3>
      <div>
        {checks.map((check) => (
          <article key={`${title}-${check.name}`} className="check-row">
            <span className={`state-dot state-dot--${check.state}`} aria-hidden="true" />
            <div>
              <strong>{check.name}</strong>
              <p>{check.detail}</p>
            </div>
            <span className="state-label">{check.state.replaceAll("_", " ")}</span>
          </article>
        ))}
      </div>
    </section>
  );
}

export function OverlayController({
  overlay,
  workspace,
  benchmark,
  onClose,
  onRequestDelete,
  onCaseRefChange,
}: OverlayControllerProps) {
  if (!overlay) {
    return null;
  }

  if (overlay.kind === "benchmark-progress") {
    return <BenchmarkProgress benchmark={benchmark} onClose={onClose} />;
  }

  if (overlay.kind === "benchmark-results") {
    return (
      <BenchmarkResults
        benchmark={benchmark}
        caseRef={overlay.caseRef}
        onCaseRefChange={onCaseRefChange}
        onClose={onClose}
      />
    );
  }

  if (overlay.kind === "diagnostics") {
    const runtime = workspace.runtime;
    return (
      <DialogFrame title="System diagnostics" onClose={onClose} className="dialog--wide">
        <header className="dialog__header">
          <div>
            <p className="eyebrow">Local system</p>
            <h2>System diagnostics</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close diagnostics dialog">×</button>
        </header>

        <div className="dialog__body">
          {workspace.diagnosticsLoading && !workspace.diagnostics ? (
            <p role="status">Reading diagnostics…</p>
          ) : workspace.diagnostics ? (
            <>
              <div className="diagnostics-summary">
                <span className={`state-dot state-dot--${workspace.diagnostics.state}`} aria-hidden="true" />
                <div>
                  <strong>{workspace.diagnostics.title}</strong>
                  <p>{workspace.diagnostics.detail}</p>
                </div>
              </div>
              <div className="check-columns">
                <CheckGroup title="Runtime" checks={workspace.diagnostics.runtime_checks} />
                <CheckGroup title="Index" checks={workspace.diagnostics.index_checks} />
                <CheckGroup title="Evaluation" checks={workspace.diagnostics.evaluation_checks} />
              </div>
            </>
          ) : (
            <p className="muted-copy">Diagnostics have not been read yet.</p>
          )}
          {workspace.actionError ? (
            <p className="inline-error" role="alert">{workspace.actionError}</p>
          ) : null}
        </div>

        <footer className="dialog__footer">
          {runtime && runtime.state !== "ready" && runtime.capabilities.can_load_models ? (
            <button
              className="button button--secondary"
              type="button"
              disabled={workspace.busy}
              onClick={() => void workspace.loadModel(runtime.configured_chat_model)}
            >
              Load models
            </button>
          ) : null}
          <button
            className="button button--quiet"
            type="button"
            disabled={workspace.diagnosticsLoading || workspace.busy}
            onClick={() => void workspace.refreshDiagnostics()}
          >
            Refresh diagnostics
          </button>
          <button className="button button--primary" type="button" onClick={onClose}>
            Close
          </button>
        </footer>
      </DialogFrame>
    );
  }

  if (overlay.kind === "document-details") {
    const document = overlay.document;
    return (
      <DialogFrame title="Document details" onClose={onClose}>
        <header className="dialog__header">
          <div>
            <p className="eyebrow">Indexed document</p>
            <h2>Document details</h2>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close document details">×</button>
        </header>
        <div className="dialog__body">
          <h3 className="document-title">{document.filename}</h3>
          <dl className="document-metadata">
            <div><dt>Status</dt><dd>{document.state}</dd></div>
            <div><dt>Pages</dt><dd>{document.page_count} pages</dd></div>
            <div><dt>Chunks</dt><dd>{document.chunk_count} chunks</dd></div>
            <div><dt>Size</dt><dd>{Math.max(1, Math.round(document.size_bytes / 1024))} KB</dd></div>
            <div><dt>Indexed</dt><dd>{document.indexed_at ? new Date(document.indexed_at).toLocaleString() : "Not indexed"}</dd></div>
          </dl>
        </div>
        <footer className="dialog__footer dialog__footer--split">
          <button
            className="button button--danger"
            type="button"
            disabled={workspace.busy}
            onClick={() => onRequestDelete(document)}
          >
            Delete document
          </button>
          <button className="button button--primary" type="button" onClick={onClose}>Close</button>
        </footer>
      </DialogFrame>
    );
  }

  const document = overlay.document;
  return (
    <DialogFrame title={`Delete ${document.filename}?`} onClose={onClose}>
      <header className="dialog__header">
        <div>
          <p className="eyebrow">Permanent action</p>
          <h2>Delete {document.filename}?</h2>
        </div>
      </header>
      <div className="dialog__body">
        <p>
          This removes <strong>{document.filename}</strong> and its indexed chunks from the local corpus.
        </p>
        {workspace.actionError ? (
          <p className="inline-error" role="alert">{workspace.actionError}</p>
        ) : null}
      </div>
      <footer className="dialog__footer">
        <button className="button button--quiet" type="button" onClick={onClose}>Cancel</button>
        <button
          className="button button--danger"
          type="button"
          disabled={workspace.busy}
          onClick={async () => {
            if (await workspace.deleteDocument(document.id)) {
              onClose();
            }
          }}
        >
          Delete
        </button>
      </footer>
    </DialogFrame>
  );
}
