import { useEffect, useState } from "react";

import type { WorkspaceController } from "./useWorkspace";

interface InspectorPanelProps {
  workspace: WorkspaceController;
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
}

export function InspectorPanel({
  workspace,
  collapsed,
  onCollapse,
}: InspectorPanelProps) {
  const [tab, setTab] = useState<"sources" | "details">("sources");
  const response = workspace.selectedExchange?.response ?? null;

  useEffect(() => {
    if (workspace.selectedSourceLabel) {
      setTab("sources");
    }
  }, [workspace.selectedSourceLabel]);

  if (collapsed) {
    return (
      <aside className="inspector inspector--collapsed" aria-label="Evidence inspector">
        <button
          className="inspector-expand"
          type="button"
          onClick={() => onCollapse(false)}
          aria-label="Expand inspector"
        >
          Evidence
        </button>
      </aside>
    );
  }

  return (
    <aside className="inspector" aria-labelledby="inspector-heading">
      <header className="inspector__header">
        <div>
          <p className="eyebrow">Evidence spine</p>
          <h2 id="inspector-heading">Answer context</h2>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={() => onCollapse(true)}
          aria-label="Collapse inspector"
        >
          →
        </button>
      </header>

      <div className="tabs" role="tablist" aria-label="Inspector views">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "sources"}
          onClick={() => setTab("sources")}
        >
          Sources
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "details"}
          onClick={() => setTab("details")}
        >
          Details
        </button>
      </div>

      <div className="inspector__content">
        {!response ? (
          <div className="inspector-empty">
            <span aria-hidden="true">01</span>
            <p>Select an answer to inspect its sources and execution details.</p>
          </div>
        ) : tab === "sources" ? (
          <div className="source-list">
            {response.sources.length === 0 ? (
              <p className="muted-copy">No sources accompanied this answer.</p>
            ) : response.sources.map((source) => {
              const selected = workspace.selectedSourceLabel === source.label;
              return (
                <button
                  key={source.label}
                  className={`source-card${selected ? " source-card--selected" : ""}`}
                  type="button"
                  aria-label={`Source ${source.label}${selected ? ", selected" : ""}`}
                  aria-current={selected ? "true" : undefined}
                  onClick={() => workspace.setSelectedSourceLabel(source.label)}
                >
                  <span className="source-card__index">{source.label}</span>
                  <span className="source-card__body">
                    <strong>{source.filename}</strong>
                    <span>{source.page === null ? "Page unavailable" : `Page ${source.page}`}</span>
                    <q>{source.excerpt}</q>
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="details-list">
            <details>
              <summary>Retrieval</summary>
              <div className="details-content">
                {response.retrieval_hits.map((hit) => (
                  <dl key={hit.chunk_id} className="metric-grid">
                    <div><dt>Chunk</dt><dd>{hit.chunk_id}</dd></div>
                    <div><dt>Selected</dt><dd>{hit.selection_score ?? "—"}</dd></div>
                    <div><dt>Fused</dt><dd>{hit.fused_score ?? "—"}</dd></div>
                    <div><dt>Page</dt><dd>{hit.page ?? "—"}</dd></div>
                  </dl>
                ))}
              </div>
            </details>
            <details>
              <summary>Execution trace</summary>
              <div className="details-content trace-list">
                {response.trace.map((event, index) => (
                  <div key={`${event.stage}-${index}`}>
                    <strong>{event.stage}</strong>
                    <span>{event.decision}</span>
                    <code>{event.duration_ms === null ? "—" : `${event.duration_ms} ms`}</code>
                  </div>
                ))}
              </div>
            </details>
            <details>
              <summary>Query diagnostics</summary>
              <dl className="details-content metric-grid">
                <div><dt>Route</dt><dd>{response.diagnostics.route}</dd></div>
                <div><dt>Evidence</dt><dd>{response.diagnostics.evidence_state}</dd></div>
                <div><dt>Conflict</dt><dd>{response.diagnostics.conflict_state}</dd></div>
                <div><dt>Citations</dt><dd>{response.diagnostics.citation_validation}</dd></div>
              </dl>
            </details>
            <details>
              <summary>Raw trace</summary>
              <pre className="raw-trace">{JSON.stringify(response.trace, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    </aside>
  );
}
