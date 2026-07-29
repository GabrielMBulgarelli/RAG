import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

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
  const sourcesTabRef = useRef<HTMLButtonElement>(null);
  const detailsTabRef = useRef<HTMLButtonElement>(null);
  const response = workspace.selectedExchange?.response ?? null;

  useEffect(() => {
    if (workspace.selectedSourceLabel) {
      setTab("sources");
    }
  }, [workspace.selectedSourceLabel]);

  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentTab: "sources" | "details",
  ) => {
    let nextTab: "sources" | "details" | null = null;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      nextTab = currentTab === "sources" ? "details" : "sources";
    } else if (event.key === "Home") {
      nextTab = "sources";
    } else if (event.key === "End") {
      nextTab = "details";
    }
    if (!nextTab) {
      return;
    }
    event.preventDefault();
    setTab(nextTab);
    (nextTab === "sources" ? sourcesTabRef : detailsTabRef).current?.focus();
  };

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
          ref={sourcesTabRef}
          id="inspector-tab-sources"
          type="button"
          role="tab"
          aria-selected={tab === "sources"}
          aria-controls="inspector-panel-sources"
          tabIndex={tab === "sources" ? 0 : -1}
          onClick={() => setTab("sources")}
          onKeyDown={(event) => handleTabKeyDown(event, "sources")}
        >
          Sources
        </button>
        <button
          ref={detailsTabRef}
          id="inspector-tab-details"
          type="button"
          role="tab"
          aria-selected={tab === "details"}
          aria-controls="inspector-panel-details"
          tabIndex={tab === "details" ? 0 : -1}
          onClick={() => setTab("details")}
          onKeyDown={(event) => handleTabKeyDown(event, "details")}
        >
          Details
        </button>
      </div>

      <div
        id="inspector-panel-sources"
        className="inspector__content"
        role="tabpanel"
        aria-labelledby="inspector-tab-sources"
        hidden={tab !== "sources"}
      >
        {!response ? (
          <div className="inspector-empty">
            <span aria-hidden="true">01</span>
            <p>Select an answer to inspect its sources and execution details.</p>
          </div>
        ) : (
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
        )}
      </div>

      <div
        id="inspector-panel-details"
        className="inspector__content"
        role="tabpanel"
        aria-labelledby="inspector-tab-details"
        hidden={tab !== "details"}
      >
        {!response ? (
          <div className="inspector-empty">
            <span aria-hidden="true">01</span>
            <p>Select an answer to inspect its sources and execution details.</p>
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
