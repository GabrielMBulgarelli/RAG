import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import type {
  BenchmarkMetric,
  BenchmarkMetricObservation,
  BenchmarkRun,
  BenchmarkSection,
  JsonValue,
} from "../api/types";
import type { BenchmarkController } from "./useBenchmark";

export interface BenchmarkCaseRef {
  caseId: string;
  systemId: string;
}

interface BenchmarkResultsProps {
  benchmark: BenchmarkController;
  caseRef: BenchmarkCaseRef | undefined;
  onCaseRefChange: (caseRef: BenchmarkCaseRef | undefined) => void;
  onClose: () => void;
}

type TabId = "summary" | "retrieval" | "grounding" | "failures";

const TABS: Array<{ id: TabId; label: string }> = [
  { id: "summary", label: "Summary" },
  { id: "retrieval", label: "Retrieval" },
  { id: "grounding", label: "Grounding" },
  { id: "failures", label: "Failures" },
];

const SUMMARY_METRICS = new Set([
  "recall_at_5",
  "mrr_at_5",
  "ndcg_at_5",
  "answer_token_f1",
  "citation_precision",
  "abstention_accuracy",
]);
const RETRIEVAL_PATTERN = /retriev|recall|mrr|ndcg|dense|bm25|hybrid/i;
const GROUNDING_PATTERN = /answer|citation|evidence|abstention|conflict|ground|end.?to.?end/i;

function isSummaryMetric(name: string): boolean {
  return SUMMARY_METRICS.has(name)
    || (/latency/i.test(name) && /p95/i.test(name))
    || (/runtime/i.test(name) && /(failure|count)/i.test(name));
}

function systemLabel(run: BenchmarkRun, systemId: string): string {
  return run.metadata.systems.find((system) => system.id === systemId)?.label ?? systemId;
}

function formatMeasured(metricName: string, value: number): string {
  const formatted = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 6,
  }).format(value);
  if (/_ms$/i.test(metricName)) {
    return `${formatted} ms`;
  }
  if (/_(seconds?|secs?)$/i.test(metricName)) {
    return `${formatted} s`;
  }
  return formatted;
}

function MetricValue({
  metric,
  observation,
}: {
  metric: BenchmarkMetric;
  observation: BenchmarkMetricObservation | undefined;
}) {
  if (!observation) {
    return <span className="metric-unavailable">— <small>Not reported</small></span>;
  }
  const supporting = observation.note
    ?? (observation.status === "not_applicable"
      ? "Not applicable"
      : observation.status === "no_eligible_cases"
        ? "No eligible cases"
        : `${observation.sample_count} samples`);
  if (observation.status !== "measured" || observation.value === null) {
    return (
      <span className="metric-unavailable">
        — <small>{supporting}</small>
      </span>
    );
  }
  return (
    <span>
      <span className="metric-value">{formatMeasured(metric.name, observation.value)}</span>
      <small className="metric-support">{supporting}</small>
    </span>
  );
}

function MetricTable({
  run,
  sections,
  empty,
}: {
  run: BenchmarkRun;
  sections: BenchmarkSection[];
  empty: string;
}) {
  const rows = sections.flatMap((section) => (
    section.metrics.map((metric) => ({ section, metric }))
  ));
  if (rows.length === 0) {
    return <p className="benchmark-empty">{empty}</p>;
  }
  return (
    <div className="benchmark-matrix" tabIndex={0} aria-label="Benchmark metric comparison">
      <table>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            {run.metadata.systems.map((system) => (
              <th key={system.id} scope="col">{system.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ section, metric }) => (
            <tr key={`${section.id}:${metric.name}`}>
              <th scope="row">
                <span>{metric.label}</span>
                <small>{section.title}</small>
              </th>
              {run.metadata.systems.map((system) => (
                <td key={system.id}>
                  <MetricValue
                    metric={metric}
                    observation={metric.observations.find(
                      (observation) => observation.system === system.id,
                    )}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function jsonText(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function EvidenceList({
  label,
  evidence,
}: {
  label: string;
  evidence: Array<Record<string, JsonValue>>;
}) {
  return (
    <section>
      <h4>{label}</h4>
      {evidence.length ? (
        <ol aria-label={label}>
          {evidence.map((item, index) => (
            <li key={index}><pre>{jsonText(item)}</pre></li>
          ))}
        </ol>
      ) : <p className="muted-copy">No {label.toLowerCase()} stored.</p>}
    </section>
  );
}

function CaseDrawer({
  run,
  benchmark,
  caseRef,
  onClose,
}: {
  run: BenchmarkRun;
  benchmark: BenchmarkController;
  caseRef: BenchmarkCaseRef;
  onClose: () => void;
}) {
  const detail = benchmark.caseDetail;
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
  }, []);
  return (
    <aside
      className="benchmark-case-drawer"
      role="dialog"
      aria-modal="false"
      aria-label={`Case ${caseRef.caseId} · ${systemLabel(run, caseRef.systemId)}`}
    >
      <header>
        <div>
          <p className="eyebrow">Case detail</p>
          <h3>{caseRef.caseId} · {systemLabel(run, caseRef.systemId)}</h3>
        </div>
        <button
          ref={closeRef}
          className="icon-button"
          type="button"
          onClick={onClose}
          aria-label="Close case details"
        >
          ×
        </button>
      </header>
      <div className="benchmark-case-drawer__body">
        {benchmark.caseLoading ? (
          <p role="status">Loading case details…</p>
        ) : benchmark.caseError ? (
          <p className="inline-error" role="alert">{benchmark.caseError}</p>
        ) : detail ? (
          <>
            <section><h4>Question</h4><p>{detail.question}</p></section>
            <section><h4>Expected answer</h4><p>{detail.expected_answer ?? "Not provided"}</p></section>
            <section><h4>Generated answer</h4><p>{detail.generated_answer ?? "Not provided"}</p></section>
            <EvidenceList label="Expected evidence" evidence={detail.expected_evidence} />
            <EvidenceList label="Retrieved evidence" evidence={detail.retrieved_evidence} />
            <section>
              <h4>Metric observations</h4>
              {detail.metric_observations.length ? (
                <ul className="case-metrics">
                  {detail.metric_observations.map((observation, index) => (
                    <li key={`${observation.system}:${index}`}>
                      <strong>{systemLabel(run, observation.system)}</strong>
                      {observation.status === "measured" && observation.value !== null
                        ? <span className="metric-value">{observation.value}</span>
                        : <span>—</span>}
                      <small>
                        {observation.note
                          ?? observation.status.replaceAll("_", " ")}
                      </small>
                    </li>
                  ))}
                </ul>
              ) : <p className="muted-copy">No metric observations stored.</p>}
            </section>
            <section>
              <h4>Failure classification</h4>
              <p>{detail.failure_classification ?? "None"}</p>
            </section>
            <details>
              <summary>Execution trace</summary>
              <pre>{jsonText(detail.public_trace)}</pre>
            </details>
            <details>
              <summary>Raw result</summary>
              <pre>{detail.sanitized_raw_result
                ? jsonText(detail.sanitized_raw_result)
                : "No raw result stored."}</pre>
            </details>
          </>
        ) : (
          <p className="muted-copy">No case detail is available.</p>
        )}
      </div>
    </aside>
  );
}

export function BenchmarkResults({
  benchmark,
  caseRef,
  onCaseRefChange,
  onClose,
}: BenchmarkResultsProps) {
  const run = benchmark.run;
  const dialogRef = useRef<HTMLDialogElement>(null);
  const lastInspectRef = useRef<HTMLButtonElement | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("summary");
  const idPrefix = useId();

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

  useEffect(() => {
    if (caseRef && run) {
      void benchmark.openCase(run.run_id, caseRef.caseId, caseRef.systemId);
    }
  }, [benchmark.openCase, caseRef, run]);

  const closeCase = () => {
    benchmark.closeCase();
    onCaseRefChange(undefined);
    lastInspectRef.current?.focus();
  };
  const closeResults = () => {
    if (caseRef) {
      benchmark.closeCase();
      onCaseRefChange(undefined);
    }
    onClose();
  };

  useEffect(() => {
    if (!caseRef) {
      return;
    }
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closeCase();
      }
    };
    window.addEventListener("keydown", closeOnEscape, true);
    return () => window.removeEventListener("keydown", closeOnEscape, true);
  });

  if (!run) {
    return null;
  }

  const summarySections = run.sections.map((section) => ({
    ...section,
    metrics: section.metrics.filter((metric) => isSummaryMetric(metric.name)),
  }));
  const retrievalSections = run.sections.filter((section) => (
    RETRIEVAL_PATTERN.test(`${section.id} ${section.title}`)
  )).map((section) => ({
    ...section,
    metrics: section.metrics.filter((metric) => (
      RETRIEVAL_PATTERN.test(`${metric.name} ${metric.label}`)
    )),
  }));
  const groundingSections = run.sections.filter((section) => (
    GROUNDING_PATTERN.test(`${section.id} ${section.title}`)
  )).map((section) => ({
    ...section,
    metrics: section.metrics.filter((metric) => (
      GROUNDING_PATTERN.test(`${metric.name} ${metric.label}`)
      || /latency/i.test(metric.name)
    )),
  }));

  const selectTab = (tab: TabId) => {
    setActiveTab(tab);
    document.getElementById(`${idPrefix}-tab-${tab}`)?.focus();
  };
  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const current = TABS.findIndex((tab) => tab.id === activeTab);
    let next = current;
    if (event.key === "ArrowRight") next = (current + 1) % TABS.length;
    if (event.key === "ArrowLeft") next = (current - 1 + TABS.length) % TABS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = TABS.length - 1;
    if (next !== current) {
      event.preventDefault();
      selectTab(TABS[next].id);
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="dialog benchmark-results"
      aria-label="RAG Benchmark"
      onCancel={(event) => {
        event.preventDefault();
        if (caseRef) closeCase();
        else closeResults();
      }}
    >
      <header className="benchmark-results__header">
        <div>
          <p className="eyebrow">Completed evaluation</p>
          <h2>RAG Benchmark</h2>
          <dl className="benchmark-metadata">
            <div><dt>Dataset</dt><dd>{run.metadata.dataset}</dd></div>
            <div><dt>Split</dt><dd>{run.metadata.split}</dd></div>
            <div><dt>Cases</dt><dd>{run.progress.total_cases}</dd></div>
            <div><dt>Chat</dt><dd>{run.metadata.chat_model}</dd></div>
            <div><dt>Embedding</dt><dd>{run.metadata.embedding_model}</dd></div>
            <div><dt>Status</dt><dd>{run.status}</dd></div>
          </dl>
        </div>
        <button className="icon-button" type="button" onClick={closeResults} aria-label="Close benchmark results">×</button>
      </header>

      <nav className="benchmark-tabs" role="tablist" aria-label="Benchmark result sections">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            id={`${idPrefix}-tab-${tab.id}`}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`${idPrefix}-panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => setActiveTab(tab.id)}
            onKeyDown={onTabKeyDown}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="benchmark-results__body">
        <section
          id={`${idPrefix}-panel-summary`}
          role="tabpanel"
          aria-labelledby={`${idPrefix}-tab-summary`}
          aria-label="Summary"
          hidden={activeTab !== "summary"}
        >
          <MetricTable
            run={run}
            sections={summarySections}
            empty="No canonical summary metrics were reported."
          />
        </section>
        <section
          id={`${idPrefix}-panel-retrieval`}
          role="tabpanel"
          aria-labelledby={`${idPrefix}-tab-retrieval`}
          aria-label="Retrieval"
          hidden={activeTab !== "retrieval"}
        >
          <MetricTable
            run={run}
            sections={retrievalSections}
            empty="No retrieval metrics were reported."
          />
        </section>
        <section
          id={`${idPrefix}-panel-grounding`}
          role="tabpanel"
          aria-labelledby={`${idPrefix}-tab-grounding`}
          aria-label="Grounding"
          hidden={activeTab !== "grounding"}
        >
          <MetricTable
            run={run}
            sections={groundingSections}
            empty="No grounding metrics were reported."
          />
        </section>
        <section
          id={`${idPrefix}-panel-failures`}
          role="tabpanel"
          aria-labelledby={`${idPrefix}-tab-failures`}
          aria-label="Failures"
          hidden={activeTab !== "failures"}
        >
          {run.failures.length ? (
            <div className="benchmark-matrix">
              <table>
                <thead><tr><th>Case</th><th>System</th><th>Failure</th><th>Inspect</th></tr></thead>
                <tbody>
                  {run.failures.map((failure, index) => (
                    <tr key={`${failure.case_id}:${failure.system}:${index}`}>
                      <td>{failure.case_id}</td>
                      <td>{systemLabel(run, failure.system)}</td>
                      <td><strong>{failure.classification}</strong><small>{failure.detail}</small></td>
                      <td>
                        <button
                          className="button button--quiet"
                          type="button"
                          aria-label={`Inspect ${failure.case_id} for ${systemLabel(run, failure.system)}`}
                          onClick={(event) => {
                            lastInspectRef.current = event.currentTarget;
                            onCaseRefChange({
                              caseId: failure.case_id,
                              systemId: failure.system,
                            });
                          }}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="benchmark-empty">No benchmark failures were reported.</p>}
        </section>
        <details>
          <summary>Benchmark configuration</summary>
          <dl className="benchmark-config">
            <div><dt>Systems</dt><dd>{run.metadata.systems.map((system) => system.label).join(", ")}</dd></div>
            <div><dt>Chat model</dt><dd>{run.metadata.chat_model}</dd></div>
            <div><dt>Embedding model</dt><dd>{run.metadata.embedding_model}</dd></div>
          </dl>
        </details>
        <details>
          <summary>Reproducibility metadata</summary>
          <pre>{jsonText(run.metadata.reproducibility)}</pre>
        </details>
        {benchmark.downloadError ? (
          <p className="inline-error" role="alert">{benchmark.downloadError}</p>
        ) : null}
      </div>

      <footer className="benchmark-results__footer">
        <button
          className="button button--secondary"
          type="button"
          aria-label="Download results"
          disabled={benchmark.downloadInFlight}
          onClick={() => void benchmark.download()}
        >
          {benchmark.downloadInFlight ? "Downloading…" : "Download results"}
        </button>
        <button className="button button--primary" type="button" onClick={closeResults}>Close</button>
      </footer>

      {caseRef ? (
        <CaseDrawer
          run={run}
          benchmark={benchmark}
          caseRef={caseRef}
          onClose={closeCase}
        />
      ) : null}
    </dialog>
  );
}
