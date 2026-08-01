import type { BenchmarkController } from "./useBenchmark";

interface BenchmarkProgressProps {
  benchmark: BenchmarkController;
  onClose: () => void;
}

const STATUS_LABELS = {
  queued: "Queued",
  running: "Running",
  cancellation_requested: "Cancellation requested",
  cancelled: "Benchmark cancelled",
  failed: "Benchmark failed",
  completed: "Benchmark completed",
} as const;

export function BenchmarkProgress({
  benchmark,
  onClose,
}: BenchmarkProgressProps) {
  const run = benchmark.run;
  const progress = run?.progress;
  const terminal = run?.status === "cancelled"
    || run?.status === "failed"
    || run?.status === "completed";
  const currentSystem = run?.metadata.systems.find(
    (system) => system.id === progress?.current_system,
  )?.label ?? progress?.current_system;

  return (
    <section
      className="benchmark-progress"
      role="dialog"
      aria-modal="false"
      aria-label="Running RAG benchmark"
    >
      <header>
        <p className="eyebrow">Evaluation</p>
        <h2>Running RAG benchmark</h2>
      </header>
      <div className="benchmark-progress__body" aria-live="polite" aria-atomic="true">
        <strong className={`benchmark-status benchmark-status--${run?.status ?? "queued"}`}>
          {run ? STATUS_LABELS[run.status] : "Starting"}
        </strong>
        {run ? (
          <>
            <p>
              Preparing {run.metadata.dataset} {run.metadata.split} cases
            </p>
            {progress && progress.current_system_index !== null ? (
              <p>
                System {progress.current_system_index} of {progress.total_systems}
                {currentSystem ? `: ${currentSystem}` : ""}
              </p>
            ) : null}
            {progress && progress.current_case_index !== null ? (
              <p>Case {progress.current_case_index} of {progress.total_cases}</p>
            ) : null}
            <progress
              aria-label="Benchmark cases completed"
              value={progress?.completed_cases ?? 0}
              max={Math.max(progress?.total_cases ?? 0, 1)}
            />
          </>
        ) : (
          <p>Starting the durable benchmark run…</p>
        )}
        {run?.error ? <p className="inline-error" role="alert">{run.error.message}</p> : null}
        {run?.status === "cancellation_requested" ? (
          <p>
            Cancellation requested. The active model request may finish; no additional cases will start.
          </p>
        ) : null}
        {benchmark.startError ? (
          <p className="inline-error" role="alert">{benchmark.startError}</p>
        ) : null}
        {benchmark.connectionError ? (
          <div className="inline-error" role="alert">
            <p>{benchmark.connectionError}</p>
            <button
              className="button button--quiet"
              type="button"
              onClick={benchmark.retryConnection}
            >
              Retry connection
            </button>
          </div>
        ) : null}
        {benchmark.cancelError ? (
          <p className="inline-error" role="alert">{benchmark.cancelError}</p>
        ) : null}
      </div>
      <footer>
        {terminal || benchmark.startError ? (
          <button
            className="button button--primary"
            type="button"
            onClick={onClose}
            aria-label="Close benchmark progress"
          >
            Close
          </button>
        ) : (
          <button
            className="button button--secondary"
            type="button"
            disabled={
              benchmark.startInFlight
              || benchmark.cancelInFlight
              || run?.status === "cancellation_requested"
            }
            onClick={() => void benchmark.cancel()}
            aria-label="Cancel benchmark"
          >
            {benchmark.cancelInFlight ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </footer>
    </section>
  );
}
