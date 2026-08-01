import { useState } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { BenchmarkRun } from "../api/types";
import {
  benchmarkCase,
  benchmarkCases,
  benchmarkRun,
  benchmarkRunId,
} from "../test/fixtures";
import { BenchmarkProgress } from "./BenchmarkProgress";
import { BenchmarkResults, type BenchmarkCaseRef } from "./BenchmarkResults";
import type { BenchmarkController } from "./useBenchmark";

function controller(
  run: BenchmarkRun | null = benchmarkRun,
  overrides: Partial<BenchmarkController> = {},
): BenchmarkController {
  return {
    run,
    startInFlight: false,
    cancelInFlight: false,
    connectionState: "idle",
    lastEventId: 0,
    startError: null,
    connectionError: null,
    cancelError: null,
    downloadError: null,
    downloadInFlight: false,
    caseDetail: null,
    cases: benchmarkCases,
    casesLoading: false,
    casesError: null,
    caseLoading: false,
    caseError: null,
    busy: false,
    start: vi.fn().mockResolvedValue(true),
    retryConnection: vi.fn(),
    cancel: vi.fn().mockResolvedValue(true),
    openCase: vi.fn().mockResolvedValue(true),
    loadCases: vi.fn().mockResolvedValue(true),
    closeCase: vi.fn(),
    download: vi.fn().mockResolvedValue(true),
    ...overrides,
  };
}

function activeRun(status: BenchmarkRun["status"]): BenchmarkRun {
  return {
    ...benchmarkRun,
    status,
    progress: {
      ...benchmarkRun.progress,
      completed_cases: 0,
      current_system_index: 0,
      current_case_index: 0,
    },
    metadata: { ...benchmarkRun.metadata, completed_at: null },
  };
}

describe("benchmark overlays", () => {
  it.each([
    ["queued", "Queued"],
    ["running", "Running"],
    ["cancellation_requested", "Cancellation requested"],
  ] as const)("shows exact active progress for %s", (status, label) => {
    render(
      <BenchmarkProgress
        benchmark={controller(activeRun(status))}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByRole("dialog", { name: "Running RAG benchmark" })).toBeVisible();
    expect(screen.getByText(label)).toBeVisible();
    expect(screen.getByText("System 0 of 7: Full RAG")).toBeVisible();
    expect(screen.getByText("Case 0 of 20")).toBeVisible();
    expect(screen.getByRole("progressbar")).toHaveAttribute("value", "0");
    if (status === "cancellation_requested") {
      expect(screen.getByRole("button", { name: "Cancel benchmark" })).toBeDisabled();
      expect(screen.getByText(
        "Cancellation requested. The active model request may finish; no additional cases will start.",
      )).toBeVisible();
    } else {
      expect(screen.getByRole("button", { name: "Cancel benchmark" })).toBeEnabled();
    }
  });

  it.each([
    ["cancelled", "Benchmark cancelled"],
    ["failed", "Benchmark failed"],
    ["completed", "Benchmark completed"],
  ] as const)("keeps terminal %s state inspectable", (status, label) => {
    const run = {
      ...activeRun(status),
      error: status === "failed" ? {
        code: "benchmark_failed",
        message: "Executor stopped.",
        details: { retryable: true },
      } : null,
    };
    render(
      <BenchmarkProgress
        benchmark={controller(run)}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText(label)).toBeVisible();
    if (status === "failed") {
      expect(screen.getByText("Executor stopped.")).toBeVisible();
    }
    expect(screen.getByRole("button", { name: "Close benchmark progress" })).toBeEnabled();
  });

  it("renders six result tabs and respects each section's system IDs", async () => {
    const user = userEvent.setup();
    render(
      <BenchmarkResults
        benchmark={controller()}
        caseRef={undefined}
        onCaseRefChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "RAG Benchmark" });
    expect(within(dialog).getByText("MultiHopRAG")).toBeVisible();
    expect(within(dialog).getByText("development")).toBeVisible();
    expect(within(dialog).getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Summary", "Retrieval", "Grounding", "Execution", "Cases", "Failures",
    ]);
    const summary = within(dialog).getByRole("tabpanel", { name: "Summary" });
    expect(within(summary).getAllByText("0", { selector: ".metric-value" })[0]).toBeVisible();
    expect(within(summary).getAllByText("Not scored for this fixture.")[0]).toBeVisible();
    expect(within(summary).getAllByText("No eligible cases")[0]).toBeVisible();
    expect(within(summary).getByText("Runtime error count")).toBeVisible();
    expect(within(summary).getByText("Runtime error rate")).toBeVisible();

    await user.click(screen.getByRole("tab", { name: "Retrieval" }));
    const retrieval = screen.getByRole("tabpanel", { name: "Retrieval" });
    expect(within(retrieval).getAllByRole("columnheader").map((cell) => cell.textContent))
      .toEqual(["Metric", "Dense", "BM25", "Hybrid"]);
    await user.click(screen.getByRole("tab", { name: "Grounding" }));
    const grounding = screen.getByRole("tabpanel", { name: "Grounding" });
    expect(within(grounding).getAllByRole("columnheader").map((cell) => cell.textContent))
      .toEqual(["Metric", "Dense RAG", "BM25 RAG", "Hybrid RAG", "Full RAG"]);
  });

  it("implements owned keyboard tabs and schema-driven section filtering", async () => {
    const user = userEvent.setup();
    render(
      <BenchmarkResults
        benchmark={controller()}
        caseRef={undefined}
        onCaseRefChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    const summary = screen.getByRole("tab", { name: "Summary" });
    summary.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Retrieval" })).toHaveFocus();
    expect(screen.getByRole("tabpanel", { name: "Retrieval" })).toHaveTextContent("Recall at 5");

    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Failures" })).toHaveFocus();
    await user.keyboard("{Home}");
    expect(summary).toHaveFocus();

    await user.click(screen.getByRole("tab", { name: "Grounding" }));
    const grounding = screen.getByRole("tabpanel", { name: "Grounding" });
    expect(grounding).toHaveTextContent("Answer token F1");
    expect(grounding).not.toHaveTextContent("Recall at 5");
    await user.click(screen.getByRole("tab", { name: "Execution" }));
    expect(screen.getByRole("tabpanel", { name: "Execution" })).toHaveTextContent("P95 latency");
  });

  it("lists successful and failed cases and opens either in the existing drawer", async () => {
    const user = userEvent.setup();
    const openCase = vi.fn().mockResolvedValue(true);
    const loadCases = vi.fn().mockResolvedValue(true);
    const onCaseRefChange = vi.fn();
    render(
      <BenchmarkResults
        benchmark={controller(benchmarkRun, { openCase, loadCases })}
        caseRef={undefined}
        onCaseRefChange={onCaseRefChange}
        onClose={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("tab", { name: "Cases" }));
    expect(loadCases).toHaveBeenCalledWith(benchmarkRunId);
    const cases = screen.getByRole("tabpanel", { name: "Cases" });
    expect(within(cases).getByText("Successful")).toBeVisible();
    expect(within(cases).getByText("Expectation failure")).toBeVisible();
    expect(within(cases).getByText("Runtime failure")).toBeVisible();
    await user.click(within(cases).getByRole("button", {
      name: "Inspect case-success for Dense",
    }));
    expect(onCaseRefChange).toHaveBeenLastCalledWith({
      caseId: "case-success",
      systemId: "dense",
    });
    await user.click(within(cases).getByRole("button", {
      name: "Inspect case-runtime for Full RAG",
    }));
    expect(onCaseRefChange).toHaveBeenLastCalledWith({
      caseId: "case-runtime",
      systemId: "full-rag",
    });
  });

  it("derives latency units and preserves readable ratio precision", () => {
    const metricRun: BenchmarkRun = {
      ...benchmarkRun,
      sections: [{
        ...benchmarkRun.sections[1],
        metrics: [{
          name: "answer_token_f1",
          label: "Answer token F1",
          observations: [{
            system: "full-rag", value: 0.712345, status: "measured", sample_count: 1, note: null,
          }],
        }],
      }, {
        ...benchmarkRun.sections[2],
        metrics: [{
          name: "p95_latency_seconds",
          label: "P95 latency",
          observations: [{
            system: "full-rag", value: 1.25, status: "measured", sample_count: 1, note: null,
          }],
        }],
      }],
    };
    render(
      <BenchmarkResults
        benchmark={controller(metricRun)}
        caseRef={undefined}
        onCaseRefChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const summary = screen.getByRole("tabpanel", { name: "Summary" });
    expect(within(summary).getByText("1.25 s")).toBeVisible();
    expect(within(summary).queryByText("1.25 ms")).not.toBeInTheDocument();
    expect(within(summary).getByText("0.712345")).toBeVisible();
  });

  it("keeps the results open under a case drawer and restores Inspect focus on Escape", async () => {
    const user = userEvent.setup();
    const openCase = vi.fn().mockResolvedValue(true);
    const closeCase = vi.fn();
    const closeResults = vi.fn();

    function Harness() {
      const [caseRef, setCaseRef] = useState<BenchmarkCaseRef>();
      return (
        <BenchmarkResults
          benchmark={controller(benchmarkRun, {
            caseDetail: benchmarkCase,
            openCase,
            closeCase,
          })}
          caseRef={caseRef}
          onCaseRefChange={setCaseRef}
          onClose={closeResults}
        />
      );
    }

    render(<Harness />);
    await user.click(screen.getByRole("tab", { name: "Failures" }));
    const inspect = screen.getByRole("button", { name: "Inspect case / 1 for Full RAG" });
    await user.click(inspect);

    expect(screen.getByRole("dialog", { name: "Case case / 1 · Full RAG" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Close case details" })).toHaveFocus();
    expect(screen.getByRole("dialog", { name: "RAG Benchmark" })).toBeVisible();
    const retrieved = screen.getByRole("list", { name: "Retrieved evidence" });
    const evidence = within(retrieved).getAllByRole("listitem");
    expect(evidence[0]).toHaveTextContent("Retrieved first.");
    expect(evidence[1]).toHaveTextContent("Retrieved second.");
    const drawer = screen.getByRole("dialog", { name: /Case case/ });
    expect(within(drawer).getByText("Recall at 5")).toBeVisible();
    expect(within(drawer).getByText("Answer token F1")).toBeVisible();
    expect(screen.getByText("Retrieval only.")).toBeVisible();
    expect(within(screen.getByRole("dialog", { name: "Case case / 1 · Full RAG" }))
      .getByText("citation_mismatch")).toBeVisible();

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: /Case case/ })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "RAG Benchmark" })).toBeVisible();
    expect(inspect).toHaveFocus();
    expect(closeCase).toHaveBeenCalled();
    expect(openCase).toHaveBeenCalledWith(benchmarkRunId, "case / 1", "full-rag");

    await user.click(inspect);
    await user.click(screen.getByRole("button", { name: "Close benchmark results" }));
    expect(closeCase).toHaveBeenCalledTimes(2);
    expect(closeResults).toHaveBeenCalledTimes(1);
  });

  it("shows case loading/error and download error without hiding results", async () => {
    const user = userEvent.setup();
    const onCaseRefChange = vi.fn();
    const { rerender } = render(
      <BenchmarkResults
        benchmark={controller(benchmarkRun, {
          caseLoading: true,
          downloadError: "Download unavailable.",
          downloadInFlight: true,
        })}
        caseRef={{ caseId: "case / 1", systemId: "full-rag" }}
        onCaseRefChange={onCaseRefChange}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Loading case details");
    expect(screen.getByRole("alert")).toHaveTextContent("Download unavailable.");
    expect(screen.getByRole("button", { name: "Download results" })).toBeDisabled();

    rerender(
      <BenchmarkResults
        benchmark={controller(benchmarkRun, { caseError: "Case unavailable." })}
        caseRef={{ caseId: "case / 1", systemId: "full-rag" }}
        onCaseRefChange={onCaseRefChange}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Case unavailable.");
    expect(screen.getByRole("button", { name: "Download results" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Close case details" }));
    expect(onCaseRefChange).toHaveBeenCalledWith(undefined);
  });
});
