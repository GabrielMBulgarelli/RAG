import { expect, test, type Page, type Route } from "@playwright/test";

const runId = "4cbdbcb9-5a57-4514-a392-2dce907456d5";

const systems = [
  { id: "dense", label: "Dense" },
  { id: "bm25", label: "BM25" },
  { id: "hybrid", label: "Hybrid" },
  { id: "dense-rag", label: "Dense RAG" },
  { id: "bm25-rag", label: "BM25 RAG" },
  { id: "hybrid-rag", label: "Hybrid RAG" },
  { id: "full-rag", label: "Full RAG" },
];

const runtime = {
  state: "ready",
  configured_chat_model: "qwen3:8b",
  active_chat_model: "qwen3:8b",
  embedding_model: "nomic-embed-text",
  available_chat_models: ["qwen3:8b"],
  detail: "Models are ready.",
  capabilities: {
    can_query: true,
    can_load_models: true,
    can_upload: true,
    can_run_benchmark: true,
  },
  active_operation: null,
  corpus: { document_count: 1, page_count: 3, chunk_count: 7, status: "ready" },
};

const alphaDocument = {
  id: "doc/alpha",
  filename: "alpha.pdf",
  state: "ready",
  size_bytes: 2048,
  page_count: 3,
  chunk_count: 7,
  indexed_at: "2026-07-29T10:00:00Z",
  updated_at: "2026-07-29T10:00:00Z",
};

const diagnostics = {
  state: "blocked",
  title: "Benchmark preparation required",
  detail: "The workspace is ready, but benchmark files are missing.",
  runtime_checks: [{ area: "runtime", name: "Chat model", state: "ready", detail: "qwen3:8b loaded" }],
  index_checks: [{ area: "index", name: "Vector index", state: "ready", detail: "7 chunks available" }],
  evaluation_checks: [{
    area: "evaluation",
    name: "Benchmark files",
    state: "blocked",
    detail: "Run uv run python scripts/prepare_multihop_eval.py --index.",
  }],
  active_operation: null,
  stale: false,
};

const completedRun = {
  run_id: runId,
  status: "completed",
  links: {
    run: `/api/benchmarks/${runId}`,
    events: `/api/benchmarks/${runId}/events`,
    download: `/api/benchmarks/${runId}/download`,
  },
  progress: {
    completed_cases: 20,
    total_cases: 20,
    current_system: "full-rag",
    current_system_index: 7,
    total_systems: 7,
    current_case_id: "case-failed",
    current_case_index: 20,
  },
  metadata: {
    dataset: "MultiHopRAG",
    split: "development",
    systems,
    chat_model: "qwen3:8b",
    embedding_model: "nomic-embed-text",
    started_at: "2026-07-29T15:30:00Z",
    completed_at: "2026-07-29T15:34:00Z",
    reproducibility: { git_commit: "b424dc7", case_ids: ["case-success", "case-failed"] },
  },
  sections: [
    {
      id: "retrieval",
      title: "Retrieval",
      system_ids: ["dense", "bm25", "hybrid"],
      detail: "Retrieval quality.",
      metrics: [{
        name: "recall_at_5",
        label: "Recall at 5",
        observations: systems.slice(0, 3).map(({ id }, index) => ({
          system: id,
          value: 0.8 + index / 100,
          status: "measured",
          sample_count: 20,
          note: null,
        })),
      }],
    },
    {
      id: "grounding",
      title: "Grounding",
      system_ids: ["dense-rag", "bm25-rag", "hybrid-rag", "full-rag"],
      detail: "Answer grounding.",
      metrics: [{
        name: "answer_token_f1",
        label: "Answer token F1",
        observations: systems.slice(3).map(({ id }, index) => ({
          system: id,
          value: 0.7 + index / 100,
          status: "measured",
          sample_count: 20,
          note: null,
        })),
      }],
    },
    {
      id: "execution",
      title: "Execution",
      system_ids: ["full-rag"],
      detail: "Runtime behavior.",
      metrics: [
        {
          name: "p95_latency_seconds",
          label: "P95 latency",
          observations: [{ system: "full-rag", value: 1.25, status: "measured", sample_count: 20, note: null }],
        },
        {
          name: "runtime_error_count",
          label: "Runtime error count",
          observations: [{ system: "full-rag", value: 1, status: "measured", sample_count: 20, note: null }],
        },
        {
          name: "runtime_error_rate",
          label: "Runtime error rate",
          observations: [{ system: "full-rag", value: 0.05, status: "measured", sample_count: 20, note: null }],
        },
      ],
    },
  ],
  failures: [{
    case_id: "case-failed",
    system: "full-rag",
    classification: "citation_mismatch",
    detail: "The cited evidence did not support the answer.",
  }],
  error: null,
};

const runningRun = {
  ...completedRun,
  status: "running",
  progress: {
    ...completedRun.progress,
    completed_cases: 3,
    current_system: "dense",
    current_system_index: 1,
    current_case_id: "case-success",
    current_case_index: 4,
  },
  metadata: { ...completedRun.metadata, completed_at: null },
  failures: [],
};

const cases = [
  {
    case_id: "case-success",
    system: "dense",
    question: "Which document matched?",
    outcome: "successful",
    failure_classification: null,
  },
  {
    case_id: "case-failed",
    system: "full-rag",
    question: "Which policy changed?",
    outcome: "expectation_failure",
    failure_classification: "citation_mismatch",
  },
];

function caseDetail(caseId: string, system: string) {
  const failed = caseId === "case-failed";
  return {
    case_id: caseId,
    system,
    question: failed ? "Which policy changed?" : "Which document matched?",
    expected_answer: failed ? "The July policy." : "alpha.pdf",
    generated_answer: failed ? "The July policy changed." : null,
    expected_evidence: [{ document_id: "expected-1", text: "Expected evidence text." }],
    retrieved_evidence: [{
      chunk_id: "chunk-1",
      document_id: "doc/alpha",
      filename: "alpha.pdf",
      page: 2,
      excerpt: "Readable retrieved evidence text.",
    }],
    metric_observations: [{
      name: "recall_at_5",
      label: "Recall at 5",
      system,
      value: failed ? 0 : 1,
      status: "measured",
      sample_count: 1,
      note: null,
    }],
    failure_classification: failed ? "citation_mismatch" : null,
    public_trace: [],
    sanitized_raw_result: { route: system },
  };
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function installApi(page: Page) {
  let documents = [alphaDocument];
  let benchmarkStarted = false;
  let cancellationRequested = false;

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (path === "/api/runtime") return json(route, runtime);
    if (path === "/api/diagnostics") return json(route, diagnostics);
    if (path === "/api/documents" && method === "GET") {
      return json(route, {
        documents,
        corpus: documents.length
          ? runtime.corpus
          : { document_count: 0, page_count: 0, chunk_count: 0, status: "empty" },
        active_operation: null,
      });
    }
    if (path === "/api/documents" && method === "POST") {
      return json(route, { accepted: [], documents, corpus: runtime.corpus });
    }
    if (path === "/api/documents/doc%2Falpha" && method === "DELETE") {
      documents = [];
      return json(route, {
        documents,
        corpus: { document_count: 0, page_count: 0, chunk_count: 0, status: "empty" },
        active_operation: null,
      });
    }
    if (path === "/api/benchmarks" && method === "POST") {
      benchmarkStarted = true;
      return json(route, {
        run_id: runId,
        status: "queued",
        links: completedRun.links,
      });
    }
    if (path === "/api/benchmarks/latest") return json(route, completedRun);
    if (path === `/api/benchmarks/${runId}/cancel` && method === "POST") {
      cancellationRequested = true;
      return json(route, { ...runningRun, status: "cancellation_requested" });
    }
    if (path === `/api/benchmarks/${runId}/events`) {
      return route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: ": connected\n\n",
      });
    }
    if (path === `/api/benchmarks/${runId}/download`) {
      return route.fulfill({
        status: 200,
        contentType: "application/zip",
        headers: { "Content-Disposition": "attachment; filename=benchmark-browser-fixture.zip" },
        body: "fixture archive",
      });
    }
    if (path === `/api/benchmarks/${runId}/cases`) return json(route, cases);

    const detailMatch = path.match(
      new RegExp(`^/api/benchmarks/${runId}/cases/([^/]+)/systems/([^/]+)$`),
    );
    if (detailMatch) {
      return json(route, caseDetail(
        decodeURIComponent(detailMatch[1]),
        decodeURIComponent(detailMatch[2]),
      ));
    }
    if (path === `/api/benchmarks/${runId}`) {
      return json(
        route,
        benchmarkStarted
          ? { ...runningRun, status: cancellationRequested ? "cancellation_requested" : "running" }
          : completedRun,
      );
    }

    return json(route, { code: "missing_browser_fixture", message: path, details: {} }, 404);
  });
}

async function expectNoViewportOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  }));
  expect(dimensions.documentWidth).toBeLessThanOrEqual(dimensions.viewportWidth);
}

for (const viewport of [
  { name: "desktop", width: 1440, height: 1000, compact: false },
  { name: "tablet", width: 1024, height: 768, compact: false },
  { name: "mobile", width: 390, height: 844, compact: true },
]) {
  test(`${viewport.name} workspace is responsive`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await installApi(page);
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "Local Document RAG" })).toBeVisible();
    await expectNoViewportOverflow(page);

    if (!viewport.compact) {
      await expect(page.getByRole("complementary", { name: "Workspace controls" })).toBeVisible();
      await expect(page.getByRole("button", { name: "Open workspace controls" })).toHaveCount(0);
      await page.getByRole("button", { name: "Collapse inspector" }).click();
      await expect(page.getByRole("button", { name: "Expand inspector" })).toBeVisible();
      await page.getByRole("button", { name: "Expand inspector" }).click();
      await expect(page.getByRole("button", { name: "Collapse inspector" })).toBeVisible();
      return;
    }

    const opener = page.getByRole("button", { name: "Open workspace controls" });
    await opener.click();
    await expect(page.getByRole("button", { name: "Close workspace controls" })).toBeFocused();
    await expect(page.getByRole("main", { includeHidden: true })).toHaveAttribute("inert", "");
    await page.keyboard.press("Escape");
    await expect(opener).toBeFocused();
    await expect(page.getByRole("main")).not.toHaveAttribute("inert", "");
  });
}

test("document controls and diagnostics stay usable inside mobile overlays", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installApi(page);
  await page.goto("/");

  const opener = page.getByRole("button", { name: "Open workspace controls" });
  await opener.click();
  await page.getByRole("button", { name: "alpha.pdf details" }).click();
  const details = page.getByRole("dialog", { name: "Document details" });
  await expect(details).toContainText("3 pages");
  await details.getByRole("button", { name: "Delete document" }).click();
  const confirmation = page.getByRole("dialog", { name: "Delete alpha.pdf?" });
  await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);

  await page.getByRole("button", { name: "System diagnostics" }).click();
  const dialog = page.getByRole("dialog", { name: "System diagnostics" });
  await expect(dialog.getByRole("heading", { name: "Runtime" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "Index" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "Evaluation" })).toBeVisible();
  await expect(dialog.getByText("uv run python scripts/prepare_multihop_eval.py --index", { exact: true })).toBeVisible();
  const box = await dialog.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.y).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
  expect(box!.y + box!.height).toBeLessThanOrEqual(844);
  await expectNoViewportOverflow(page);
});

test("benchmark progress, cancellation, stored results, cases, and download work end to end", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await installApi(page);
  await page.goto("/");

  await page.getByRole("button", { name: "Run benchmark" }).click();
  const progress = page.getByRole("dialog", { name: "Running RAG benchmark" });
  await expect(progress).toContainText("Running");
  await expect(progress).toContainText("Case 4 of 20");
  await expect(page.getByLabel("Ask about your documents")).toBeDisabled();
  await expect(page.getByRole("button", { name: "Send question" })).toBeDisabled();
  await progress.getByRole("button", { name: "Cancel benchmark" }).click();
  await expect(progress).toContainText("Cancellation requested");
  await expect(page.getByLabel("Ask about your documents")).toBeDisabled();

  await page.reload();
  await page.getByRole("button", { name: "View latest results" }).click();
  const results = page.getByRole("dialog", { name: "RAG Benchmark" });
  await expect(results).toBeVisible();
  await expect(results.getByRole("tab")).toHaveText([
    "Summary",
    "Retrieval",
    "Grounding",
    "Execution",
    "Cases",
    "Failures",
  ]);
  await expect(results.getByRole("tabpanel", { name: "Summary" })).toContainText("Runtime error count");

  for (const tab of ["Retrieval", "Grounding", "Execution"]) {
    await results.getByRole("tab", { name: tab }).click();
    await expect(results.getByRole("tabpanel", { name: tab })).toBeVisible();
  }

  await results.getByRole("tab", { name: "Cases" }).click();
  await expect(results.getByText("Successful", { exact: true })).toBeVisible();
  const successfulInspect = results.getByRole("button", { name: "Inspect case-success for Dense" });
  await successfulInspect.click();
  const successfulCase = page.getByRole("dialog", { name: "Case case-success · Dense" });
  await expect(successfulCase).toContainText("Readable retrieved evidence text.");
  await expect(successfulCase.getByRole("button", { name: "Close case details" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(successfulInspect).toBeFocused();

  await results.getByRole("tab", { name: "Failures" }).click();
  const failedInspect = results.getByRole("button", { name: "Inspect case-failed for Full RAG" });
  await failedInspect.click();
  const failedCase = page.getByRole("dialog", { name: "Case case-failed · Full RAG" });
  await expect(failedCase).toContainText("citation_mismatch");
  await failedCase.getByRole("button", { name: "Close case details" }).click();
  await expect(failedInspect).toBeFocused();

  const downloadPromise = page.waitForEvent("download");
  await results.getByRole("button", { name: "Download results" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("benchmark-browser-fixture.zip");
  await expectNoViewportOverflow(page);
});
