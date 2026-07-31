import { beforeEach, describe, expect, it, vi } from "vitest";

import { createApiClient } from "./client";

const RUN_ID = "4cbdbcb9-5a57-4514-a392-2dce907456d5";

function responseFromChunks(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  }), {
    headers: { "Content-Type": "text/event-stream" },
  });
}

function eventJson(eventId = 1, type = "benchmark.started"): string {
  return JSON.stringify({
    event_id: eventId,
    run_id: RUN_ID,
    type,
    timestamp: "2026-07-29T15:31:00Z",
    data: { phase: "running" },
  });
}

describe("benchmark event stream", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  it("parses arbitrary CRLF chunks and multiline data with Last-Event-ID", async () => {
    const json = JSON.stringify(JSON.parse(eventJson()), null, 2);
    const wire = [
      "id: 1",
      "event: benchmark.started",
      ...json.split("\n").map((line) => `data: ${line}`),
      "",
      "",
    ].join("\r\n");
    fetchMock.mockResolvedValueOnce(responseFromChunks([
      wire.slice(0, 19),
      wire.slice(19, 67),
      wire.slice(67, 111),
      wire.slice(111),
    ]));
    const accepted: unknown[] = [];
    const controller = new AbortController();

    await createApiClient().streamBenchmarkEvents(
      RUN_ID,
      0,
      controller.signal,
      (event) => accepted.push(event),
    );

    expect(fetchMock).toHaveBeenCalledWith(
      `/api/benchmarks/${RUN_ID}/events`,
      expect.objectContaining({
        headers: expect.objectContaining({ "Last-Event-ID": "0" }),
        signal: controller.signal,
      }),
    );
    expect(accepted).toEqual([
      expect.objectContaining({
        event_id: 1,
        run_id: RUN_ID,
        type: "benchmark.started",
      }),
    ]);
  });

  it("rejects malformed or inconsistent events without publishing partial data", async () => {
    fetchMock
      .mockResolvedValueOnce(responseFromChunks([
        "id: 1\nevent: benchmark.started\ndata: {\"event_id\":1}\n\n",
      ]))
      .mockResolvedValueOnce(responseFromChunks([
        `id: 2\nevent: benchmark.started\ndata: ${eventJson(1)}\n\n`,
      ]));
    const accepted = vi.fn();
    const api = createApiClient();

    await expect(api.streamBenchmarkEvents(
      RUN_ID,
      0,
      new AbortController().signal,
      accepted,
    )).rejects.toThrow(/malformed benchmark event/i);
    await expect(api.streamBenchmarkEvents(
      RUN_ID,
      0,
      new AbortController().signal,
      accepted,
    )).rejects.toThrow(/malformed benchmark event/i);
    expect(accepted).not.toHaveBeenCalled();
  });

  it("stops reading promptly when aborted", async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      pull() {
        return new Promise(() => {});
      },
      cancel() {
        cancelled = true;
      },
    });
    fetchMock.mockResolvedValueOnce(new Response(stream, {
      headers: { "Content-Type": "text/event-stream" },
    }));
    const abort = new AbortController();

    const reading = createApiClient().streamBenchmarkEvents(
      RUN_ID,
      3,
      abort.signal,
      vi.fn(),
    );
    abort.abort();

    await expect(reading).rejects.toMatchObject({ name: "AbortError" });
    expect(cancelled).toBe(true);
  });
});
