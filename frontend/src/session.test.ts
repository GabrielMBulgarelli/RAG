import { describe, expect, it, vi } from "vitest";

import { getSessionId } from "./session";

describe("workspace session id", () => {
  it("persists one cryptographic UUID in session storage", () => {
    const storage = new Map<string, string>();
    const sessionStorage = {
      getItem: vi.fn((key: string) => storage.get(key) ?? null),
      setItem: vi.fn((key: string, value: string) => storage.set(key, value)),
    };

    const first = getSessionId(sessionStorage);
    const second = getSessionId(sessionStorage);

    expect(first).toMatch(/^[0-9a-f-]{36}$/);
    expect(second).toBe(first);
    expect(sessionStorage.setItem).toHaveBeenCalledTimes(1);
  });

  it("keeps a stable in-memory UUID when session storage is unavailable", () => {
    const unavailableStorage = {
      getItem: vi.fn(() => {
        throw new DOMException("Denied");
      }),
      setItem: vi.fn(() => {
        throw new DOMException("Denied");
      }),
    };

    expect(getSessionId(unavailableStorage)).toBe(getSessionId(unavailableStorage));
  });
});
