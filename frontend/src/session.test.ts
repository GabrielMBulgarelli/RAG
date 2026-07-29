import { afterEach, describe, expect, it, vi } from "vitest";

import { getSessionId } from "./session";

describe("workspace session id", () => {
  const originalSessionStorage = Object.getOwnPropertyDescriptor(window, "sessionStorage");

  afterEach(() => {
    if (originalSessionStorage) {
      Object.defineProperty(window, "sessionStorage", originalSessionStorage);
    }
  });

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

  it("falls back when reading the window sessionStorage property throws", () => {
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get: () => {
        throw new DOMException("Storage access denied");
      },
    });

    expect(() => getSessionId()).not.toThrow();
    expect(getSessionId()).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it("replaces an invalid stored session identifier", () => {
    const values = new Map([["rag.workspace.session-id", "not-a-uuid"]]);
    const storage = {
      getItem: vi.fn((key: string) => values.get(key) ?? null),
      setItem: vi.fn((key: string, value: string) => values.set(key, value)),
    };

    const sessionId = getSessionId(storage);

    expect(sessionId).not.toBe("not-a-uuid");
    expect(sessionId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
    expect(storage.setItem).toHaveBeenCalledWith("rag.workspace.session-id", sessionId);
  });
});
