const SESSION_STORAGE_KEY = "rag.workspace.session-id";

type SessionStorage = Pick<Storage, "getItem" | "setItem">;

let fallbackSessionId: string | undefined;
const UUID_PATTERN = (
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
);

function createUuid(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const value = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return [
    value.slice(0, 8),
    value.slice(8, 12),
    value.slice(12, 16),
    value.slice(16, 20),
    value.slice(20),
  ].join("-");
}

export function getSessionId(
  storage?: SessionStorage,
): string {
  try {
    const availableStorage = storage ?? window.sessionStorage;
    const existing = availableStorage.getItem(SESSION_STORAGE_KEY);
    if (existing && UUID_PATTERN.test(existing)) {
      return existing;
    }
    const created = createUuid();
    availableStorage.setItem(SESSION_STORAGE_KEY, created);
    return created;
  } catch {
    fallbackSessionId ??= createUuid();
    return fallbackSessionId;
  }
}
