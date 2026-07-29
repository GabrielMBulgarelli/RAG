import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(cleanup);

Object.defineProperty(URL, "createObjectURL", {
  configurable: true,
  value: vi.fn(() => "blob:workspace-export"),
});

Object.defineProperty(URL, "revokeObjectURL", {
  configurable: true,
  value: vi.fn(),
});

if (!HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function showModal() {
    this.open = true;
  };
}

if (!HTMLDialogElement.prototype.close) {
  HTMLDialogElement.prototype.close = function close() {
    this.open = false;
  };
}
