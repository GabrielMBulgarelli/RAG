# React single-workspace client design

## Scope

Build a React, TypeScript, and Vite client in `frontend/` for the existing
presentation-neutral FastAPI workspace endpoints. The client has one permanent
workspace view and does not add routing, benchmark results, FastAPI static
serving, or changes to the existing Gradio UI.

## Architecture

The client uses four layers:

1. `api/types.ts` mirrors the existing JSON contracts without adding a second
   backend or view-model API.
2. `api/client.ts` owns fetch, multipart uploads, URL encoding, problem-envelope
   parsing, and blob downloads.
3. `workspace/useWorkspace.ts` owns the stable session UUID, startup reads,
   mutations, conversation history, selected answer/source, busy/error states,
   and refresh rules.
4. `App` composes focused presentational components and owns the single overlay
   discriminated union.

No router, context provider, reducer framework, query library, global state
library, component framework, or CSS framework is needed.

## State and data flow

- Startup fetches runtime and documents.
- One session UUID is created with `crypto.randomUUID` and persisted in
  `sessionStorage`, with an in-memory fallback when storage is unavailable.
- Query submission trims the question, appends the user message immediately,
  then appends the returned assistant message and selects its response for the
  inspector. Failure keeps the user message and exposes a retry action.
- Upload, delete, and model-load mutations refresh both runtime and documents
  after success.
- Diagnostics refresh only on demand.
- Clear removes local history only after the API succeeds.
- Export creates and revokes an object URL and triggers a download without
  navigation.
- Capabilities and active operations disable conflicting controls.
- `Run benchmark` is enabled only when `can_run_benchmark` is true and a real
  `onRunBenchmark` callback is supplied. Otherwise it is visibly disabled with
  concise accessible explanation.

## Composition and overlays

The single workspace contains:

- a compact header with product name and model readiness;
- a fixed document/runtime/action sidebar;
- the dominant conversation panel and composer;
- a collapsible Sources/Details inspector;
- one central overlay controller.

`App` owns:

```text
null
| { kind: "diagnostics" }
| { kind: "document-details"; documentId: string }
| { kind: "delete-confirmation"; documentId: string }
```

Sidebar controls dispatch intents and never mount duplicate hidden dialogs.
Native dialogs provide Escape, focus, accessible names, and backdrop behavior.

## Visual system

- Canvas `#F4F7FA`, panels `#FFFFFF`, ink `#18212A`, muted text `#647281`,
  hairlines `#DCE3EA`.
- Spectral blue `#345BFF` owns actions and focus; teal `#0E8A7A` owns evidence;
  amber `#B56A12` owns warnings.
- Display: `Bahnschrift SemiCondensed`; body: `Segoe UI Variable`, `Aptos`,
  sans-serif; trace/data: `Cascadia Mono`.
- Desktop uses a compact fixed sidebar and approximately 72/28
  conversation/inspector split.
- Below the responsive breakpoint, the sidebar is a drawer and the inspector
  becomes a stacked bottom region.
- The signature evidence spine uses matching citation/source numbers and active
  teal treatment to connect answer evidence with the inspector.
- Motion is minimal and removed under `prefers-reduced-motion`.

## Accessibility and safety

Landmarks, visible labels, native buttons/dialogs/details elements, 40px
practical targets, visible focus rings, keyboard submission, and non-color
state labels are required. The client never renders unsafe HTML; raw trace data
is formatted sanitized JSON text.

## Testing

Vitest, jsdom, Testing Library, and user-event cover:

- exact API/error/session behavior;
- startup, empty, ready, busy, and unavailable states;
- query success/failure, citations, inspector tabs/accordions/collapse;
- uploads, document overlays/deletion, model load, diagnostics;
- clear/export and object URL lifecycle;
- keyboard composer and dialog behavior;
- TypeScript and production Vite builds.

Python API/application tests and the full non-Ollama suite verify backend
preservation.
