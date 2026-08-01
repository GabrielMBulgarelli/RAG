# Product Design Context

## Product

Local Document RAG is a desktop-first, locally hosted workbench for indexing
PDF and TXT files, asking grounded questions, reviewing cited evidence, running
the seven-system Full RAG Benchmark, and diagnosing the local Ollama and index
state. It is both a working AI tool and an applied-engineering portfolio
project.

## Users and goals

- A local user can add and manage documents, load a model, ask questions,
  distinguish supported answers from limited answers or abstentions, and
  inspect the evidence.
- Developers and technical reviewers can inspect retrieval decisions, public
  traces, persisted benchmark results, and reproducibility metadata.
- Advanced controls remain available without competing with the conversation.

## Experience principles

1. **One workspace:** the sidebar, conversation, inspector, and overlays share
   one continuous route.
2. **Evidence first:** answer state and cited sources are prominent; retrieval
   and execution detail is secondary and collapsible.
3. **Actionable readiness:** model, uploaded-index, and benchmark-preparation
   states use plain language and identify the next command or action.
4. **Safe operations:** destructive actions require confirmation, and benchmark
   cancellation preserves completed work while the active request exits.
5. **Responsive containment:** the page stays within the viewport and wide
   result regions scroll inside their container.

## Accessibility target

- Target WCAG 2.2 AA.
- Normal text has at least 4.5:1 contrast; large text and non-text boundaries
  have at least 3:1 contrast.
- Interactive controls have a practical minimum height of 44 CSS pixels and a
  visible keyboard focus indicator.
- Dynamic status messages are announced without moving focus.
- Labels and state meaning do not depend on color alone.
- Dialogs contain focus, close predictably, and restore focus to the invoking
  control. Reduced-motion preferences are respected.

## Visual direction

- Use restrained indigo and slate tones with explicit surface and text colors.
- Follow a 4/8-pixel spacing rhythm and keep control interiors compact.
- Keep the conversation visually dominant; use overlays for diagnostics,
  benchmark progress and results, case inspection, and confirmations.
- Avoid decorative dashboard chrome, glass effects, neon gradients, excessive
  cards, and motion that does not communicate state.

## Technical boundaries

- React owns presentation and browser interaction.
- FastAPI owns application services, typed HTTP responses, persistence, and
  production delivery of the built frontend.
- The interface loads without Ollama and exposes limited-readiness guidance.
- Uploaded documents and benchmark data use separate indexes.
- Benchmark work uses the fixed 20-case development set across seven systems.
- No external hosted service or in-application benchmark download/indexing flow
  is part of the current product.
