# Roadmap

## Status

Version 0.1.0 provides the completed local-first workflows described in the
product definition. This roadmap lists only follow-up work supported by
recorded repository evidence. It does not assign delivery dates or imply a
commitment to hosted or multi-user operation.

## Current priorities

### Enforce model request timeouts end to end

The [live Ollama validation](evidence/live-ollama-validation.md#complete-benchmark)
recorded model-backed case latency beyond the configured request timeout. Future
work should identify the boundary that bypasses or extends that timeout, enforce
the intended behaviour, and demonstrate it with deterministic timeout and
cancellation tests plus a real local-model run.

### Improve structured evidence-grading reliability

The same validation found an incomplete structured evidence grade for an exact
retrieved fixture. The conservative abstention was correct, but the result
supports evaluating stricter output contracts, bounded recovery, or a more
reliable local grading model. Any change must retain abstention safety and show
measured improvement on fixed cases without weakening citation validation.

### Broaden reproducible validation

The current benchmark uses 20 fixed MultiHopRAG development cases on one
recorded local model and hardware environment. Broader claims require additional
case selections, models, and environments while preserving dataset hashes,
case IDs, settings, artifacts, failure classifications, and applicability rules
defined by the [benchmark methodology](benchmark-methodology.md).

## Evidence required for completion

A roadmap item is complete only when its implementation, focused regression
coverage, full `./scripts/verify.sh` result, and relevant benchmark or live-model
evidence are recorded. A changed aggregate without fixed inputs and inspectable
case-level artifacts is not evidence of improvement.
