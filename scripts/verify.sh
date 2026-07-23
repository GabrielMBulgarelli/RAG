#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Resolve environment"
if [[ -f uv.lock ]]; then
  uv sync --frozen
else
  uv sync
fi

echo "==> Ruff"
uv run ruff check .
uv run ruff format --check .

echo "==> Pyright"
uv run pyright

echo "==> Lanorme"
uvx --python 3.13 --from lanorme==0.14.2 lanorme check .

echo "==> Tests without live Ollama"
uv run pytest -m "not ollama"

echo "==> Offline diagnostics"
if uv run python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('modules.diagnostics') else 1)"; then
  uv run python -m modules.diagnostics --offline
else
  echo "modules.diagnostics does not exist yet; it becomes mandatory in the diagnostics phase."
fi

echo "==> Verification complete"
