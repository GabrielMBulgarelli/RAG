#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Resolve environment"
if [[ -f uv.lock ]]; then
  uv sync --frozen
else
  uv sync
fi

echo "==> Frontend dependencies"
npm --prefix frontend ci

echo "==> Frontend tests and build"
npm --prefix frontend test -- --run
npm --prefix frontend run typecheck
npm --prefix frontend run build
npm --prefix frontend audit --audit-level=low

echo "==> Responsive browser tests"
if [[ "${PLAYWRIGHT_INSTALL_WITH_DEPS:-0}" == "1" ]]; then
  npm --prefix frontend exec playwright install --with-deps chromium
else
  npm --prefix frontend exec playwright install chromium
fi
npm --prefix frontend run test:e2e

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
uv run python -c "from modules.config import config; from modules.run import collect_runtime_diagnostics; assert collect_runtime_diagnostics(config, check_ollama=False) == []"

echo "==> Verification complete"
