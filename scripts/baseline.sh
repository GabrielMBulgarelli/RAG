#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

run() {
  echo
  echo ">>> $*"
  "$@"
  status=$?
  echo "exit_status=$status"
  return 0
}

run git status --short
run find . -maxdepth 4 -type f -not -path "./.git/*" -print
run python --version
run uv --version
run ollama list
if [[ -f pyproject.toml ]]; then
  run uv sync
  run uv run pytest
  run uv run python -m modules.run
else
  echo
  echo ">>> No pyproject.toml; UV project initialization belongs to Phase 1."

  if [[ -f requirements.txt ]]; then
    run sed -n '1,240p' requirements.txt
  fi

  run python -m pytest
  run python -m modules.run
fi
