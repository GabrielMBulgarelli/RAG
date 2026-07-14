"""Legacy setup entry point retained as a safe migration aid."""

import sys


def main() -> int:
    """Direct users to the reproducible UV workflow without mutating the runtime."""
    print(
        "Runtime dependency installation has been removed. "
        "Run `uv python install 3.12` followed by `uv sync`, then start with "
        "`uv run python -m modules.run`.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
