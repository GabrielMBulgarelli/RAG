from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_default_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
