from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

from modules import run
from modules.config import Settings


def test_runtime_diagnostics_allow_empty_upload_directory_without_importing_app(
    tmp_path: Path,
) -> None:
    settings = Settings(
        sources_dir=tmp_path / "missing",
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )

    diagnostics = run.collect_runtime_diagnostics(settings, check_ollama=False)

    assert diagnostics == []


def test_runtime_diagnostics_report_unavailable_ollama(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    settings = Settings(
        sources_dir=sources,
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )

    def unavailable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(run, "urlopen", unavailable)

    diagnostics = run.collect_runtime_diagnostics(settings)

    assert diagnostics == [
        f"Ollama is unavailable at {settings.ollama_base_url}. "
        "Start it with `ollama serve` and pull the configured models."
    ]


def test_main_can_launch_before_documents_are_uploaded(
    tmp_path: Path,
    capsys,
) -> None:
    settings = Settings(
        sources_dir=tmp_path / "missing",
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )

    exit_code = run.main(
        settings=settings,
        check_ollama=False,
        app_runner=lambda: 0,
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_main_runs_injected_application_after_prerequisites_pass(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    settings = Settings(sources_dir=sources, _env_file=None)  # pyright: ignore[reportCallIssue]
    calls: list[str] = []

    exit_code = run.main(
        settings=settings,
        check_ollama=False,
        app_runner=lambda: calls.append("launched") or 0,
    )

    assert exit_code == 0
    assert calls == ["launched"]


def test_main_launches_ui_when_ollama_is_unavailable(tmp_path: Path, monkeypatch, capsys) -> None:
    settings = Settings(
        sources_dir=tmp_path / "sources",
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )
    calls: list[str] = []

    monkeypatch.setattr(
        run,
        "collect_runtime_diagnostics",
        lambda *_args, **_kwargs: ["Ollama is unavailable."],
    )

    exit_code = run.main(settings=settings, app_runner=lambda: calls.append("launched") or 0)

    assert exit_code == 0
    assert calls == ["launched"]
    assert "started with limited readiness" in capsys.readouterr().err


def test_load_application_runner_resolves_facade(monkeypatch) -> None:
    def expected_runner() -> int:
        return 0

    monkeypatch.setattr(
        run.importlib,
        "import_module",
        lambda module_name: (
            SimpleNamespace(main=expected_runner) if module_name == "modules.app" else None
        ),
    )

    assert run.load_application_runner() is expected_runner
