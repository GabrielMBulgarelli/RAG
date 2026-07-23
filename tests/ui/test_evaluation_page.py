import json
from pathlib import Path
from typing import Any, cast

from modules.config import config as app_config
from modules.ui import application as application_module
from modules.ui import presenters
from modules.ui.application import EvaluationReadiness, RAGApplication
from modules.ui.contracts import (
    EvaluationCategorySummary,
    EvaluationMetricObservation,
    EvaluationMetricSummary,
    EvaluationPageSnapshot,
    EvaluationSummary,
)
from modules.ui.pages import ask_callbacks


def _summary() -> EvaluationSummary:
    return EvaluationSummary(
        result_path="/tmp/standard",
        split="development",
        systems=("dense", "bm25", "hybrid", "agentic"),
        case_count=8,
        result_kind="standard",
        created_at="2026-07-22T12:00:00+00:00",
        chat_model="<model:latest>",
        quality_categories=(
            EvaluationCategorySummary(
                "Retrieval",
                (
                    EvaluationMetricSummary(
                        name="recall_at_5",
                        label="Recall at 5",
                        observations=(
                            EvaluationMetricObservation("dense", 0.75, "measured", 8),
                            EvaluationMetricObservation("bm25", None, "no_eligible_cases", 0),
                            EvaluationMetricObservation("hybrid", None, "missing", None),
                        ),
                    ),
                ),
            ),
            EvaluationCategorySummary("Evidence and grounding", ()),
            EvaluationCategorySummary("Answer quality", ()),
            EvaluationCategorySummary("Workflow cost", ()),
        ),
    )


def test_evaluation_presenter_keeps_metadata_and_exact_metric_states() -> None:
    # Arrange
    snapshot = EvaluationPageSnapshot(
        state="saved_result",
        split="development",
        systems=("dense", "bm25", "hybrid", "agentic"),
        requires_index=True,
        requires_embeddings=True,
        requires_chat=True,
        problems=(),
        latest=_summary(),
        metric_rows=(
            (
                "Retrieval",
                "Recall at 5",
                "75.0% · n=8",
                "— No eligible cases",
                "—",
                "— Not applicable",
            ),
        ),
        failure_rows=(("<case-1>", "Dense", "Simple search", "Semantic", "Retrieval miss"),),
    )

    # Act
    rendered = presenters.render_evaluation_dashboard(snapshot)

    # Then unlike metrics and unavailable states remain explicit.
    assert "Standard benchmark" in rendered.context
    assert "development" in rendered.context
    assert "&lt;model:latest&gt;" in rendered.context
    assert "<model:latest>" not in rendered.context
    assert rendered.context.count('class="evaluation-context__item"') == 6
    assert "75.0%" in rendered.matrix
    assert "n=8" in rendered.matrix
    assert "No eligible cases" in rendered.matrix
    assert "Not applicable" in rendered.matrix
    assert "Missing observation" in rendered.matrix
    assert not hasattr(rendered, "categories")
    assert not hasattr(rendered, "failures")


def test_evaluation_snapshot_loads_complete_schema_v2_result(tmp_path: Path, monkeypatch) -> None:
    # Arrange
    result = tmp_path / "standard"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "result_kind": "standard_benchmark",
                "configuration": {
                    "run_id": "standard",
                    "dataset_name": "multihop",
                    "evaluated_split": "development",
                    "systems": ["dense", "bm25", "hybrid", "agentic"],
                    "timestamp": "2026-07-22T12:00:00+00:00",
                },
                "metrics": {
                    "dense": {
                        "recall_at_5": {
                            "value": 0.75,
                            "status": "measured",
                            "sample_count": 8,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "system": "dense",
                "route": "simple_search",
                "strategy": "semantic",
                "failure_labels": ["retrieval_miss"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    application = object.__new__(RAGApplication)
    monkeypatch.setattr(
        application,
        "evaluation_readiness",
        lambda _split, _systems, _chat_model=None: EvaluationReadiness(
            state="result",
            latest_result=result,
            systems=("dense", "bm25", "hybrid", "agentic"),
            split="development",
        ),
    )

    # Act
    snapshot = application.evaluation_snapshot(
        "development", ("dense", "bm25", "hybrid", "agentic")
    )

    # Then the immutable snapshot preserves the complete schema-v2 result.
    assert snapshot.latest is not None
    assert snapshot.latest.result_kind == "standard"
    assert snapshot.metric_rows[0][:2] == ("Retrieval", "Recall at 5")
    assert snapshot.metric_rows[0][2] == "75.0% · n=8"
    assert snapshot.failure_rows == (
        ("case-1", "Dense", "Simple search", "Semantic", "Retrieval miss"),
    )


def test_evaluation_empty_state_preserves_actionable_blocking_reason() -> None:
    # Arrange
    snapshot = EvaluationPageSnapshot(
        state="blocked",
        split="development",
        systems=("dense", "bm25", "hybrid", "agentic"),
        requires_index=True,
        requires_embeddings=True,
        requires_chat=True,
        problems=("Start Ollama with: ollama serve",),
        latest=None,
    )

    # Act
    rendered = presenters.render_evaluation_dashboard(snapshot)

    # Then readiness guidance remains visible without a saved result.
    assert "No compatible standard benchmark" in rendered.empty_state
    assert "ollama serve" in rendered.empty_state


def test_ask_evaluation_callback_always_uses_the_standard_configuration() -> None:
    # Arrange
    snapshot = EvaluationPageSnapshot(
        state="ready",
        split="development",
        systems=("dense",),
        requires_index=True,
        requires_embeddings=True,
        requires_chat=False,
        problems=(),
        latest=None,
    )

    class RecordingApplication:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, list[str] | str | None]] = []

        def run_evaluation_snapshot(
            self,
            *,
            split: str,
            systems: list[str] | str | None,
            chat_model: str,
        ) -> EvaluationPageSnapshot:
            self.calls.append(("run", split, systems))
            assert chat_model == app_config.llm_model
            return snapshot

    application = RecordingApplication()

    # Act
    ask_callbacks.run_default_evaluation(cast(Any, application))

    # Then
    assert application.calls == [
        ("run", "development", ["dense", "bm25", "hybrid", "agentic"]),
    ]


def test_evaluation_callbacks_cover_running_success_blocked_and_error_states() -> None:
    # Given the initial running transition

    # When evaluation begins
    run_update, status_update = ask_callbacks.begin_evaluation()

    # Then only the primary action is disabled and progress is visible.
    assert run_update.get("interactive") is False
    assert status_update.get("visible") is True
    assert "Running evaluation" in str(status_update.get("value"))

    expected_statuses = {
        "saved_result": "Evaluation complete",
        "blocked": "Evaluation unavailable",
        "error": "Evaluation could not run",
    }

    class StateApplication:
        def __init__(self, snapshot: EvaluationPageSnapshot) -> None:
            self.snapshot = snapshot

        def run_evaluation_snapshot(self, **_kwargs: Any) -> EvaluationPageSnapshot:
            return self.snapshot

    for state, expected_status in expected_statuses.items():
        snapshot = EvaluationPageSnapshot(
            state=cast(Any, state),
            split="development",
            systems=("dense",),
            requires_index=True,
            requires_embeddings=True,
            requires_chat=False,
            problems=("Readiness problem",) if state != "saved_result" else (),
            latest=_summary() if state == "saved_result" else None,
        )
        updates = ask_callbacks.run_default_evaluation(cast(Any, StateApplication(snapshot)))

        assert len(updates) == 2
        assert updates[0].get("interactive") is True
        assert expected_status in str(updates[1].get("value"))
        assert updates[1].get("visible") is True


def test_custom_run_returns_the_result_it_just_created(tmp_path: Path, monkeypatch) -> None:
    # Arrange
    result = tmp_path / "custom"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "result_kind": "custom",
                "configuration": {
                    "run_id": "custom",
                    "dataset_name": "multihop",
                    "evaluated_split": "development",
                    "systems": ["bm25"],
                    "timestamp": "2026-07-22T12:00:00+00:00",
                },
                "metrics": {
                    "bm25": {
                        "recall_at_5": {
                            "value": 0.5,
                            "status": "measured",
                            "sample_count": 4,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text("", encoding="utf-8")
    application = object.__new__(RAGApplication)
    monkeypatch.setattr(
        application,
        "evaluation_readiness",
        lambda _split, _systems, _chat_model=None: EvaluationReadiness(
            state="ready",
            systems=("bm25",),
            split="development",
        ),
    )
    monkeypatch.setattr(application_module, "run_evaluation", lambda *_args, **_kwargs: result)

    # Act
    snapshot = application.run_evaluation_snapshot(
        split="development",
        systems=["bm25"],
    )

    # Then the page receives this custom result rather than a prior standard run.
    assert snapshot.latest is not None
    assert snapshot.latest.result_path == str(result)
    assert snapshot.latest.result_kind == "custom"
    assert snapshot.systems == ("bm25",)
    assert snapshot.metric_rows[0][3] == "50.0% · n=4"
