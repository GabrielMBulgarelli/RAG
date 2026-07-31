"""Persistence and human-readable reporting for evaluation experiments."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from modules.config import PROJECT_ROOT
from modules.evaluation_models import (
    CaseResult,
    ExperimentConfig,
    MetricObservation,
    evaluation_result_kind,
)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_experiment(
    results_root: Path,
    experiment: ExperimentConfig,
    results: Sequence[CaseResult],
    metrics: dict[str, dict[str, MetricObservation]],
) -> Path:
    output = results_root / experiment.run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")
    serialized_metrics = {
        system: {name: observation.model_dump(mode="json") for name, observation in values.items()}
        for system, values in metrics.items()
    }
    configuration = experiment.model_dump(mode="json")
    summary_core = {
        "configuration": configuration,
    }
    summary = {
        **summary_core,
        "result_kind": evaluation_result_kind(summary_core),
        "metrics": serialized_metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result_label = (
        "Full RAG Benchmark artifact"
        if summary["result_kind"] == "standard_benchmark"
        else "Custom evaluation"
    )
    lines = [
        f"# Evaluation {experiment.run_id}",
        "",
        f"Result: **{result_label}**",
        "",
        f"Split: `{experiment.evaluated_split}`",
        "",
    ]
    for system, values in metrics.items():
        lines.extend([f"## {system}", ""])
        for name, observation in values.items():
            if observation.status == "measured":
                display = f"{observation.value:.6f} · n={observation.sample_count}"
            elif observation.status == "not_applicable":
                display = "Not applicable"
            else:
                display = "No eligible cases"
            note = f" — {observation.note}" if observation.note else ""
            lines.append(f"- {name}: {display}{note}")
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output
