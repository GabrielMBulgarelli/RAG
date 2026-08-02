"""Production dependency composition for the local workspace."""

from modules.api.dependencies import ApplicationContainer
from modules.application.benchmark_manager import BenchmarkManager
from modules.application.full_rag_benchmark import FullRagBenchmarkExecutor
from modules.application.operation_coordinator import WorkspaceOperationCoordinator
from modules.application.workspace_service import WorkspaceService
from modules.config import Settings, config


def create_application_container(settings: Settings = config) -> ApplicationContainer:
    coordinator = WorkspaceOperationCoordinator()
    workspace: WorkspaceService | None = None
    benchmarks = BenchmarkManager(
        executor=FullRagBenchmarkExecutor(
            settings=settings,
            chat_model_provider=lambda: (
                (workspace.active_chat_model if workspace is not None else None)
                or settings.llm_model
            ),
        ),
        settings=settings,
        coordinator=coordinator,
    )
    workspace = WorkspaceService(
        settings=settings,
        coordinator=coordinator,
        completed_benchmark_probe=benchmarks.has_completed_benchmark,
    )
    return ApplicationContainer(
        workspace=workspace,
        benchmarks=benchmarks,
    )
