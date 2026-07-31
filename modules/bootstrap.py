"""Production dependency composition for the local workspace."""

from modules.api.dependencies import ApplicationContainer
from modules.application.benchmark_manager import BenchmarkManager
from modules.application.full_rag_benchmark import FullRagBenchmarkExecutor
from modules.application.operation_coordinator import WorkspaceOperationCoordinator
from modules.application.workspace_service import WorkspaceService
from modules.config import Settings, config


def create_application_container(settings: Settings = config) -> ApplicationContainer:
    coordinator = WorkspaceOperationCoordinator()
    workspace = WorkspaceService(
        settings=settings,
        coordinator=coordinator,
        benchmark_available=True,
    )
    return ApplicationContainer(
        workspace=workspace,
        benchmarks=BenchmarkManager(
            executor=FullRagBenchmarkExecutor(
                settings=settings,
                chat_model_provider=lambda: workspace.active_chat_model or settings.llm_model,
            ),
            settings=settings,
            coordinator=coordinator,
        ),
    )
