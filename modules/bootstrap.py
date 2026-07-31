"""Production dependency composition for the local workspace."""

from modules.api.dependencies import ApplicationContainer
from modules.application.operation_coordinator import WorkspaceOperationCoordinator
from modules.application.unavailable_benchmarks import UnavailableBenchmarkManager
from modules.application.workspace_service import WorkspaceService
from modules.config import Settings, config


def create_application_container(settings: Settings = config) -> ApplicationContainer:
    coordinator = WorkspaceOperationCoordinator()
    return ApplicationContainer(
        workspace=WorkspaceService(
            settings=settings,
            coordinator=coordinator,
            benchmark_available=False,
        ),
        benchmarks=UnavailableBenchmarkManager(),
    )
