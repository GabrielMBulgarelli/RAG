import asyncio
from uuid import uuid4

import pytest

from modules.application.errors import BenchmarkUnavailableError
from modules.application.unavailable_benchmarks import UnavailableBenchmarkManager


def test_unavailable_benchmark_manager_owns_no_resources() -> None:
    # Given the production placeholder manager
    manager = UnavailableBenchmarkManager()

    # When its lifecycle is exercised repeatedly
    asyncio.run(manager.start())
    asyncio.run(manager.close())
    asyncio.run(manager.close())


@pytest.mark.parametrize(
    "operation",
    [
        lambda manager: manager.start_benchmark(),
        lambda manager: manager.latest_benchmark(),
        lambda manager: manager.get_benchmark(uuid4()),
        lambda manager: manager.get_case(
            run_id=uuid4(),
            case_id="case-1",
            system_id="dense-rag",
        ),
        lambda manager: manager.stream_events(run_id=uuid4(), last_event_id=None),
        lambda manager: manager.cancel_benchmark(uuid4()),
        lambda manager: manager.download_benchmark(uuid4()),
    ],
)
def test_unavailable_benchmark_operations_raise_a_typed_error(operation) -> None:
    # Given the production placeholder manager
    manager = UnavailableBenchmarkManager()

    # When a benchmark operation is attempted
    with pytest.raises(BenchmarkUnavailableError) as error:
        asyncio.run(operation(manager))

    # Then callers receive the public unavailable contract
    assert error.value.code == "benchmark_unavailable"
    assert error.value.details == {}
