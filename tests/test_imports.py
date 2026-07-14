import importlib


def test_application_modules_import_as_a_package() -> None:
    for module_name in (
        "modules.config",
        "modules.error_handler",
        "modules.tools",
        "modules.vector_db",
        "modules.rag_graph",
        "modules.app",
        "modules.run",
    ):
        importlib.import_module(module_name)
