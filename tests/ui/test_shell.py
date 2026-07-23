from pathlib import Path
from typing import Any, cast

from modules.ui.shell import build_application


class StubApplication:
    pass


def test_application_exposes_only_the_ask_root_route() -> None:
    # Arrange
    interface = build_application(cast(Any, StubApplication()))

    # Act
    pages = cast(list[tuple[str, str, bool]], interface.get_config_file()["pages"])

    # Then
    assert [path for path, _name, _root in pages] == [""]


def test_application_uses_shared_collapsible_sidebar_and_full_shell() -> None:
    # Arrange
    application = cast(Any, StubApplication())

    # Act
    config = build_application(application).get_config_file()

    sidebars = [
        component["props"] for component in config["components"] if component["type"] == "sidebar"
    ]

    # Then
    assert config["title"] == "Local Document RAG"
    assert config["fill_width"] is True
    assert config["fill_height"] is True
    navbars = [
        component["props"] for component in config["components"] if component["type"] == "navbar"
    ]
    html_values = [
        component["props"].get("value", "")
        for component in config["components"]
        if component["type"] == "html"
    ]

    assert len(sidebars) == 1
    assert all(
        sidebar["label"] == "Application navigation"
        and sidebar["open"] is True
        and sidebar["width"] == 280
        and sidebar["elem_id"] == "application-sidebar"
        for sidebar in sidebars
    )
    assert len(navbars) == 1
    assert all(
        navbar["visible"] is False and navbar["main_page_name"] is False for navbar in navbars
    )
    assert all('href="/"' not in value for value in html_values)
    assert all('href="/evaluations"' not in value for value in html_values)
    stylesheet = (Path(__file__).parents[2] / "modules" / "ui" / "app.css").read_text(
        encoding="utf-8"
    )
    assert ":is(aside, .sidebar) nav" not in stylesheet
    assert all('aria-label="Primary"' not in value for value in html_values)
    assert any("Loading Models and Corpus…" in value for value in html_values)
    assert all("Knowledge Base" not in value for value in html_values)
    assert all(
        'href="/ask"' not in value and 'href="/system"' not in value and "Overview" not in value
        for value in html_values
    )


def test_every_sidebar_has_an_automatic_multi_file_uploader() -> None:
    # Arrange
    config = build_application(cast(Any, StubApplication())).get_config_file()
    components = config["components"]

    # Act
    uploaders = [
        component["props"]
        for component in components
        if component["props"].get("elem_id") == "sidebar-document-upload"
    ]
    values = {
        component["props"].get("value")
        for component in components
        if isinstance(component["props"].get("value"), str)
    }

    # Assert the remaining root route exposes only automatic ingestion controls.
    assert len(uploaders) == 1
    uploader = uploaders[0]
    assert uploader["label"] == "Drop files here or click to upload"
    assert uploader["show_label"] is False
    assert uploader["file_count"] == "multiple"
    assert uploader["file_types"] == [".pdf", ".txt"]
    assert all("Add documents" not in value for value in values)
    assert all(
        "PDF or TXT files are indexed as soon as you select them." not in value for value in values
    )
    assert {
        "Add Documents",
        "Reindex Changed Documents",
        "Check Index Consistency",
        "Rebuild Index",
    }.isdisjoint(values)


def test_stylesheet_suppresses_gradio_generated_navbar_fallback() -> None:
    # Act
    stylesheet = (Path(__file__).parents[2] / "modules" / "ui" / "app.css").read_text(
        encoding="utf-8"
    )

    # Then
    assert ".gradio-container > .nav-holder" in stylesheet
    assert "display: none !important" in stylesheet


def test_stylesheet_keeps_application_sidebar_persistent_only_on_desktop() -> None:
    # Act
    stylesheet = (Path(__file__).parents[2] / "modules" / "ui" / "app.css").read_text(
        encoding="utf-8"
    )

    # Then
    assert "@media (min-width: 769px)" in stylesheet
    assert "#application-sidebar" in stylesheet
    assert ".sidebar-parent:has(#application-sidebar)" in stylesheet
    assert "#application-sidebar .toggle-button" in stylesheet
