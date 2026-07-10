import tomllib
from pathlib import Path

from jstudio.ui.app import create_application, main


def test_console_entry_point_and_application_factory_are_exposed():
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["scripts"]["j-studio"] == "jstudio.ui.app:main"
    assert callable(create_application)
    assert callable(main)
