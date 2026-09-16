from __future__ import annotations

import tomllib
from pathlib import Path

from great_kingdom_ai import __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_console_scripts_are_current_entrypoints() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    assert pyproject["project"]["scripts"] == {
        "great-kingdom-play": "great_kingdom_ai.cli:main",
        "great-kingdom-evaluate": "great_kingdom_ai.evaluate:main",
        "great-kingdom-export-onnx": "great_kingdom_ai.onnx_export:main",
        "great-kingdom-klent": "great_kingdom_ai.klent.cli:main",
    }
