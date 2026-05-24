from pathlib import Path

import hypster


def _read_pyproject_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    in_project_section = False

    for raw_line in pyproject_path.read_text().splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project_section = True
            continue
        if line.startswith("[") and in_project_section:
            break
        if in_project_section and line.startswith("version"):
            _, value = line.split("=", 1)
            return value.strip().strip('"')

    raise AssertionError("Could not find [project].version in pyproject.toml")


def test_package_version_matches_pyproject():
    assert hypster.__version__ == _read_pyproject_version()
