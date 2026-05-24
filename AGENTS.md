# Agent Notes

## Version Handling

- `hypster.__version__` is intentionally defined in `src/hypster/_version.py` instead of using `importlib.metadata` at import time.
- When changing `[project].version` in `pyproject.toml`, update `src/hypster/_version.py` in the same change.
- Keep `src/hypster/__init__.py` free of eager `importlib.metadata` imports; they significantly slow down `import hypster`.
- Run `uv run pytest tests/test_version.py` after version-related changes.

## Positioning And README Work

- Use `dev/positioning/` as the source of raw positioning notes before changing the GitHub README, docs front page, or product messaging.
- Keep public copy technical and concrete: lead with code and benefits, avoid salesy language, and preserve the FastAPI/FastMCP-style README shape.
