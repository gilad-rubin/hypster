# Contributing

## Setup

```bash
uv pip install -e ".[dev]"
uv run pre-commit install
```

## Checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/hypster
uv run pytest
```

- Add a note to `CHANGELOG.md` for user-facing changes.
- Prefer small, focused PRs with tests.
