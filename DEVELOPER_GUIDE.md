# Developer Guide - Using uv

This project has been converted to use [uv](https://docs.astral.sh/uv/) as the package manager instead of poetry/conda/pip.

## Quick Start

### Installing Dependencies

```bash
# Install all dependencies
uv sync

# Install with development dependencies
uv sync --extra dev

# Install with jupyter dependencies
uv sync --extra jupyter
```

### Running Commands

```bash
# Run tests
uv run pytest

# Run python scripts
uv run python your_script.py

# Run linting
uv run ruff check
uv run mypy src/

# Build the package
uv build
```

### Managing Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add an optional dependency
uv add --optional jupyter package-name

# Remove a dependency
uv remove package-name
```

### Installing the Package

Users can install the package using either uv or pip:

```bash
# Using uv (recommended)
uv add hypster

# With extras
uv add 'hypster[jupyter]'
uv add 'hypster[dev]'

# Using pip (traditional)
pip install hypster
pip install hypster[jupyter]
pip install hypster[dev]
```
