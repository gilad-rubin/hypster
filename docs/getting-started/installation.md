# Installation

Hypster's core package has no runtime dependencies and supports Python 3.10, 3.11, and 3.12.

## Install With uv

```bash
uv add hypster
```

Install Optuna support when you want hyperparameter optimization:

```bash
uv add "hypster[optuna]"
```

Install the notebook visualization extra when you want Hypster's interactive instantiation UI:

```bash
uv add "hypster[viz]"
```

The `viz` extra installs `anywidget`, `ipywidgets`, and `jupyterlab_widgets` for Jupyter Notebook, JupyterLab, and VS Code notebooks.

If you are starting a notebook project from scratch, install the notebook frontend and Hypster widget runtime together:

```bash
uv add "hypster[viz]" jupyterlab
```

## Install With pip

Check that `python` points at a supported interpreter first:

```bash
python --version
```

Hypster supports Python 3.10, 3.11, and 3.12.

```bash
pip install hypster
```

Optional extras:

```bash
pip install "hypster[optuna]"
pip install "hypster[viz]"
```

For a new JupyterLab environment:

```bash
python -m pip install "hypster[viz]" jupyterlab
```

## Verify The Install

Run a smoke test in the same environment where your project runs:

```python
from dataclasses import dataclass

from hypster import HP, explore, instantiate


@dataclass(frozen=True)
class DataSettings:
    path: str
    batch_size: int


def config(hp: HP) -> DataSettings:
    return DataSettings(
        path=hp.text("data/train.csv", name="path"),
        batch_size=hp.int(32, name="batch_size", min=1),
    )


explore(config)
settings = instantiate(config, values={"batch_size": 64})
assert settings.batch_size == 64
```

Expected tree:

```text
config
├── path: text = 'data/train.csv'
└── batch_size: int = 32  (min: 1)
```

## Check The Version

With uv:

```bash
uv run python -c "import hypster; print(hypster.__version__)"
```

With pip or a manually managed interpreter:

```bash
python -c "import hypster; print(hypster.__version__)"
```

Inside Python:

```python
import hypster

print(hypster.__version__)
```

## Development Setup

```bash
git clone https://github.com/gilad-rubin/hypster.git
cd hypster
uv sync --all-extras --dev
uv run pytest
```

Hypster's maintainer tooling lives in local `uv` dependency groups rather than a published `dev` extra, so a `pip`-based setup installs runtime extras and maintainer tools separately:

```bash
git clone https://github.com/gilad-rubin/hypster.git
cd hypster
python -m pip install -e ".[viz,optuna]"
python -m pip install pytest pytest-cov "coverage[toml]" ruff mypy pre-commit pytest-codspeed
```

Adjust the extras in the first command if you only need a subset of the optional runtime integrations.

## Troubleshooting

If `import hypster` fails, check that your package manager installed Hypster into the interpreter running your code:

```bash
python -m pip show hypster
python -c "import hypster; print(hypster.__version__)"
```

If Optuna imports fail, install the optional extra:

```bash
python -m pip install "hypster[optuna]"
```

If `interact()` says the visualization extra is missing, install the notebook widget extra and restart the notebook kernel:

```bash
python -m pip install "hypster[viz]"
```

For notebook UI issues, make sure your notebook frontend is installed:

```bash
# JupyterLab
uv add jupyterlab

# Classic Jupyter Notebook
uv add notebook
```

With pip:

```bash
python -m pip install -U jupyterlab
python -m pip install -U notebook
```

In VS Code notebooks, the Jupyter extension may ask to enable downloads for `anywidget` support files the first time the widget renders. Accept that prompt, then rerun the cell.

Hypster config functions are plain Python. No CLI, project initialization, or config file format is required.
