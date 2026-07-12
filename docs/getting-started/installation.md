# Installation

Hypster's core package has no runtime dependencies and supports Python 3.10, 3.11, 3.12, and 3.13.

## Install With uv

{% code overflow="wrap" %}
```bash
uv add hypster
```
{% endcode %}

Install Optuna support when you want hyperparameter optimization:

{% code overflow="wrap" %}
```bash
uv add "hypster[optuna]"
```
{% endcode %}

Install the notebook visualization extra when you want Hypster's interactive instantiation UI:

{% code overflow="wrap" %}
```bash
uv add "hypster[viz]"
```
{% endcode %}

The `viz` extra installs `anywidget`, `ipywidgets`, and `jupyterlab_widgets` for Jupyter Notebook, JupyterLab, and VS Code notebooks.

If you are starting a notebook project from scratch, install the notebook frontend and Hypster widget runtime together:

{% code overflow="wrap" %}
```bash
uv add "hypster[viz]" jupyterlab
```
{% endcode %}

## Install With pip

Check that `python` points at a supported interpreter first:

{% code overflow="wrap" %}
```bash
python --version
```
{% endcode %}

Hypster supports Python 3.10, 3.11, 3.12, and 3.13.

{% code overflow="wrap" %}
```bash
pip install hypster
```
{% endcode %}

Optional extras:

{% code overflow="wrap" %}
```bash
pip install "hypster[optuna]"
pip install "hypster[viz]"
```
{% endcode %}

For a new JupyterLab environment:

{% code overflow="wrap" %}
```bash
python -m pip install "hypster[viz]" jupyterlab
```
{% endcode %}

## Verify The Install

Run a smoke test in the same environment where your project runs:

{% code overflow="wrap" %}
```python
from pathlib import Path

from hypster import HP, explore, instantiate


def config(hp: HP) -> Path:
    data_dir = hp.text("data", name="data_dir")
    split = hp.select(["train", "validation"], name="split")
    return Path(data_dir) / f"{split}.csv"


explore(config)
path = instantiate(config, values={"split": "validation"})
assert path == Path("data/validation.csv")
```
{% endcode %}

Expected tree:

{% code overflow="wrap" %}
```text
config
├── data_dir: text = "data"
└── split: select = "train"  (options: ["train", "validation"])
```
{% endcode %}

## Check The Version

With uv:

{% code overflow="wrap" %}
```bash
uv run python -c "import hypster; print(hypster.__version__)"
```
{% endcode %}

With pip or a manually managed interpreter:

{% code overflow="wrap" %}
```bash
python -c "import hypster; print(hypster.__version__)"
```
{% endcode %}

Inside Python:

{% code overflow="wrap" %}
```python
import hypster

print(hypster.__version__)
```
{% endcode %}

## Development Setup

{% code overflow="wrap" %}
```bash
git clone https://github.com/gilad-rubin/hypster.git
cd hypster
uv sync --all-extras --dev
uv run pytest
```
{% endcode %}

Hypster's maintainer tooling lives in local `uv` dependency groups rather than a published `dev` extra, so a `pip`-based setup installs runtime extras and maintainer tools separately:

{% code overflow="wrap" %}
```bash
git clone https://github.com/gilad-rubin/hypster.git
cd hypster
python -m pip install -e ".[viz,optuna]"
python -m pip install pytest pytest-cov "coverage[toml]" ruff mypy pre-commit pytest-codspeed
```
{% endcode %}

Adjust the extras in the first command if you only need a subset of the optional runtime integrations.

## Troubleshooting

If `import hypster` fails, check that your package manager installed Hypster into the interpreter running your code:

{% code overflow="wrap" %}
```bash
python -m pip show hypster
python -c "import hypster; print(hypster.__version__)"
```
{% endcode %}

If Optuna imports fail, install the optional extra:

{% code overflow="wrap" %}
```bash
python -m pip install "hypster[optuna]"
```
{% endcode %}

If `interact()` says the visualization extra is missing, install the notebook widget extra and restart the notebook kernel:

{% code overflow="wrap" %}
```bash
python -m pip install "hypster[viz]"
```
{% endcode %}

For notebook UI issues, make sure your notebook frontend is installed:

{% code overflow="wrap" %}
```bash
# JupyterLab
uv add jupyterlab

# Classic Jupyter Notebook
uv add notebook
```
{% endcode %}

With pip:

{% code overflow="wrap" %}
```bash
python -m pip install -U jupyterlab
python -m pip install -U notebook
```
{% endcode %}

In VS Code notebooks, the Jupyter extension may ask to enable downloads for `anywidget` support files the first time the widget renders. Accept that prompt, then rerun the cell.

Hypster config functions are plain Python. No CLI, project initialization, or config file format is required.
