# 🖥️ Installation

Hypster is a lightweight package, mainly dependent on `Pydantic` for type-checking.

## Basic Installation

```bash
pip install hypster
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)

## Interactive Jupyter UI

Hypster comes with an interactive Jupyter Notebook UI to make configuration selection as easy as :pie:

### Installation

```bash
pip install hypster[jupyter]
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)

### In your notebook

Here's an example of an interactive UI for a [modular-RAG](https://github.com/gilad-rubin/modular-rag) configuration.

```python
from hypster.ui import interactive_config
results = interactive_config(my_config)
```

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

## Development

Interested in **contributing to Hypster?** Go ahead and install the full development suite using:

```bash
pip install hypster[dev]
```

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
* [ruff](https://github.com/astral-sh/ruff)
* [mypy](https://github.com/python/mypy)
* [pytest](https://github.com/pytest-dev/pytest)

