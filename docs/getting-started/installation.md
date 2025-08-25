# 🖥️ Installation

Hypster is a lightweight package, mainly dependent on `Pydantic` for type-checking.

{% tabs %}
{% tab title="Basic Installation" %}
Using uv (recommended):
```bash
uv add hypster
```

Or using pip:
```bash
pip install hypster
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
{% endtab %}

{% tab title="Interactive Jupyter UI" %}
Hypster comes with an interactive **Jupyter Notebook UI** to make instantiation as easy as :pie:

Using uv:
```bash
uv add 'hypster[jupyter]'
```

Or using pip:
```bash
pip install hypster[jupyter]
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
{% endtab %}

{% tab title="Development" %}
Interested in **contributing to Hypster?** Go ahead and install the full development suite using:

Using uv:
```bash
uv add 'hypster[dev]'
```

Or using pip:
```bash
pip install hypster[dev]
```

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
* [ruff](https://github.com/astral-sh/ruff)
* [mypy](https://github.com/python/mypy)
* [pytest](https://github.com/pytest-dev/pytest)
{% endtab %}
{% endtabs %}

## Verification

After installation, you can verify your setup by running:

```python
import hypster
print(hypster.__version__)
```

## System Requirements

* Python 3.10 or higher
* Optional: Jupyter Notebook/Lab for interactive features

## Troubleshooting

If you encounter any installation issues:

1. Ensure your package manager is up to date:

Using uv:
```bash
uv self update
```

Or using pip:
```bash
pip install -U pip
```

2. Ensure `hypster` is up to date

Using uv:
```bash
uv add --upgrade hypster
```

Or using pip:
```bash
pip install -U hypster
```

3. For Jupyter-related issues, make sure Jupyter is properly installed:

Using uv:
```bash
# For JupyterLab
uv add jupyterlab

# Or 'classic' Jupyter Notebook
uv add notebook
```

Or using pip:
```bash
# For JupyterLab
pip install -U jupyterlab

# Or 'classic' Jupyter Notebook
pip install -U notebook
```

3. If you're still having problems, please [open an issue](https://github.com/gilad-rubin/hypster/issues) on our GitHub repository.
