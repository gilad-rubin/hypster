# 🖥️ Installation

Hypster is a lightweight package with minimal dependencies.

{% tabs %}
{% tab title="Basic Installation (uv)" %}
```bash
uv add hypster
```
{% endtab %}

{% tab title="Basic Installation (pip)" %}
```bash
pip install hypster
```
{% endtab %}

{% tab title="Interactive Jupyter UI (uv)" %}
Hypster comes with an interactive **Jupyter Notebook UI** to make instantiation as easy as :pie:

```bash
uv add "hypster[jupyter]"
```

Dependencies:

* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
{% endtab %}

{% tab title="Hyperparameter Optimization (uv)" %}
Install Hypster with Optuna extras:

```bash
uv add 'hypster[optuna]'
```

Or add Optuna directly:

```bash
uv add optuna
```
{% endtab %}

{% tab title="Interactive Jupyter UI (pip)" %}
Hypster comes with an interactive **Jupyter Notebook UI** to make instantiation as easy as :pie:

```bash
pip install hypster[jupyter]
```

Dependencies:

* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
{% endtab %}

{% tab title="Development (uv)" %}
Interested in **contributing to Hypster?** Go ahead and install the full development suite using:

```bash
uv add "hypster[dev]"
```

Dependencies:

* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
* [ruff](https://github.com/astral-sh/ruff)
* [mypy](https://github.com/python/mypy)
* [pytest](https://github.com/pytest-dev/pytest)
{% endtab %}

{% tab title="Development (pip)" %}
Interested in **contributing to Hypster?** Go ahead and install the full development suite using:

```bash
pip install hypster[dev]
```

Dependencies:

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

### With uv (recommended):

1. Ensure uv is up to date:

```bash
uv self update
```

2. Ensure `hypster` is up to date

```bash
uv add --upgrade hypster
```

3. For Jupyter-related issues, make sure Jupyter is properly installed:

```bash
# For JupyterLab
uv add jupyterlab

# Or 'classic' Jupyter Notebook
uv add notebook
```

### With pip:

1. Ensure your pip is up to date:

```bash
pip install -U pip
```

2. Ensure `hypster` is up to date

```bash
pip install -U hypster
```

3. For Jupyter-related issues, make sure Jupyter is properly installed:

```bash
# For JupyterLab
pip install -U jupyterlab

# Or 'classic' Jupyter Notebook
pip install -U notebook
```

3. If you're still having problems, please [open an issue](https://github.com/gilad-rubin/hypster/issues) on our GitHub repository.
