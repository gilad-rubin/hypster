# 🖥️ Installation

Hypster is a lightweight package, mainly dependent on `Pydantic` for type-checking.

{% tabs %}
{% tab title="Basic Installation" %}
```bash
pip install hypster
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
{% endtab %}

{% tab title="Interactive Jupyter UI" %}
Hypster comes with an interactive **Jupyter Notebook UI** to make instantiation as easy as :pie:

```bash
pip install hypster[jupyter]
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
{% endtab %}

{% tab title="Development" %}
Interested in **contributing to Hypster?** Go ahead and install the full development suite using:

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

\#TODO: consider adding more content here! it's pretty empty :)
