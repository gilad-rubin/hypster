# Interactive Instantiation UI

Use `interact()` in a notebook when you want to instantiate a configuration through a live widget UI.

`interact()` uses the same pure-Python define-by-run model as `explore()`: it runs the config to discover the currently reachable controls. In auto-apply mode it can rerun the config on every valid widget change, and in manual mode it still explores draft changes so dependent controls stay current. Keep configs used with `interact()` fast and side-effect-free.

Install the notebook renderer with the visualization extra:

{% code overflow="wrap" %}
```bash
uv add "hypster[viz]"
```
{% endcode %}

or:

{% code overflow="wrap" %}
```bash
pip install "hypster[viz]"
```
{% endcode %}

The `viz` extra installs the widget runtime needed by Jupyter Notebook, JupyterLab, and VS Code notebooks. In VS Code, the Jupyter extension may ask to enable downloads for `anywidget` support files the first time a widget is displayed. Accept that prompt, then rerun the cell.

## Start An Interaction

{% code overflow="wrap" %}
```python
from hypster import HP, interact
from my_app.llms import AnthropicClient, OpenAIClient


def openai_config(hp: HP) -> OpenAIClient:
    model_name = hp.select(
        ["gpt-5.5-mini", "gpt-5.5"],
        name="model_name",
        default="gpt-5.5-mini",
        options_only=True,
    )
    temperature = hp.float(0.2, name="temperature", min=0.0, max=1.0)
    cache = hp.bool(True, name="cache")
    return OpenAIClient(model=model_name, temperature=temperature, cache=cache)


def anthropic_config(hp: HP) -> AnthropicClient:
    model_name = hp.select(
        ["claude-sonnet", "claude-opus"],
        name="model_name",
        default="claude-sonnet",
        options_only=True,
    )
    temperature = hp.float(0.2, name="temperature", min=0.0, max=1.0)
    cache = hp.bool(True, name="cache")
    return AnthropicClient(model=model_name, temperature=temperature, cache=cache)


model_options = {
    "openai": openai_config,
    "anthropic": anthropic_config,
}


def model_config(hp: HP):
    selected_config = hp.select(
        model_options,
        name="provider",
        default="openai",
        options_only=True,
        description="Chooses which provider branch is active.",
    )
    return hp.nest(selected_config, name="model")


result = interact(model_config)
```
{% endcode %}

`interact()` returns an interactive result handle, not the raw configured object. After changing the widget, read the current applied object and replayable selected params from Python:

{% code overflow="wrap" %}
```python
client = result.value
params = result.params
```
{% endcode %}

`result.value` has the same type as the config function return value. In this example it is an initialized `OpenAIClient` or `AnthropicClient`.

`result.params` is a flat dotted-path dictionary that can be replayed through `instantiate(..., values=result.params)` or logged to experiment-tracking tools.

## Applying Changes

By default, widget changes apply immediately. Valid changes update `result.value` and `result.params` in the running kernel.

Use manual apply mode when you want to stage widget edits before updating the applied result:

{% code overflow="wrap" %}
```python
result = interact(model_config, auto_apply=False)
```
{% endcode %}

In manual mode, the UI continues to explore draft values so dependent controls stay current, but `result.value` and `result.params` keep returning the last applied state until Apply succeeds.

If a widget selection is invalid, the UI shows the current error. In auto-apply mode, `result.value` and `result.params` raise that error until the widget state is fixed; snapshots report `selected_params=None` so renderer code does not confuse stale params with the current invalid state.

| State | What the widget shows | `result.value` / `result.params` |
| --- | --- | --- |
| Auto-apply, valid edit | The edit is applied immediately. | Updated to the new applied value and params. |
| Auto-apply, invalid edit | The UI shows the validation error. | Raise `RuntimeError` until the state is fixed. |
| Manual mode, valid draft edit | Controls update and dependent branches are explored. | Keep returning the last applied value and params until Apply succeeds. |
| Manual mode, invalid draft edit | The UI shows a draft error and disables Apply. | Keep returning the last successfully applied value and params. |
| Manual mode, Apply succeeds | The draft becomes the applied state. | Updated to the new applied value and params. |
| Manual mode, Apply fails during instantiation | The UI shows the apply error. | Raise `RuntimeError` until a valid state is applied. |

## Continuing An Interaction

Call `result.interact()` to render another live view of the same interaction:

{% code overflow="wrap" %}
```python
result.interact()
```
{% endcode %}

To start a fresh session from a previous selection, pass selected params explicitly:

{% code overflow="wrap" %}
```python
result2 = interact(model_config, values=result.params)
```
{% endcode %}

For framework-specific UIs outside notebooks, use the schema returned by `explore(..., return_info=True)`. See [Build an Interactive UI](../how-to/build-an-interactive-ui.md).
