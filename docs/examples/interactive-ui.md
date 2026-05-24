# Interactive UI From Schema

Hypster ships a notebook UI through `interact()`. This example shows the lower-level schema path for custom Streamlit, Gradio, Panel, web app, or internal dashboard UIs.

The public schema hook is `explore(config, return_info=True)`: it returns metadata that you can map to controls and then replay through `instantiate()`.

## Turn A Config Into Field Metadata

{% code overflow="wrap" %}
```python
from hypster import HP, explore, instantiate
from my_app.backends import AppRuntime, LocalBackend, RemoteBackend

def app_config(hp: HP) -> AppRuntime:
    provider = hp.select(["local", "remote"], name="provider", default="local", options_only=True)
    batch_size = hp.int(32, name="batch_size", min=1, max=512)

    if provider == "remote":
        endpoint = hp.text("https://api.example.com", name="endpoint")
        timeout = hp.float(10.0, name="timeout", min=0.1, max=120.0)
        backend = RemoteBackend(endpoint=endpoint, timeout=timeout)
        return AppRuntime(provider=provider, batch_size=batch_size, backend=backend)

    threads = hp.int(4, name="threads", min=1, max=64)
    backend = LocalBackend(threads=threads)
    return AppRuntime(provider=provider, batch_size=batch_size, backend=backend)

def flatten_parameters(parameters):
    fields = []
    for parameter in parameters:
        if parameter["kind"] == "group":
            fields.extend(flatten_parameters(parameter["children"]))
        else:
            fields.append(parameter)
    return fields

schema = explore(app_config, return_info=True)
fields = flatten_parameters(schema.to_dict()["parameters"])

for field in fields:
    print(field["path"], field["kind"], field["selected_value"])
```
{% endcode %}

## Feed UI State Back Into Hypster

Your UI state should be a dictionary whose keys are Hypster parameter paths:

{% code overflow="wrap" %}
```python
ui_values = {
    "provider": "remote",
    "batch_size": 64,
    "endpoint": "https://staging.example.com",
    "timeout": 30.0,
}

schema = explore(app_config, values=ui_values, return_info=True)
cfg = instantiate(app_config, values=ui_values)

assert schema.defaults()["provider"] == "local"
assert cfg.provider == "remote"
assert cfg.backend.timeout == 30.0
```
{% endcode %}

## Control Mapping

| Hypster kind | Typical UI control |
| --- | --- |
| `select` | dropdown, segmented control, radio group |
| `multi_select` | multiselect checklist |
| `int` | integer input or slider |
| `float` | number input or slider |
| `bool` | checkbox or switch |
| `text` | text input or textarea |
| `group` | fieldset, section, accordion, or nested panel |

## Streamlit Shape

This is the adapter shape. It assumes you already installed Streamlit and are running inside a Streamlit app.

{% code overflow="wrap" %}
```python
def render_field(st, field):
    path = field["path"]
    kind = field["kind"]
    value = field["selected_value"]

    if kind == "select":
        options = field["options"] or []
        return st.selectbox(path, options, index=options.index(value))
    if kind == "bool":
        return st.checkbox(path, value=value)
    if kind == "int":
        return st.number_input(path, value=value, step=1)
    if kind == "float":
        return st.number_input(path, value=value)
    if kind == "text":
        return st.text_input(path, value=value)

    raise ValueError(f"Unsupported field kind: {kind}")
```
{% endcode %}

For conditional UIs, rerun `explore(config, values=current_ui_values, return_info=True)` whenever a branch-selecting value changes. That keeps the rendered fields aligned with the active branch.
