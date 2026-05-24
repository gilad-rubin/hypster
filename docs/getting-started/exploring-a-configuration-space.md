# Explore A Configuration Space

Use `explore()` to inspect the parameters that are reachable for the active branch of a config function. Because Hypster configs are pure Python rather than a declarative DSL, exploration discovers the schema by executing the config with a schema-recording `HP`. It follows the same Python conditionals, loops, and helper calls as `instantiate()`.

That execution should be cheap and safe. Keep side effects, paid API calls, database work, file writes, training, and costly object construction outside the config paths you explore.

## Print The Parameter Tree

{% code overflow="wrap" %}
```python
from hypster import HP, explore
from my_app.backends import Application, LocalBackend, RemoteBackend

def local_backend(hp: HP) -> LocalBackend:
    threads = hp.int(4, name="threads", min=1, max=64)
    cache = hp.bool(True, name="cache")
    return LocalBackend(threads=threads, cache=cache)

def remote_backend(hp: HP) -> RemoteBackend:
    endpoint = hp.text("https://api.example.com", name="endpoint")
    timeout = hp.float(10.0, name="timeout", min=0.1, max=120.0)
    return RemoteBackend(endpoint=endpoint, timeout=timeout)

backend_options = {
    "local": local_backend,
    "remote": remote_backend,
}

def app_config(hp: HP) -> Application:
    selected_config = hp.select(backend_options, name="backend", default="local", options_only=True)
    backend = hp.nest(selected_config, name="backend_settings")
    return Application(backend=backend)

explore(app_config)
```
{% endcode %}

Expected output:

{% code overflow="wrap" %}
```text
app_config
├── backend: select = "local"  (options: ["local", "remote"])
└── backend_settings
    ├── threads: int = 4  (1-64)
    └── cache: bool = True
```
{% endcode %}

## Explore A Different Conditional Branch

Pass `values=` to choose a branch before tracing it:

{% code overflow="wrap" %}
```python
explore(
    app_config,
    values={"backend": "remote", "backend_settings.timeout": 30.0},
)
```
{% endcode %}

Expected output:

{% code overflow="wrap" %}
```text
app_config
├── backend: select = "remote"  (options: ["local", "remote"])
└── backend_settings
    ├── endpoint: text = "https://api.example.com"
    └── timeout: float = 30.0  (0.1-120.0)
```
{% endcode %}

## Get Structured Metadata

Use `return_info=True` when you want to inspect the schema in code:

{% code overflow="wrap" %}
```python
info = explore(app_config, return_info=True)

print(info.defaults())
print(info.to_dict())
```
{% endcode %}

For programmatic inspection before instantiation, use `schema = explore(config, return_info=True)` and read `schema.to_dict()["parameters"]`. Use plain `explore(config)` when a printed tree is enough.

`defaults()` returns a flat dictionary of the active branch's default parameter values:

{% code overflow="wrap" %}
```python
{
    "backend": "local",
    "backend_settings.threads": 4,
    "backend_settings.cache": True,
}
```
{% endcode %}

## When To Use `explore()` vs `instantiate()`

| Use | API |
| --- | --- |
| Print the active tree | `explore(config)` |
| Build a UI or schema export | `explore(config, return_info=True)` |
| Inspect a conditional branch | `explore(config, values={...})` |
| Get the runtime object | `instantiate(config, values={...})` |

By default, `explore()` raises when `values=` contains unknown names or overrides for a branch that was not reached. Pass `on_unknown="warn"` to inspect while warning, or `on_unknown="ignore"` to silence it.

## Notes

* Exploration runs your config function. Keep config functions fast and side-effect-free, just like you would for HPO or interactive UI generation.
* Avoid paid API calls, database calls, file writes, training loops, and costly resource initialization in code paths that `explore()` will execute.
* `explore()` records `hp.*` parameters, nested groups, defaults, selected values, options, and numeric bounds.
* Select choices are converted to JSON-friendly values in `to_dict()`.
* `explore()` does not instantiate external services unless your config function does so directly.
