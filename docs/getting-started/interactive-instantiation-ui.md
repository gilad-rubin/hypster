# Interactive Instantiation UI

Use `interact()` in a Jupyter notebook when you want to instantiate a configuration through a live widget UI.

```python
from hypster import HP, interact


def model_cfg(hp: HP):
    provider = hp.select(
        ["openai", "gemini"],
        name="provider",
        default="openai",
        description="Chooses which provider branch is active.",
    )
    if provider == "gemini":
        model = hp.select(["flash-lite", "pro"], name="model", default="flash-lite")
    else:
        model = hp.select(["gpt-4o-mini", "gpt-4.1"], name="model", default="gpt-4o-mini")

    temperature = hp.float(0.2, name="temperature", min=0.0, max=1.0)
    return {"provider": provider, "model": model, "temperature": temperature}


result = interact(model_cfg)
```

`interact()` returns an interactive result handle, not the raw configured object. After changing the widget, read the current applied object and replayable selected params from Python:

```python
result.value
result.params
```

`result.params` is a flat dotted-path dictionary that can be replayed through `instantiate(..., values=result.params)` or logged to experiment-tracking tools.

## Applying Changes

By default, widget changes apply immediately. Valid changes update `result.value` and `result.params` in the running kernel.

Use manual apply mode when you want to stage widget edits before updating the applied result:

```python
result = interact(model_cfg, auto_apply=False)
```

In manual mode, the UI continues to explore draft values so dependent controls stay current, but `result.value` and `result.params` keep returning the last applied state until Apply succeeds.

## Continuing An Interaction

Call `result.interact()` to render another live view of the same interaction:

```python
result.interact()
```

To start a fresh session from a previous selection, pass selected params explicitly:

```python
result2 = interact(model_cfg, values=result.params)
```
