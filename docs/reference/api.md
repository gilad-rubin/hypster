# Public API

This page lists the public API exposed by `hypster`.

{% code overflow="wrap" %}
```python
from hypster import HP, InteractiveResult, explore, instantiate, instantiate_with_params, interact
```
{% endcode %}

## Config Function Contract

A config function must be callable and its first positional parameter must be named `hp`. A keyword-only `hp` parameter is rejected before the config executes because Hypster passes the `HP` object positionally.

Config functions are pure Python, not a DSL. `instantiate()`, `explore()`, `interact()`, and the HPO adapter all execute the function to discover or select values, so public API calls inherit any side effects or expensive work in the function body.

{% code overflow="wrap" %}
```python
from hypster import HP

def config(hp: HP) -> int:
    return hp.int(32, name="batch_size")
```
{% endcode %}

The `hp: HP` annotation is recommended but not mandatory. If the first parameter has a type annotation, it must include `HP`. Callable objects are supported when `inspect.signature()` can read their `__call__` signature; signature validation errors use the class name.

Reference examples use small return values for compactness. In application docs and production code, the usual pattern is to return the initialized object your application will use.

Config functions may accept extra keyword-only execution arguments. Pass those directly; Hypster-owned names such as `values`, `on_unknown`, `return_schema`, `auto_apply`, `name`, and `description` are reserved at their API boundaries.

## instantiate

{% code overflow="wrap" %}
```python
instantiate(
    func,
    *,
    values=None,
    on_unknown="raise",
    **kwargs,
)
```
{% endcode %}

Executes a config function and returns whatever the function returns.

| Parameter | Meaning |
| --- | --- |
| `func` | Config function whose first argument is `hp`. |
| `values` | Optional dictionary of parameter paths to overrides. Nested dictionaries are flattened to dotted paths. |
| `on_unknown` | One of `"raise"`, `"warn"`, or `"ignore"`. Defaults to `"raise"`. |
| `**kwargs` | Extra execution arguments forwarded to `func`. |

`values=` keys must match parameters reached during the run. Unknown values and inactive-branch values raise by default.

## instantiate_with_params

{% code overflow="wrap" %}
```python
instantiate_with_params(
    func,
    *,
    values=None,
    on_unknown="raise",
    **kwargs,
)
```
{% endcode %}

Executes a config function and returns an `InstantiationOutput`.

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate_with_params

def config(hp: HP) -> str:
    return hp.select(["small", "large"], name="model", default="small")

run = instantiate_with_params(config, values={"model": "large"})

assert run.value == "large"
assert run.params == {"model": "large"}
```
{% endcode %}

`run.params` includes every reachable `hp.*` parameter selected during the run, including defaults.

## InstantiationOutput

{% code overflow="wrap" %}
```python
InstantiationOutput(value, params)
```
{% endcode %}

| Attribute | Meaning |
| --- | --- |
| `value` | The value returned by the config function. |
| `params` | A copied dictionary of selected parameter paths to selected values. |

## explore

{% code overflow="wrap" %}
```python
explore(
    func,
    *,
    values=None,
    on_unknown="raise",
    return_schema=False,
    **kwargs,
)
```
{% endcode %}

Traces a config function with a schema-recording `HP`. This executes the function to discover the active branch.

* With `return_schema=False`, prints a tree and returns `None`.
* With `return_schema=True`, returns a `ConfigSchema`.

{% code overflow="wrap" %}
```python
schema = explore(config, return_schema=True)
print(schema.defaults())
print(schema.to_dict())
```
{% endcode %}

## ConfigSchema

Returned by `explore(..., return_schema=True)`.

| Method | Meaning |
| --- | --- |
| `to_dict()` | Returns JSON-serializable schema metadata. |
| `defaults()` | Returns a flat dictionary of default values for the active branch. |
| `format_tree()` | Returns the same tree string printed by `explore()`. |

## interact

{% code overflow="wrap" %}
```python
interact(
    func,
    *,
    values=None,
    on_unknown="raise",
    auto_apply=True,
    **kwargs,
) -> InteractiveResult
```
{% endcode %}

Creates a notebook widget session for a config function and returns an `InteractiveResult` handle. Install the visualization extra before using it:

{% code overflow="wrap" %}
```bash
uv add "hypster[viz]"
```
{% endcode %}

`interact()` explores the config to render reachable controls, then instantiates the config from the current widget state. Widget changes can trigger repeated config execution to keep dependent controls current. The returned handle is not the configured object itself:

{% code overflow="wrap" %}
```python
result = interact(config)
current_value = result.value
current_params = result.params
```
{% endcode %}

| Parameter | Meaning |
| --- | --- |
| `func` | Config function whose first argument is `hp`. |
| `values` | Optional starting values, using the same flat or nested path forms as `instantiate()`. |
| `on_unknown` | Unknown-value policy used while exploring and applying widget state. |
| `auto_apply` | When `True`, valid widget edits update `.value` and `.params` immediately. When `False`, edits stay draft-only until Apply succeeds. |
| `**kwargs` | Extra execution arguments forwarded to the config function. |

## InteractiveResult

{% code overflow="wrap" %}
```python
result.value
result.params
result.snapshot
result.interact()
```
{% endcode %}

| Attribute or method | Meaning |
| --- | --- |
| `value` | The currently applied config return value. Raises `RuntimeError` while the applied state is invalid. |
| `params` | Replayable selected params for the currently applied state. Raises `RuntimeError` while the applied state is invalid. |
| `snapshot` | Widget-facing state with schema, draft values, applied values, selected params, mode, status, and error. |
| `interact()` | Renders another live widget view backed by the same session. |

Replay an interactive selection the same way you replay any Hypster run:

{% code overflow="wrap" %}
```python
result = interact(config)
replayed = instantiate(config, values=result.params)
```
{% endcode %}

## ParameterInfo

Each schema parameter contains:

| Field | Meaning |
| --- | --- |
| `name` | Local parameter name. |
| `path` | Dotted parameter path used in `values=`. |
| `display_label` | Human-friendly label derived from `name`, useful for generated UIs. |
| `kind` | `int`, `float`, `text`, `bool`, `select`, `multi_*`, or `group`. |
| `default_value` | Default value before overrides. |
| `selected_value` | Value selected in the traced run. |
| `options` | Select or multi-select options when available. |
| `minimum` / `maximum` | Numeric bounds when available. |
| `description` | Optional human-readable help text from `description=`. |
| `metadata` | Optional opaque UI/tooling hints from `metadata=`; omitted when absent or empty. |
| `children` | Nested parameters for groups. |

`ConfigSchema.to_dict()` is intended for rendering and inspection, not as a complete validation schema. It exposes the active branch, selected values, options, numeric bounds, descriptions, optional metadata, and nested groups. It does not currently expose every HP call option, such as `allow_none`, `options_only`, or `strict`.

UI builders should use schema metadata to render controls, then round-trip user input through `explore(..., values=..., return_schema=True)` and `instantiate(..., values=...)` for authoritative validation. For dict-backed selects, `options` contains replayable keys, not mapped runtime objects.

## HP Scalar Methods

All parameter names must be valid Python identifier-style strings: use letters, numbers, and underscores, and do not include dots, spaces, hyphens, or Python keywords. Hypster composes dotted paths from nesting.

{% code overflow="wrap" %}
```python
hp.int(default, *, name, min=None, max=None, strict=False, allow_none=False, hpo_spec=None, description=None, metadata=None)
hp.float(default, *, name, min=None, max=None, strict=False, allow_none=False, hpo_spec=None, description=None, metadata=None)
hp.text(default, *, name, allow_none=False, description=None, metadata=None)
hp.bool(default, *, name, allow_none=False, description=None, metadata=None)
```
{% endcode %}

| Method | Selected value |
| --- | --- |
| `hp.int` | `int`; accepts real integers and integral floats like `64.0` unless `strict=True`; rejects bool. |
| `hp.float` | `float`; accepts real floats and integer overrides unless `strict=True`; rejects bool. |
| `hp.text` | `str`. |
| `hp.bool` | `bool`. |

Use `allow_none=True` when `None` is a real scalar value.
Numeric coercion is consistent for top-level parameters and nested paths.
Use `metadata={...}` for opaque JSON-compatible hints that should appear on schema nodes without affecting selected values or runtime return objects.

## HP Select Methods

{% code overflow="wrap" %}
```python
hp.select(options, *, name, default=NO_DEFAULT, options_only=False, allow_none=False, hpo_spec=None, description=None, metadata=None)
hp.multi_select(options, *, name, default=None, options_only=False, allow_none=False, description=None, metadata=None)
```
{% endcode %}

`options` may be a list of logging-safe scalar choices or a dictionary from logging-safe scalar keys to any runtime values.

Use `allow_none=True` when `None` is one of the list-form choices:

{% code overflow="wrap" %}
```python
hp.select([None, "basic"], name="tokenizer", default=None, allow_none=True)
```
{% endcode %}

`hp.select([], name="choice", allow_none=True)` is valid and defaults to `None`. Without `allow_none=True`, an empty option list with no explicit default raises.

{% code overflow="wrap" %}
```python
model = hp.select(
    {
        "small": {"layers": 2},
        "large": {"layers": 4},
    },
    name="model",
    default="small",
)
```
{% endcode %}

The selected params record `"small"` or `"large"`, while the config function receives the mapped dictionary value.

By default, `options_only=False`, so custom scalar values outside the listed options are allowed. Use `options_only=True` for finite enums.

## HP Multi-Value Methods

{% code overflow="wrap" %}
```python
hp.multi_int(default, *, name, min=None, max=None, strict=False, allow_none=False, description=None, metadata=None)
hp.multi_float(default, *, name, min=None, max=None, strict=False, allow_none=False, description=None, metadata=None)
hp.multi_text(default, *, name, allow_none=False, description=None, metadata=None)
hp.multi_bool(default, *, name, allow_none=False, description=None, metadata=None)
```
{% endcode %}

These methods select lists whose elements are validated like the corresponding scalar type. `multi_int` accepts integral floats by default, and `multi_float` accepts integers by default. Both reject bool values. Nullable elements are not supported for `multi_int`, `multi_float`, `multi_text`, or `multi_bool`. Use `multi_select(..., allow_none=True)` for nullable categorical lists.

## HP.nest

{% code overflow="wrap" %}
```python
hp.nest(
    child,
    *,
    name,
    values=None,
    description=None,
    **kwargs,
)
```
{% endcode %}

Executes another config function under a named path.

{% code overflow="wrap" %}
```python
def child(hp: HP):
    return {"x": hp.int(1, name="x")}

def parent(hp: HP):
    return hp.nest(child, name="child")
```
{% endcode %}

Override nested values with dotted paths:

{% code overflow="wrap" %}
```python
instantiate(parent, values={"child.x": 2})
```
{% endcode %}

Nested dictionaries are normalized to the same dotted paths, so `values={"child": {"x": 2}}` is also valid. The scope name itself is not a parameter leaf: `values={"child": 2}` raises as unknown or unreachable.

Explicit child-local `values=` passed to `hp.nest(child, name="child", values=...)` are validated after the child config runs. Unknown or unreachable child keys raise instead of being ignored.

## HP.collect

{% code overflow="wrap" %}
```python
hp.collect(locals_dict, include=None, exclude=None)
```
{% endcode %}

Collects local variables into a dictionary while excluding `hp`, `self`, dunder names, and private names.

{% code overflow="wrap" %}
```python
def config(hp: HP):
    batch_size = hp.int(32, name="batch_size")
    learning_rate = hp.float(0.001, name="learning_rate")
    helper = "not returned"
    return hp.collect(locals(), exclude=["helper"])
```
{% endcode %}
