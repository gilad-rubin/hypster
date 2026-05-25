# Nested Configurations

`hp.nest()` lets one config function call another config function under a named scope. This keeps large workflows readable and gives nested parameters stable dotted paths.

## Basic Nesting

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate
from my_app.training import AdamW, DataLoaders, BatchSampler, TrainingRun

def optimizer_config(hp: HP) -> AdamW:
    learning_rate = hp.float(0.001, name="learning_rate", min=1e-6, max=1.0)
    weight_decay = hp.float(0.0, name="weight_decay", min=0.0, max=1.0)
    return AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

def training_config(hp: HP) -> TrainingRun:
    epochs = hp.int(10, name="epochs", min=1)
    optimizer = hp.nest(optimizer_config, name="optimizer")
    return TrainingRun(epochs=epochs, optimizer=optimizer)

cfg = instantiate(training_config, values={"optimizer.learning_rate": 0.01})
assert cfg.optimizer.learning_rate == 0.01
```
{% endcode %}

## Signature

{% code overflow="wrap" %}
```python
hp.nest(
    child,
    *,
    name,
    values=None,
    args=(),
    kwargs=None,
)
```
{% endcode %}

| Argument | Meaning |
| --- | --- |
| `child` | Config function whose first parameter is `hp`. |
| `name` | Scope name. Must be a valid Python identifier. |
| `values` | Child-local values merged into the nested call. |
| `args` | Positional arguments passed to the child. |
| `kwargs` | Keyword arguments passed to the child. |

## Child-Local Values

`values=` inside `hp.nest()` is local to the child and is merged after parent-provided values for that child. Use it when the parent intentionally fixes or supplies child defaults.

{% code overflow="wrap" %}
```python
def parent(hp: HP):
    return hp.nest(
        optimizer_config,
        name="optimizer",
        values={"learning_rate": 0.005},
)

assert instantiate(parent).learning_rate == 0.005
assert instantiate(parent, values={"optimizer.learning_rate": 0.02}).learning_rate == 0.005
```
{% endcode %}

Think of this as a parent-fixed child value: the parent is choosing what the child sees. If you want callers to override the child value, leave `values=` off the `hp.nest()` call and put the default in the child parameter:

{% code overflow="wrap" %}
```python
def overridable_parent(hp: HP):
    return hp.nest(optimizer_config, name="optimizer")

cfg = instantiate(overridable_parent, values={"optimizer.learning_rate": 0.02})
assert cfg.learning_rate == 0.02
```
{% endcode %}

Explicit child values are validated after the child config runs. Unknown or unreachable child keys raise instead of being ignored:

{% code overflow="wrap" %}
```python
def parent_with_typo(hp: HP):
    return hp.nest(optimizer_config, name="optimizer", values={"learnig_rate": 0.005})

instantiate(parent_with_typo)
# ValueError: Unknown or unreachable parameters
```
{% endcode %}

Use child-local `values=` for parent-owned policy, test fixtures, or internal composition defaults that should win over caller-provided nested values.

## Args And Kwargs

{% code overflow="wrap" %}
```python
def sampler_config(hp: HP, default_batch_size: int) -> BatchSampler:
    batch_size = hp.int(default_batch_size, name="batch_size", min=1)
    shuffle = hp.bool(True, name="shuffle")
    return BatchSampler(batch_size=batch_size, shuffle=shuffle)

def data_config(hp: HP) -> DataLoaders:
    train = hp.nest(sampler_config, name="train", args=(128,))
    eval = hp.nest(sampler_config, name="eval", kwargs={"default_batch_size": 256})
    return DataLoaders(train=train, eval=eval)
```
{% endcode %}

## Conditional Nesting

You can choose which child config to run:

{% code overflow="wrap" %}
```python
from my_app.backends import AppRuntime, LocalBackend, RemoteBackend

def local_config(hp: HP) -> LocalBackend:
    threads = hp.int(4, name="threads", min=1)
    return LocalBackend(threads=threads)

def remote_config(hp: HP) -> RemoteBackend:
    endpoint = hp.text("https://api.example.com", name="endpoint")
    return RemoteBackend(endpoint=endpoint)

backend_options = {"local": local_config, "remote": remote_config}

def app_config(hp: HP) -> AppRuntime:
    selected_config = hp.select(backend_options, name="backend", default="local", options_only=True)
    backend = hp.nest(selected_config, name="settings")
    return AppRuntime(backend=backend)
```
{% endcode %}

This value is valid:

{% code overflow="wrap" %}
```python
instantiate(app_config, values={"backend": "remote", "settings.endpoint": "https://staging.example.com"})
```
{% endcode %}

This value raises by default because `settings.threads` is unreachable on the `remote` branch:

{% code overflow="wrap" %}
```python
instantiate(app_config, values={"backend": "remote", "settings.threads": 8})
```
{% endcode %}

## Name Collisions

Nested scopes share one parameter namespace. Hypster raises if the same full path is defined twice or if a parent parameter reserves a prefix needed by a nested child. Use unique scope names such as `encoder`, `decoder`, `train_loader`, and `eval_loader`.
