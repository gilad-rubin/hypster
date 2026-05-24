---
coverY: 0
layout:
  cover:
    visible: false
    size: full
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# Select Return Values

Hypster config functions return exactly what you decide to return. There is no framework-specific config object to build unless you want one.

## Return A Dict

```python
from hypster import HP, instantiate

def config(hp: HP):
    model_name = hp.select(["small", "large"], name="model_name", default="small")
    batch_size = hp.int(32, name="batch_size")
    return {"model_name": model_name, "batch_size": batch_size}

cfg = instantiate(config, values={"model_name": "large"})
assert cfg == {"model_name": "large", "batch_size": 32}
```

## Return A Dataclass

```python
from dataclasses import dataclass
from hypster import HP, instantiate

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float

def config(hp: HP) -> TrainingConfig:
    return TrainingConfig(
        batch_size=hp.int(32, name="batch_size", min=1),
        learning_rate=hp.float(0.001, name="learning_rate", min=0.0),
    )

cfg = instantiate(config, values={"batch_size": 64})
assert cfg.batch_size == 64
assert cfg == TrainingConfig(batch_size=64, learning_rate=0.001)
```

## Return Initialized Runtime Objects

One common Hypster style is to make the config a typed factory for the object your application will use.

```python
from sklearn.ensemble import RandomForestClassifier
from hypster import HP, instantiate

def classifier_config(hp: HP) -> RandomForestClassifier:
    n_estimators = hp.int(200, name="n_estimators", min=10, max=1000)
    max_depth = hp.int(None, name="max_depth", allow_none=True)
    min_samples_leaf = hp.int(2, name="min_samples_leaf", min=1, max=50)

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

model = instantiate(classifier_config, values={"n_estimators": 500})
```

The return type annotation helps readers, IDEs, and downstream code understand what instantiation produces.

Use dict-backed `select` when a parameter should log a simple key but return a more complex runtime value:

```python
architecture = hp.select(
    {
        "small": {"layers": 2, "units": [64, 32]},
        "large": {"layers": 4, "units": [256, 128]},
    },
    name="architecture",
    default="small",
)
```

## Use hp.collect

`hp.collect()` helps gather local variables while excluding `hp`, private names, and anything you explicitly exclude.

```python
from hypster import HP

def config(hp: HP):
    batch_size = hp.int(32, name="batch_size")
    learning_rate = hp.float(0.001, name="learning_rate")
    helper = "not returned"
    return hp.collect(locals(), exclude=["helper"])
```

Use `include=[...]` when you want to whitelist outputs:

```python
def config(hp: HP):
    batch_size = hp.int(32, name="batch_size")
    learning_rate = hp.float(0.001, name="learning_rate")
    debug_label = "local"
    return hp.collect(locals(), include=["batch_size", "learning_rate"])
```

## Keep Expensive Side Effects Outside Configs

Initializing in-memory objects is a good fit for Hypster configs. Keep expensive effects such as training, network calls, or database writes outside the config function when possible. That keeps `explore()`, HPO, UI generation, and replay fast and predictable.
