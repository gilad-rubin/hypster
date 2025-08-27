### Hypster: simple, explicit configuration functions for Python

Hypster lets you define configuration spaces as plain Python functions and instantiate them with concrete values. No decorators, no AST rewriting, no registries. You write functions, name the parameters you want overrideable with `hp.*` calls, and run them.

---

## Motivation and design philosophy

- **Pure functions**: a config is `def f(hp: HP, ...) -> Any`. The only requirement is the first parameter is `hp: HP`.
- **Minimal magic**: no AST mutation, no auto-naming, no runtime filtering, no registry, no save/load. You control composition explicitly.
- **Explicit overrides**: only parameters created via `hp.*(name=...)` are overrideable with the `values` dict. “No name, no override.”
- **Pass-through returns**: return a dict, a single object, or a typed container (dataclass, NamedTuple, Pydantic). Hypster returns it as-is.
- **Callable-only nesting**: `hp.nest(child, name=...)` composes configs by calling child functions. `name` provides a stable dot-prefix for overrides.
- **Dotted keys (and nested dicts) for overrides**: use canonical dotted keys; nested dict overrides are supported and take precedence on conflicts.
- **Stateless core**: no built-in history or “explore mode.” Exploration, HPO, and UI state live outside, driven via a `describe(...)` schema.
- **Clear validation**: manual validators with friendly errors; intelligent suggestions for unknown names.

---

## Quick start

### 1) Define a config function
```python
from typing import Any
from hypster import HP

# Contract: first parameter is hp: HP

def model_conf(hp: HP) -> dict:
    # Explicit names are required for overrides
    model_name = hp.select(["gpt-4o", "haiku"], default="haiku", name="model.name")
    lr = hp.float(0.7, name="trainer.lr", min=0.0, max=1.0)
    return {"model_name": model_name, "lr": lr}
```

### 2) Instantiate with overrides
```python
from hypster import instantiate

cfg = instantiate(model_conf)
# {'model_name': 'haiku', 'lr': 0.7}

cfg = instantiate(model_conf, values={"model.name": "gpt-4o", "trainer.lr": 0.3})
# {'model_name': 'gpt-4o', 'lr': 0.3}
```

- Unknown keys trigger a helpful error with suggested matches.
- Non-float overrides for `hp.float` raise a clear type error; bounds are enforced.

---

## Return shapes and IDE DX

- **Single object** (best autocompletion):
```python
from sklearn.ensemble import RandomForestClassifier

def rf_conf(hp: HP) -> RandomForestClassifier:
    n = hp.int(200, name="rf.n_estimators", min=1)
    return RandomForestClassifier(n_estimators=n)

rf = instantiate(rf_conf, values={"rf.n_estimators": 300})
rf.fit(X, y); rf.predict(X)
```

- **Typed container** (multiple outputs, dot access):
```python
from dataclasses import dataclass

@dataclass
class TrainerOut:
    lr: float
    epochs: int

def trainer_conf(hp: HP) -> TrainerOut:
    lr = hp.float(0.1, name="lr", min=0.0, max=1.0)
    epochs = hp.int(10, name="epochs", min=1)
    return TrainerOut(lr=lr, epochs=epochs)

out = instantiate(trainer_conf, values={"lr": 0.05})
out.lr  # 0.05
```

---

## Explicit naming (name is required)

- All `hp.*` calls require `name`. Missing `name` raises an error when the function executes, even without overrides.
```python
def bad_conf(hp: HP):
    k = hp.int(10)  # no name
    return {"k": k}

instantiate(bad_conf)  # raises: missing 'name' for parameter; every hp.* must declare name
```

- Dotted keys and nested dicts are both supported; nested dict takes precedence:
```python
instantiate(trainer_conf, values={"trainer": {"lr": 0.2}, "trainer.lr": 0.5}) #@add a warning here that two identical values were defined and that the dict takes precendence

# effective lr = 0.2
```

---

## Composition with callable-only nesting

`hp.nest` composes configurations. It forwards to `instantiate(child, ...)` internally and uses `name` as the dot-prefix for overrides and snapshots.

```python
from typing import Protocol

class BaseRetriever(Protocol):
    def retrieve(self, q: str) -> list[str]: ...

def tfidf_conf(hp: HP, index) -> BaseRetriever:
    k = hp.int(5, name="k", min=1)
    return TFIDFRetriever(index=index, k=k)

def m2v_conf(hp: HP, *, dim_default=128) -> BaseRetriever:
    dim = hp.int(dim_default, name="dim", min=1)
    return M2VRetriever(dim=dim)

def pipeline(hp: HP, index) -> dict:
    # Choose a variant in plain Python
    choices = {"tfidf": tfidf_conf, "m2v": m2v_conf}
    kind = hp.select(list(choices), default="tfidf", name="retriever.kind")

    # Forward args/kwargs to the child; use 'retriever' as the stable prefix
    retr = hp.nest(
        choices[kind],
        name="retriever",
        args=(index,),              # for tfidf_conf
        kwargs={"dim_default": 256}  # for m2v_conf
    )
    return {"retriever": retr}

# Dotted override
out = instantiate(pipeline, values={"retriever.kind": "m2v", "retriever.dim": 512}, args=(my_index,))
```

### Nesting semantics
- `ConfigFunc` contract: first parameter must be `hp: HP`. Extra args/kwargs are allowed.
- Signature: `hp.nest(child, *, name: str, values=None, args=(), kwargs=None) -> Any`.
- Internally calls `instantiate(child, values=..., args=..., kwargs=...)`.
- Overrides: dotted and nested dict forms; nested dict entries override dotted keys on conflicts.

---

## Collecting locals ergonomically

```python
def conf(hp: HP) -> dict:
    a = hp.int(1, name="a")
    b = 2
    tmp = object()
    return hp.collect(locals(), include=["a", "b"])  # filters out hp/tmp/dunders/callables
```

- `hp.collect(locals(), include=[...])` and `exclude=[...]` help curate returned dicts.

---

## Float-only parameters: hp.float
#@ instead, put a table of each hp call type and show its signature
```python
def float_conf(hp: HP) -> float:
    temp = hp.float(0.5, name="temperature", min=0.0, max=1.0)
    return temp

instantiate(float_conf, values={"temperature": 0.7})   # ok
instantiate(float_conf, values={"temperature": 1.2})   # error: > max
instantiate(float_conf, values={"temperature": 1})     # error: must be float, got int
```

---

## Describe + Ask/Tell + Interactive UI

### Describe (schema extraction)
```python
from hypster import describe

schema = describe(pipeline, args=(my_index,))
# Example (shape will be a dict capturing parameters):
# {
#   'retriever.kind': {'type': 'select', 'options': ['tfidf', 'm2v'], 'default': 'tfidf'},
#   'retriever.k':    {'type': 'int', 'min': 1, 'default': 5},
#   'retriever.dim':  {'type': 'int', 'min': 1, 'default': 128}
# }
```

### Optuna (ask/tell) integration
```python
import optuna

schema = describe(rf_conf)
# Map to distributions (example):
space = {
    "rf.n_estimators": optuna.distributions.IntDistribution(50, 500)
}

study = optuna.create_study(direction="maximize")
for _ in range(20):
    trial = study.ask(space)
    values = trial.params  # keys should match hp names
    model = instantiate(rf_conf, values=values)
    score = evaluate(model)  # your objective
    study.tell(trial, score)
```

### Interactive UI
- Use `describe(func)` to build components (select/int/float/text/bool based on schema).
- On user change, compute a new `values` dict and call `instantiate(func, values, args, kwargs)`; render the result and updated components.
- Keep UI state externally; Hypster core remains stateless.

---

## Validation and error handling

- **Unknown names**: suggest close matches (e.g., did you mean `trainer.lr`?).
- **Type mismatches**: `hp.float` rejects ints; `hp.int` rejects floats; `hp.select(..., options_only=True)` enforces options.
- **Bounds**: numeric bounds raise friendly errors (`>= min`, `<= max`).
- **Missing names**: attempting to override an unnamed parameter raises a clear error.

---

## API reference (summary)

- `instantiate(func #@(ConfigFunc?), *, values=None, args=(), kwargs=None) -> Any`
- `hp.nest(child, *, name: str, values=None, args=(), kwargs=None) -> Any`
- `hp.collect(locals_dict, include=None, exclude=None) -> dict`
- Parameter calls (all require `name` to be overrideable):
  - `hp.select(options, *, name, default=None, options_only=False) -> Any`
  - `hp.multi_select(options, *, name, default=list(), options_only=False) -> list`
  - `hp.int(default: int, *, name, min=None, max=None) -> int`
  - `hp.float(default: float, *, name, min=None, max=None) -> float`
  - `hp.number(default: int|float, *, name, min=None, max=None) -> int|float`
  - `hp.text(default: str, *, name) -> str`
  - `hp.bool(default: bool, *, name) -> bool`
  - Multi variants: `hp.multi_int`, `hp.multi_number`, `hp.multi_text`, `hp.multi_bool`

---

## How Hypster compares

- **Optuna**
  - Similar: functions receive a handle (`hp` vs `trial`); ask/tell workflows map cleanly using `describe`.
  - Different: Hypster separates config definition from optimization; it does not manage studies, pruners, or trials.
  - Use Hypster to define config spaces and instantiate; use Optuna (or any HPO) to search values.

- **Hydra**
  - Similar: express configuration spaces.
  - Different: Hypster uses direct Python with explicit `hp` calls; it doesn’t orchestrate execution or rely on YAML composition.
  - Goal: a smaller, more straightforward API for parameterization and composition; you own execution.

---

## Best practices for production

- Add precise type hints on all public functions; annotate return types.
- Use Google/NumPy-style docstrings (`Args`, `Returns`, `Raises`).
- Run strict tooling: ruff (lint+isort), black, mypy, pydocstyle, codespell.
- Use explicit names for every overrideable parameter (`name="..."`).
- Prefer single-object returns when possible for best IDE autocompletion.
- Keep UI/HPO state external; treat Hypster core as stateless.

---

## Edge cases

- Reserved prefix collisions with hp.nest
  - Rule: when nesting with `name=prefix`, the parent must not define any HP under `prefix.*`, and child leaves become `prefix.*`. Any duplicate fully-qualified names cause an error.
  - Example (conflict):
```python
def parent(hp: HP):
    x = hp.text("gpt-4o", name="model.name")  # parent defines under 'model.'
    child = hp.nest(model_child, name="model")  # ERROR: prefix 'model' reserved for child

# child defines 'name' → would become 'model.name'
def model_child(hp: HP):
    model_name = hp.select(["gpt-4o", "haiku"], default="haiku", name="name")
    return {"name": model_name}
```
  - Example (fix by rename):
```python
def parent(hp: HP):
    kind = hp.text("gpt-4o", name="model.kind")  # parent uses a different leaf
    child = hp.nest(model_child, name="model")    # OK
```

## FAQ

- Can my config accept extra arguments besides `hp`?
  - Yes. Pass them via `instantiate(..., args=(), kwargs={})` or to `hp.nest(..., args=(), kwargs={})`.
- Can I mix dotted overrides and nested dict values?
  - Yes. If both specify the same leaf, the nested dict entry wins.
- Do I need a registry or special loader?
  - No. Resolve callables via normal imports or Python dicts.
