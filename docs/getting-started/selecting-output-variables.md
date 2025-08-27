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

# 🍡 Selecting Output Variables

Return exactly what your execution code needs. A configuration function can return a dict, object, or any structure you prefer.

## Return directly

```python
from hypster import HP


def model_cfg(hp: HP):
    # ... define parameters ...
    return {"model": model, "config": config}

cfg = instantiate(model_cfg, values={...})
run("Hello", **cfg)  # consumes only what it needs
```

## Using hp.collect

If you prefer, you can gather outputs from locals using `hp.collect`:

```python

def build(hp: HP):
    model = ...
    optimizer = ...
    learning_rate = ...
    return hp.collect(locals(), include=["model", "optimizer", "learning_rate"])
```

Keep your returns explicit and minimal to avoid signature mismatches and accidental coupling.
