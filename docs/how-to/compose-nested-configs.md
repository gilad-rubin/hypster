# Compose Nested Configs

Use this guide when one workflow should be assembled from reusable smaller config functions.

## Start With Child Configs

```python
from hypster import HP, explore, instantiate

def tokenizer_config(hp: HP):
    return {
        "kind": hp.select(["wordpiece", "bpe"], name="kind", default="bpe", options_only=True),
        "lowercase": hp.bool(True, name="lowercase"),
    }

def encoder_config(hp: HP):
    size = hp.select(["small", "base"], name="size", default="small", options_only=True)
    hidden_size = 384 if size == "small" else 768
    return {"size": size, "hidden_size": hidden_size}
```

## Nest Them In A Parent

```python
def embedding_pipeline(hp: HP):
    return {
        "tokenizer": hp.nest(tokenizer_config, name="tokenizer"),
        "encoder": hp.nest(encoder_config, name="encoder"),
        "normalize": hp.bool(True, name="normalize"),
    }
```

## Override Nested Values

```python
cfg = instantiate(
    embedding_pipeline,
    values={
        "tokenizer.kind": "wordpiece",
        "encoder.size": "base",
        "normalize": False,
    },
)

assert cfg["encoder"]["hidden_size"] == 768
```

Nested dictionaries are equivalent:

```python
cfg = instantiate(
    embedding_pipeline,
    values={
        "tokenizer": {"kind": "wordpiece"},
        "encoder": {"size": "base"},
        "normalize": False,
    },
)
```

The nested scope name is a prefix, not a leaf value. `values={"tokenizer": "wordpiece"}` raises as unknown because it does not target `tokenizer.kind`.

When you pass child-local values through `hp.nest(child, name="child", values=...)`, Hypster validates those explicit child values after the child runs. Typos and inactive child-branch keys raise instead of being ignored.

## Pass Args And Kwargs To Children

```python
def sampler_config(hp: HP, default_batch_size: int):
    return {
        "batch_size": hp.int(default_batch_size, name="batch_size", min=1),
        "shuffle": hp.bool(True, name="shuffle"),
    }

def training_config(hp: HP):
    return {
        "train": hp.nest(sampler_config, name="train", args=(128,)),
        "eval": hp.nest(sampler_config, name="eval", kwargs={"default_batch_size": 256}),
    }
```

## Inspect Branches Before Running

```python
explore(embedding_pipeline, values={"encoder.size": "base"})
```

If an override points at a parameter that is not reached on the active branch, Hypster raises by default. That keeps `values=` safe to log and replay.
