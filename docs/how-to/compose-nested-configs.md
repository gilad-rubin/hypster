# Compose Nested Configs

Use this guide when one workflow should be assembled from reusable smaller config functions.

## Start With Child Configs

{% code overflow="wrap" %}
```python
from hypster import HP, explore, instantiate
from my_app.embeddings import BpeTokenizer, EmbeddingPipeline, Encoder, Tokenizer, WordPieceTokenizer

tokenizer_options = {
    "wordpiece": WordPieceTokenizer,
    "bpe": BpeTokenizer,
}

def tokenizer_config(hp: HP) -> Tokenizer:
    tokenizer_cls = hp.select(tokenizer_options, name="kind", default="bpe", options_only=True)
    lowercase = hp.bool(True, name="lowercase")
    return tokenizer_cls(lowercase=lowercase)

def encoder_config(hp: HP) -> Encoder:
    size = hp.select(["small", "base"], name="size", default="small", options_only=True)
    hidden_size = 384 if size == "small" else 768
    return Encoder(size=size, hidden_size=hidden_size)
```
{% endcode %}

Use this named-options shape when a parent config chooses between swappable children. The dictionary provides the stable keys Hypster logs, while the values can be classes, callables, or full child config functions.

## Nest Them In A Parent

{% code overflow="wrap" %}
```python
def embedding_pipeline(hp: HP) -> EmbeddingPipeline:
    tokenizer = hp.nest(tokenizer_config, name="tokenizer")
    encoder = hp.nest(encoder_config, name="encoder")
    normalize = hp.bool(True, name="normalize")
    return EmbeddingPipeline(tokenizer=tokenizer, encoder=encoder, normalize=normalize)
```
{% endcode %}

## Override Nested Values

{% code overflow="wrap" %}
```python
cfg = instantiate(
    embedding_pipeline,
    values={
        "tokenizer.kind": "wordpiece",
        "encoder.size": "base",
        "normalize": False,
    },
)

assert cfg.encoder.hidden_size == 768
```
{% endcode %}

Nested dictionaries are equivalent:

{% code overflow="wrap" %}
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
{% endcode %}

The nested scope name is a prefix, not a leaf value. `values={"tokenizer": "wordpiece"}` raises as unknown because it does not target `tokenizer.kind`.

When you pass child-local values through `hp.nest(child, name="child", values=...)`, Hypster validates those explicit child values after the child runs. Typos and inactive child-branch keys raise instead of being ignored.

## Pass Execution Arguments To Children

{% code overflow="wrap" %}
```python
from my_app.training import BatchSampler, TrainingInputs

def sampler_config(hp: HP, default_batch_size: int) -> BatchSampler:
    batch_size = hp.int(default_batch_size, name="batch_size", min=1)
    shuffle = hp.bool(True, name="shuffle")
    return BatchSampler(batch_size=batch_size, shuffle=shuffle)

def training_config(hp: HP) -> TrainingInputs:
    train = hp.nest(sampler_config, name="train", default_batch_size=128)
    eval = hp.nest(sampler_config, name="eval", default_batch_size=256)
    return TrainingInputs(train=train, eval=eval)
```
{% endcode %}

## Inspect Branches Before Running

{% code overflow="wrap" %}
```python
explore(embedding_pipeline, values={"encoder.size": "base"})
```
{% endcode %}

If an override points at a parameter that is not reached on the active branch, Hypster raises by default. That keeps `values=` safe to log and replay.
