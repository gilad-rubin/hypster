# Nested Configurations

Use `hp.nest()` to compose a configuration function from smaller configuration functions while keeping parameter paths stable and replayable.

```python
from hypster import HP, instantiate


def llm_config(hp: HP):
    provider = hp.select(["openai", "gemini"], name="provider", default="openai")
    temperature = hp.float(0.2, name="temperature", min=0.0, max=2.0)
    return {"provider": provider, "temperature": temperature}


def pipeline_config(hp: HP):
    llm = hp.nest(llm_config, name="llm")
    max_tokens = hp.int(4096, name="max_tokens", min=1)
    return {"llm": llm, "max_tokens": max_tokens}


run = instantiate(
    pipeline_config,
    values={"llm.provider": "gemini", "max_tokens": 8192},
)
```

## Signature

```python
def nest(
    child: Callable,
    *,
    name: str,
    values: dict[str, Any] | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> Any
```

`name` is required. It must be a valid Python identifier, and it cannot contain dots, spaces, hyphens, or Python keywords. Hypster owns dotted path construction from nested names.

## Values

Nested overrides can be provided with dotted keys:

```python
instantiate(pipeline_config, values={"llm.temperature": 0.7})
```

Or with nested dictionaries:

```python
instantiate(pipeline_config, values={"llm": {"temperature": 0.7}})
```

These are two spellings of the same parameter path. Do not provide both forms for the same leaf:

```python
instantiate(
    pipeline_config,
    values={
        "llm.temperature": 0.7,
        "llm": {"temperature": 0.7},
    },
)
# ValueError: Duplicate value for 'llm.temperature'
```

Nested dictionary keys are individual name segments, so they must be valid Python identifiers. Use dotted keys at the top level when you want to spell a full path.

## Passing Values To Children

Use the `values=` argument on `hp.nest()` when the parent config needs to derive child values:

```python
def retriever_config(hp: HP):
    return {"top_k": hp.int(5, name="top_k")}


def rag_config(hp: HP):
    documents = hp.int(100, name="documents")
    retriever = hp.nest(
        retriever_config,
        name="retriever",
        values={"top_k": max(1, documents // 20)},
    )
    return {"documents": documents, "retriever": retriever}
```

Explicit child `values=` are normalized with the same dotted/nested rules as top-level `values=`.

## Conditional Nesting

Nested configs can be selected conditionally. Unknown or unreachable values raise by default, so branch-specific overrides must match the active branch.

```python
def openai_config(hp: HP):
    return {"model": hp.select(["gpt-4o-mini", "gpt-4.1"], name="model")}


def gemini_config(hp: HP):
    return {"model": hp.select(["flash-lite", "pro"], name="model")}


def model_config(hp: HP):
    provider = hp.select(["openai", "gemini"], name="provider", default="openai")
    if provider == "openai":
        return hp.nest(openai_config, name="openai")
    return hp.nest(gemini_config, name="gemini")
```

To inspect a branch before passing values to it, run `explore(model_config, values={"provider": "gemini"})`.
