# Textual Types

Use `hp.text()` for one string and `hp.multi_text()` for a list of strings.

## Signatures

{% code overflow="wrap" %}
```python
hp.text(default, *, name, allow_none=False)
hp.multi_text(default, *, name, allow_none=False)
```
{% endcode %}

## Single Text Values

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate

def llm_config(hp: HP):
    return {
        "model_name": hp.text("gpt-5.4-mini", name="model_name"),
        "system_prompt": hp.text("Answer concisely.", name="system_prompt"),
    }

cfg = instantiate(
    llm_config,
    values={"system_prompt": "Answer with citations."},
)

assert cfg["system_prompt"] == "Answer with citations."
```
{% endcode %}

## Nullable Text

Use `allow_none=True` when `None` is an intentional text value:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return hp.text(None, name="system_prompt", allow_none=True)

assert instantiate(config) is None
```
{% endcode %}

## Multiple Text Values

{% code overflow="wrap" %}
```python
def generation_config(hp: HP):
    return {
        "stop_sequences": hp.multi_text(["###", "END"], name="stop_sequences"),
        "columns": hp.multi_text(["title", "body"], name="columns"),
    }

cfg = instantiate(
    generation_config,
    values={"stop_sequences": ["STOP", "DONE"]},
)

assert cfg["stop_sequences"] == ["STOP", "DONE"]
```
{% endcode %}

Nullable elements are not supported for `multi_text`; use `multi_select(..., allow_none=True)` for nullable categorical lists.
