# Boolean Types

Use `hp.bool()` for one boolean and `hp.multi_bool()` for a list of booleans.

## Signatures

{% code overflow="wrap" %}
```python
hp.bool(default, *, name, allow_none=False)
hp.multi_bool(default, *, name, allow_none=False)
```
{% endcode %}

## Single Boolean

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate

def config(hp: HP):
    return {
        "stream": hp.bool(True, name="stream"),
        "use_cache": hp.bool(True, name="use_cache"),
    }

cfg = instantiate(config, values={"stream": False})
assert cfg == {"stream": False, "use_cache": True}
```
{% endcode %}

`hp.bool` requires actual booleans. Strings such as `"true"` are rejected.

## Nullable Boolean

Use `allow_none=True` for tri-state values:

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return hp.bool(None, name="stream", allow_none=True)

assert instantiate(config) is None
assert instantiate(config, values={"stream": True}) is True
```
{% endcode %}

## Multiple Booleans

{% code overflow="wrap" %}
```python
def config(hp: HP):
    return hp.multi_bool([True, False, True], name="feature_flags")

assert instantiate(config, values={"feature_flags": [False, False, True]}) == [False, False, True]
```
{% endcode %}

Nullable elements are not supported for `multi_bool`. Use `multi_select(..., allow_none=True)` for nullable categorical lists.
