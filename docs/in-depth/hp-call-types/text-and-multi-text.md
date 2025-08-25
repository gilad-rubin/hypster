# Textual Types

Hypster provides string parameter configuration through `text` and `multi_text` methods. These methods handle string values without additional validation.

## Function Signatures

```python
def text(
    default: str,
    *,
    name: str
) -> str

def multi_text(
    default: List[str] = [],
    *,
    name: str
) -> List[str]
```

### Parameters

* `default`: Default text value (single) or list of text values (multi)
* `name`: Required name for the parameter (used for identification and access)

## Usage Examples

### Single Text Values

```python
# Single text parameter with default
model_name = hp.text("gpt-4", name="model_name")
prompt_prefix = hp.text("You are a helpful assistant.", name="prompt_prefix")

# Usage
config(values={"model_name": "claude-3"})
config(values={"prompt_prefix": "You are an expert programmer."})
```

### Multiple Text Values

```python
# Multiple text parameters with defaults
stop_sequences = hp.multi_text(["###", "END"])
system_prompts = hp.multi_text([
    "You are a helpful assistant.",
    "Answer concisely."
])

# Usage
config(values={"stop_sequences": ["STOP", "END", "DONE"]})
config(values={"system_prompts": ["Be precise.", "Show examples."]})
```

## Reproducibility

All textual parameters are fully serializable and reproducible:

```python
# Configuration will be exactly reproduced
snapshot = config.get_last_snapshot()
restored_config = config(values=snapshot)
```
