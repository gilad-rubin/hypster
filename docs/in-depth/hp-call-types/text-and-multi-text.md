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

## Usage Examples

### Single Text Values

```python
from hypster import HP, instantiate

def llm_config(hp: HP):
    # Single text parameters with defaults
    model_name = hp.text("gpt-4", name="model_name")
    prompt_prefix = hp.text("You are a helpful assistant.", name="prompt_prefix")

    return {
        "model_name": model_name,
        "prompt_prefix": prompt_prefix,
    }

# Usage with overrides
cfg = instantiate(llm_config, values={
    "model_name": "claude-3",
    "prompt_prefix": "You are an expert programmer."
})
# cfg -> {"model_name": "claude-3", "prompt_prefix": "You are an expert programmer."}
```

### Multiple Text Values

```python
from hypster import HP, instantiate

def generation_config(hp: HP):
    # Multiple text parameters with defaults
    stop_sequences = hp.multi_text(["###", "END"], name="stop_sequences")
    system_prompts = hp.multi_text([
        "You are a helpful assistant.",
        "Answer concisely."
    ], name="system_prompts")

    return {
        "stop_sequences": stop_sequences,
        "system_prompts": system_prompts
    }

# Usage with overrides
cfg = instantiate(generation_config, values={
    "stop_sequences": ["STOP", "END", "DONE"],
    "system_prompts": ["Be precise.", "Show examples."]
})
```

## Required Name Parameter

{% hint style="warning" %}
All `hp.*` calls that you want to be overrideable must include an explicit `name="..."` argument.
{% endhint %}

```python
# Correct usage - explicit names
model_name = hp.text("gpt-4", name="model_name")
stop_sequences = hp.multi_text(["###", "END"], name="stop_sequences")

# Incorrect usage - missing names (will raise error)
model_name = hp.text("gpt-4")        # Error: missing name
stop_sequences = hp.multi_text(["###", "END"])  # Error: missing name
```
