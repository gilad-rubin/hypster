# text & multi\_text

Hypster provides string parameter configuration through `text` and `multi_text` methods. These methods handle string values without additional validation.

## Function Signatures

```python
def text(default: str, *, name: Optional[str] = None) -> str

def multi_text(default: List[str] = [], *, name: Optional[str] = None) -> List[str]
```

## Usage Examples

### Single Text Values
```python
# Single text parameter with default
model_name = hp.text("gpt-4")
prompt_prefix = hp.text("You are a helpful assistant.")

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
