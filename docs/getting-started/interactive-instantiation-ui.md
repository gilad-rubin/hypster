# 🎮 Interactive Instantiation (UI)

{% hint style="info" %}
**Note**: Interactive UI functionality has been temporarily removed in v0.3 as part of the major revamp. The Hypster team is working on bringing back an improved UI experience in a future release.
{% endhint %}

## Current Workflow

For now, you can instantiate configurations manually using the `instantiate()` function:

```python
from hypster import HP, instantiate

def model_cfg(hp: HP):
    model_name = hp.select(["gpt-4", "claude-3-sonnet"], name="model_name")
    temperature = hp.float(0.2, name="temperature", min=0.0, max=1.0)
    max_tokens = hp.int(256, name="max_tokens", min=0, max=4096)
    return {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens}

# Manual configuration
cfg = instantiate(
    model_cfg,
    values={
        "model_name": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 1024,
    },
)
```

## Stay Updated

Follow the [GitHub repository](https://github.com/gilad-rubin/hypster) for updates on when the interactive UI will be restored with enhanced capabilities.
