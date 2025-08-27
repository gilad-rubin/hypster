# 🚀 Defining of A Config Function

A Hypster configuration function is a regular Python function. It receives `hp: HP` as its first parameter and returns whatever your code needs (a dict, an object, etc.).

{% stepper %}
{% step %}
### Imports

```python
from hypster import HP, instantiate
```

This makes sure you have the `HP` class for the first parameter and `instantiate(...)` to execute your config.
{% endstep %}

{% step %}
### Signature

```python
def my_config(hp: HP):
    ...
```

- The first parameter must be named `hp` and typed as `HP` for IDE autocomplete and validation.
- You can add more parameters (e.g., knobs) as long as `hp` is first: `def my_config(hp: HP, *, env: str = "dev")`.
{% endstep %}

{% step %}
### Body

Define parameters using `hp.*` and return the values you need. Use normal Python control flow for conditionals.

```python
from hypster import HP


def model_cfg(hp: HP):
    # Categorical choice (list form)
    model_name = hp.select(["gpt-5", "claude-sonnet-4-0", "gemini-2.5-flash"], name="model_name")

    # Numeric parameters
    temperature = hp.float(0.2, name="temperature", min=0.0, max=1.0)
    max_tokens = hp.int(256, name="max_tokens", min=0, max=4096)

    # Conditional logic
    if model_name == "gpt-5":
        # Extra knob only for gpt-5
        top_p = hp.float(1.0, name="top_p", min=0.1, max=1.0)
        return {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}

    return {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens}
```

Hypster comes with the following HP calls:

- `hp.select()` and `hp.multi_select()` for [categorical choices](../in-depth/hp-call-types/select-and-multi-select.md)
- `hp.int()` and `hp.multi_int()` for [integer values](../in-depth/hp-call-types/int-and-multi-int.md)
- `hp.float()` and `hp.multi_float()` for [numeric values](../in-depth/hp-call-types/int-and-multi-int.md)
- `hp.text()` and `hp.multi_text()` for [string values](../in-depth/hp-call-types/text-and-multi-text.md)
- `hp.bool()` and `hp.multi_bool()` for [boolean values](../in-depth/hp-call-types/bool-and-multi-bool.md)
- `hp.nest()` for [nested configurations](../in-depth/hp-call-types/nest.md)

{% hint style="warning" %}
Import libraries you need inside the function body when portability matters (so the config can be executed in isolation).
{% endhint %}

{% hint style="info" %}
Return exactly what your downstream code needs. You can also gather outputs from locals using `hp.collect(locals(), include=[...])`.
{% endhint %}
{% endstep %}

{% step %}
### Instantiation

Execute your configuration and override parameters using `values=`. See "Values & Overrides" for dotted vs nested overrides and precedence.

```python
from hypster import instantiate

cfg = instantiate(
    model_cfg,
    values={
        "model_name": "gpt-5",
        "temperature": 0.5,
        "max_tokens": 1024,
    },
)
# cfg -> {"model_name": "gpt-5", "temperature": 0.5, "max_tokens": 1024}
```

Control unknown or unreachable values via `on_unknown`: `"warn"` (default), `"raise"`, or `"ignore"`.
{% endstep %}
{% endstepper %}

## Saving & Loading Config Functions

Save configurations to reuse them across projects:

```python
# Save directly from config function
my_config.save("configs/my_config.py") # Creates directories if needed

# Save using hypster.save
from hypster import save
save(my_config, "configs/nested/my_config.py")
```

#### Loading Configurations

Load saved configurations in two ways:

```python
# Method 1: Direct loading
from hypster import load
my_config = load("configs/my_config.py")

# Method 2: Load for nesting
@config
def parent_config(hp: HP):
    nested_config = hp.nest("configs/my_config.py")
```
