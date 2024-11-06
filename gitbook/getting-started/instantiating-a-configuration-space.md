# 🔮 Instantiating a Config Function

## Configuration Function

In this section, we'll use the following toy configuration function:

```python
from hypster import config, HP

@config
def llm_config(hp: HP):
    model_name = hp.select({"sonnet" : "claude-3-5-sonnet-20241022"
                            "haiku" : "claude-3-5-haiku-20241022"},
                            default="haiku")

    if model_type == "haiku":
        max_tokens = hp.int(256, min=0, max=2048)
    else:
        max_tokens = hp.int(126, min=0, max=1024)

    cache = Cache(folder=hp.text("./cache"))
    config_dct = {"temperature" : hp.number(0, min=0, max=1),
                  "max_tokens" : max_tokens}

    model = Model(model_name, cache)
```

## Parameter Resolution Rules

### Default Values

Parameters use their default values when not specified:

```python
config = llm_config()
# returns model_name = "haiku", max_tokens = 256, cache = Cache(folder="./cache"), ...
```

### Conditional Logic

Values must respect the configuration's conditional logic:

```python
# Valid: haiku model allows up to 2048 tokens
config = llm_config(values={
    "model_name": "haiku",
    "max_tokens": 2000
})

# Invalid: sonnet model only allows up to 1024 tokens
config = llm_config(values={
    "model_name": "sonnet",
    "max_tokens": 2000  # Will raise error
})
```

### Value Validation

Numeric parameters undergo bounds validation, if specified:

```python
# These will raise validation errors:
config = llm_config(values={
    "config_dct.temperature": 1.5,  # Exceeds max=1
    "max_tokens": -10               # Below min=0
})
```

## Available Parameter Types

Each parameter type has specific validation and behavior rules. See each section for more details:

### Core Types

* [**select & multi\_select**](../in-depth/hp-call-types/select-and-multi\_select.md) - For categorical choices
* [**number\_input & multi\_number**](broken-reference) - For floating-point numbers
* [**int & multi\_int**](../in-depth/hp-call-types/int-and-multi\_int.md) - For integer values
* [**bool & multi\_bool**](../in-depth/hp-call-types/bool-and-multi\_bool.md) - For boolean values
* [**text & multi\_text**](../in-depth/hp-call-types/text-and-multi\_text.md) - For string values
* [**propagate**](../in-depth/hp-call-types/propagate.md) - For nested configurations
