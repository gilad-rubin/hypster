# ⚡ Instantiating a Config Function

In this section, we'll use the following configuration function:

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

## Instantiation Rules

### Default Values

Parameters use their default values when not specified:

```python
config = llm_config()
# equivalent to {"model_name" : "haiku", "max_tokens" = 256, "cache.folder" : "./cache"), ...
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

### Numeric Bounds Validation

Numeric parameters undergo bounds validation, if specified:

```python
# These will raise validation errors:
config = llm_config(values={
    "config_dct.temperature": 1.5,  # Exceeds max=1
    "max_tokens": -10               # Below min=0
})
```

## Variable Selection Methods

To ensure we pass only the required variables, we have two filtering approaches:

1. **Include specific variables using `final_vars`**:

```python
config = my_config(final_vars=["model", "config_dict"], values={...})
run("Hello", **config)
```

Use `final_vars` when you need only a few specific variables

{% hint style="info" %}
When `final_vars` is empty, all variables are returned (except those in `exclude_vars`)
{% endhint %}

2. **Exclude unwanted variables using `exclude_vars`**:

```python
config = my_config(exclude_vars=["cache", "temp_data"], values={...})
run("Hello", **config)
```

Choose `exclude_vars` when you have many variables to keep and little to filter out.

## Available Parameter Types

Each parameter type has specific validation and behavior rules. See each section for more details:

### HP Call Types

* [**select & multi\_select**](../in-depth/hp-call-types/select-and-multi\_select.md) - For categorical choices
* [**int, number & multi\_int, multi\_number**](../in-depth/hp-call-types/int-and-multi\_int.md) - For numeric values
* [**bool & multi\_bool**](../in-depth/hp-call-types/bool-and-multi\_bool.md) - For boolean values
* [**text & multi\_text**](../in-depth/hp-call-types/text-and-multi\_text.md) - For string values
* [**propagate**](../in-depth/hp-call-types/propagate.md) - For nested configurations
