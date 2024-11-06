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
        max_tokens = hp.int(256, max=2048)
    else:
        max_tokens = hp.int(126, max=1024)
        
    cache = Cache(folder=hp.text("./cache"))
    config_dct = {"temperature" : hp.number(0, min=0, max=1),
                  "max_tokens" : max_tokens}
              
    model = Model(model_name, cache)
```

## Basic Instantiation

Parameter values can be set using a `values` dictionary:

```python
config1 = llm_config(values={"model_name" : "sonnet"})
config2 = llm_config(values={"max_tokens" : 256, "config_dct.temperature" : 0.5})
```

Key considerations:&#x20;

* Parameter values must align with the **conditional logic** in the configuration
* **Dot notation** (e.g. `config_dct.temperature`) is used for automatic naming of nested parameter. More on that in the [Automatic Naming](../in-depth/automatic-naming.md) section.
* **Default values** are used when specific values aren’t provided

## `select` and `multi_select` parameters

The `select` and `multi_select` methods allow defining categorical parameters using either lists or dictionaries:

```python
# List form - values and keys are identical
model_type = hp.select(["haiku", "sonnet"])

# Dictionary form - separate values and keys
callbacks = hp.multi_select({"cost" : cost_callaback,
                             "runtime" : runtime_callback})
```

Use dictionary form when working with:

1. Long values (e.g. `{"claude" : "claude-3-5-sonnet-20241022"}`)
2. Specific numeric values (e.g. `{"small" : 1.524322}`)
3. Complex objects (e.g.`{"rf" : RandomForest(n_estimators=100, ...)}`)

### `options_only` parameter

The `options_only` parameter controls value validation:

```python
# Flexible validation (default)
model_type = hp.select(["haiku", "sonnet"], options_only=False)

# Strict validation - only predefined options allowed
model_type = hp.select(["haiku", "sonnet"], options_only=True)
```

* When `options_only=True`: Only pre-defined values are accepted
* When `options_only=False`: any value is accepted

### Parameter Instantiation Examples

#### **Using Predefined Values**

```python
my_config(values={"model_type": "haiku"})
```

#### **Using Custom Values**

```python
my_config(values={"model_type": "claude-3-opus-20240229"})
```

### Reproducibility and Value History

Hypster maintains a historical record of parameter values to ensure configuration reproducibility across different runs. This history can be accessed using `my_config.get_last_snapshot()`, allowing you to view and reuse previous configurations.

#### Value Serialization

When instantiating parameters with values outside the predefined options, Hypster handles serialization in two ways:

* Simple types (str, int, float, bool) are properly logged and reproducible
* Complex objects: Serialized as strings, not reproducible

### `number`, `int` and `multi_number` & `multi_int`

Hypster provides two types of numeric parameters with automatic validation:

```python
# Integer parameters
max_tokens = hp.int(256, min=0, max=2048)  # Only accepts integers

# Float parameters
temperature = hp.number(0.7, min=0, max=1)  # Accepts both floats and integers
```

#### Valid instantiations

```python
config = my_config(values={"max_tokens": 1024})  # Within bounds
config = my_config(values={"temperature": 0.5})  # Within bounds
```

#### Invalid instantiations - will raise errors

```python
config = my_config(values={"max_tokens": 3000})  # Exceeds max
config = my_config(values={"temperature": -0.1})  # Below min
```

## Text and Boolean Parameters

Simple parameter types for strings and booleans:

```python
# Text parameter - accepts any string
cache_dir = hp.text("./cache")
log_file = hp.text("app.log")

# Boolean parameter - True/False only
use_cache = hp.bool(True)
verbose = hp.bool(False)
```

### Example Usage

```python
config = my_config(values={
    "cache_dir": "/tmp/cache",     # Valid text value
    "use_cache": True,             # Valid boolean
    "verbose": False               # Valid boolean
})
```

## Nested Configurations

In more complex scenarios, you might want to nest configurations from different modules or files. Hypster supports this through the `hp.propagate` method, which allows you to include configurations from other files.

### Example: RAG Configuration

```python
from hypster import config, HP

@config
def rag_config(hp: HP):
    llm = hp.propagate("path_to_llm_config.py")  # Load LLM config from another file
    retriever = hp.propagate("path_to_retriever_config.py")  # Load retriever config
    
    pipeline = [retriever["embedder"], llm["model"]]
```

Instantiate nested parameters using either:

1. **Dot Notation**

```python
config = rag_config(values={
    "retriever.embedding": "jina",
    "llm.model_name": "haiku"
})
```

2. **Nested Dictionary**

```python
config = rag_config(values={
    "retriever": {"embedding": "jina"},
    "llm": {"model_name": "haiku"}
})
```
