# Parameter Instantiation and Variable Selection

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

In this section, we’ll explore how to configure parameters and select variables in a configuration setup. The process involves two main aspects: defining parameter values and selecting which variables to include in the final output.

## Configuration Setup

In this section, we'll work with the following configuration function:

```python
from hypster import config, HP

@config
def my_config(hp: HP):
    model_type = hp.select(["type1" ,"type2"], default="type1")
    
    if model_type == "type1":
        param = hp.int(3, min=3, max=10)
    else:
        param = hp.int(10, min=10, max=100)
        
    cache = Cache(folder=hp.text("./cache"))
    config_dct = {"temperature" : hp.number(0, min=0, max=1),
                  "max_tokens" : hp.int(16, max=256),
                  "param" : param}
              
    model = Model(model_type, cache)
```

## Parameter Instantiation

Parameter values can be set using a `values` dictionary:

```python
config1 = my_config(values={"model_type" : "type2", "param" : 10})
config2 = my_config(values={"max_tokens" : 256})
```

Key considerations:&#x20;

* Parameter values must align with the conditional logic in the configuration
* Default values are used when specific values aren’t provided

## Variable Selection

When working with configuration functions, not all variables defined within them are needed for the final execution. Consider this execution function:

```python
def run(input: str, model: Model, config_dict: Dict[str, Any]) -> str:
    return model.run(input, **config_dict)
```

This function only requires `model` and `config_dict`, but our configuration function creates additional variables like `cache`, `model_type`, and `param`. Passing unnecessary variables could:

* Cause function signature mismatches
* Lead to memory inefficiency
* Create potential naming conflicts

### Variable Selection Methods

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

Choose `exclude_vars` when you have many variables to keep.

## Instantiation ?

There are two main ways to instantiate a configuration function:

This:

```python
results = modular_rag(
    values={
        "indexing.enrich_doc_w_llm": True,
        "indexing.llm.model": "gpt-4o-mini",
        "document_store_type": "qdrant",
        "retrieval.bm25_weight": 0.8,
        "embedder_type": "fastembed",
        "reranker.model": "tiny-bert-v2",
        "response.llm.model": "haiku",
        "indexing.splitter.split_length": 6,
        "reranker.top_k": 3,
    },
)
```

Or this:

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

Once you start getting used to creating configuration spaces, they can get somewhat complex. This makes it hard to track manually and instantiate them in a valid way.

Let's look at a simple example:&#x20;

In this toy example, if we want to instantiate `my_config` - we have to remember that `var2` can only be defined if `var == "a"`. This becomes much more difficult when we start using nested configurations and multiple conditions.

To address this challenge - hypster offers a built-in Jupyter Notebook based UI to interactively select valid configurations inside your IDE using `ipywidgets`.

## Interactive Instantiation



## Manual Instantiation
