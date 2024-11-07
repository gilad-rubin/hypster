---
coverY: 0
layout:
  cover:
    visible: false
    size: full
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 🍡 Selecting Output Variables

When working with configuration functions, not all variables defined within them are needed for the final execution engine.&#x20;

Consider this configuration function:

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

Along with this execution function:

```python
def run(input: str, model: Model, config_dict: Dict[str, Any]) -> str:
    return model.run(input, **config_dict)
```

This function only requires `model` and `config_dict`, but our configuration function creates additional variables like `cache`, `model_type`, and `param`. Passing unnecessary variables could:

* Cause function signature mismatches
* Lead to memory inefficiency
* Create potential naming conflicts

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
