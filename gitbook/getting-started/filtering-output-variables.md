# Filtering Output Variables

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
