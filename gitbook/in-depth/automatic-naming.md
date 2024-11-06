# Automatic Variable Naming

Hypster provides sensible defaults for naming your variables to keep your code DRY (**D**on't **R**epeat **Y**ourself), while also supporting explicit naming when needed.

## Naming Methods

### Explicit Naming
Use the `name` parameter when you want full control over variable names:
```python
@config
def explicit_naming(hp: HP):
    var = hp.select(["o1", "o2"], name="my_explicit_variable")
    learning_rate = hp.number(0.001, name="lr")
```

### Automatic Naming
Hypster automatically infers names from three contexts:

1. **Variable Assignment**
   ```python
   # Name becomes "model_type"
   model_type = hp.select(["cnn", "rnn"])

   # Name becomes "learning_rate"
   learning_rate = hp.number(0.001)
   ```

2. **Dictionary Keys**
   ```python
   config = {
       "learning_rate": hp.number(0.001),    # "config.learning_rate"
       "model_params": {
           "layers": hp.int(3)               # "config.model_params.layers"
       }
   }
   ```

3. **Function/Class Keywords**
   ```python
   # Class initialization
   model = ModelConfig(
       model_type=hp.select(["cnn", "rnn"]),  # "model.model_type"
       learning_rate=hp.number(0.001)         # "model.learning_rate"
   )

   # Function calls
   result = process_data(
       batch_size=hp.int(32)                  # "result.batch_size"
   )
   ```

## Name Injection Process

Hypster uses Python's Abstract Syntax Tree (AST) to automatically inject parameter names:

```python
# Original code
@config
def my_config(hp: HP):
    model = hp.select(["cnn", "rnn"])
    config = {"lr": hp.number(0.001)}

# After name injection (internal representation)
def my_config(hp: HP):
    model = hp.select(["cnn", "rnn"], name="model")
    config = {"lr": hp.number(0.001, name="config.lr")}
```

## Important Notes

1. **Assignment Priority**
   ```python
   # Names are based on assignment targets, not function names
   result = some_func(param=hp.select([1, 2]))  # Creates "result.param, not some_func.param"
   ```

2. **Nested Naming**
   ```python
   model = Model(
       type=hp.select(["cnn", "rnn"]),         # "model.type"
       params={
           "lr": hp.number(0.1),               # "model.params.lr"
           "layers": hp.int(3)                 # "model.params.layers"
       }
   )
   ```

3. **Warning**: Avoid ambiguous assignments
   ```python
   # Bad: Unclear naming
   x = y = hp.select(["a", "b"])  # Which name should be used?

   # Good: Clear assignment
   model_type = hp.select(["a", "b"])
   ```

## Disabling Automatic Naming

> **Security Note**: While name injection is designed to be safe, users with strict security requirements can disable it using `inject_names=False`.

```python
@config(inject_names=False)
def my_config(hp: HP):
    # Must provide explicit names
    model = hp.select(["cnn", "rnn"], name="model_type")
    config = {
        "lr": hp.number(0.001, name="learning_rate")
    }
```

## Best Practices

1. **Use Descriptive Variables**
   ```python
   # Good: Clear variable names
   learning_rate = hp.number(0.001)
   model_type = hp.select(["cnn", "rnn"])

   # Bad: Unclear names
   x = hp.number(0.001)
   y = hp.select(["cnn", "rnn"])
   ```

2. **Consistent Naming**
   ```python
   # Good: Consistent structure
   model_config = {
       "type": hp.select(["cnn", "rnn"]),
       "params": {
           "learning_rate": hp.number(0.001)
       }
   }
   ```

3. **Explicit Names for Clarity**
   ```python
   # Use explicit names when auto-naming might be ambiguous
   result = complex_function(
       param=hp.select([1, 2], name="specific_param_name")
   )
   ```
