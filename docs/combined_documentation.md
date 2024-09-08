# doc


## Core Concepts


# The Hypster Object


# The Hypster Object

- When you decorate a function with `@config`, Hypster creates an object that wraps your configuration function.
- This object, an instance of the `Hypster` class, provides additional functionality and allows for flexible usage of your configuration.

## Creation

The Hypster object is created automatically when you use the `@config` decorator:

```python
from hypster import HP, config

@config
def hp_config(hp: HP):
    # Your configuration code here
    pass
```

In this example, `hp_config` is no longer a simple function - it's now a Hypster object.

## Features

The Hypster object provides several key features:

1. **Callable**: It can be called like a function, executing your configuration code with optional parameters:

   ```python
   result = hp_config(selections={'param': 'value'}, overrides={'other_param': 42})
   ```

1. **Saving and Loading**: It can be saved to and loaded from files:

   ```python
   import hypster
   hypster.save(hp_config, "my_config.py")
   loaded_config = hypster.load("my_config.py")
   ```

1. **Propagation**: It can be used in nested configurations:

   ```python
   @config
   def parent_config(hp: HP):
       nested_result = hp.propagate(hp_config)
   ```

## Accessing the Original Function

If you need to access the original, undecorated function, you can use the `func` attribute of the Hypster object:

```python
original_function = hp_config.func
```

This can be useful in certain advanced scenarios or for introspection.

## Best Practices

1. **Type Hinting**: Always use `hp: HP` in your configuration function parameters. This helps with IDE autocomplete and type checking.

2. **Naming**: Choose descriptive names for your configuration functions. They become part of your API.

3. **Modularity**: Create separate Hypster objects for different components of your system. This allows for better organization and reusability.

By understanding the Hypster object, you can take full advantage of Hypster's powerful configuration management capabilities.
# HP Object Methods


# HP Object Methods in Hypster

The `HP` object provides several methods for defining hyperparameters in your configurations. This guide covers the main methods: `select`, `text_input`, and `number_input`.

## 1. The `select` Method

`hp.select(options, default=None, name=None)`

The `select` method allows you to choose from a list or dictionary of options.

### 1.1 Dictionary Options

When using a dictionary, keys must be of type `str`, `int`, `bool`, or `float`. Values can be of any type.

```python
optimizer = hp.select({
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    1: 'custom_optimizer',
    True: lambda lr: torch.optim.AdamW(lr=lr)
}, default='adam')
```

### 1.2 List Options

For lists, values must be of type `str`, `int`, `bool`, or `float`. Lists are internally converted to dictionaries.

```python
activation = hp.select(['relu', 'tanh', 'sigmoid'], default='relu')
```

This is equivalent to:

```python
activation = hp.select({'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid'}, default='relu')
```

## 2. The `text_input` Method

`hp.text_input(default=None, name=None)`

Defines a text input with an optional default value.

```python
model_name = hp.text_input(default='my_awesome_model')
```

Future implementations will include runtime type checking for string values.

## 3. The `number_input` Method

`hp.number_input(default=None, name=None)`

Defines a numeric input with an optional default value.

```python
learning_rate = hp.number_input(default=0.001)
```

Future implementations will include runtime type checking for numeric values (integers or floats).

## 4. Comprehensive Example

Here's an example demonstrating all `HP` methods:

```python
from hypster import HP, config

@config
def hp_methods_example(hp: HP):
    import torch

    activation = hp.select(['relu', 'tanh', 'sigmoid'], default='relu')

    optimizer = hp.select({
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        1: 'custom_optimizer',
        True: lambda lr: torch.optim.AdamW(lr=lr)
    }, default='adam')

    model_name = hp.text_input(default='my_awesome_model')
    learning_rate = hp.number_input(default=0.001)

    # Using the selected values
    act_func = {
        'relu': torch.nn.ReLU(),
        'tanh': torch.nn.Tanh(),
        'sigmoid': torch.nn.Sigmoid()
    }[activation]

    if isinstance(optimizer, str):
        opt = optimizer  # It's a custom optimizer name
    elif callable(optimizer):
        opt = optimizer(learning_rate)  # It's a lambda function
    else:
        opt = optimizer(torch.nn.Linear(10, 10).parameters(), lr=learning_rate)

    print(f"Configured {model_name} with {activation} activation and {opt.__class__.__name__} optimizer")

# Usage
result = hp_methods_example(selections={'activation': 'tanh', 'optimizer': 'sgd'})
```

This example showcases all `HP` methods, including selections from lists and dictionaries with various key types, and using text and number inputs.
# Selections, Overrides & Final Vars


# Selections, Overrides, and Final Variables

## Final Variables

The `final_vars` parameter is a list that defines which variables within the functions should be returned. If left empty (or if an empty list is provided), all objects will be returned.

Example:
```python
import hypster
from hypster import HP


@hypster.config
def my_config(hp: HP):
    var1 = hp.select(["a", "b"], default="b")
    var2 = hp.select({"c": 5, "d": 7}, default="d")
    var3 = hp.text_input("hello")
    var4 = hp.number_input(10)


my_config(final_vars=["var2", "var3"])
# {'var2': 7, 'var3': 'hello'}
```

## Selections

- Selections only work with `hp.select` and need to be one of the *keys* for the options.
- For dictionaries, the keys are used, and for lists, the values themselves are used as keys.
- If there's a selection, it takes precedence over the default value.
- If a selection is not part of the options keys, it will raise an error.

### Example:
```python
import hypster
from hypster import HP


@hypster.config
def my_config(hp: HP):
    var1 = hp.select(["a", "b"], default="b")
    var2 = hp.select({"c": 5, "d": 7}, default="d")
    var3 = hp.text_input("hello")
    var4 = hp.number_input(10)


my_config(selections={"var1": "a", "var2": "d"})
# {'var1': 'a', 'var2': 7, 'var3': 'hello', 'var4': 10}
```

## Overrides

- Overrides work on both `hp.select`, `text_input` & `number_input` methods.
- For `hp.select`, if the override is a key in the options, it will output the value associated with that key.
- If it's not in the option keys or if it's selected for a parameter that uses `text_input` or `number_input`, it will output that value directly.

The precedence order is: overrides > selections > defaults```{warning}
Currently, Hypster doesn't support type-checking. This feature will be added in the future.
```
### Example:
```python
import hypster
from hypster import HP


@hypster.config
def my_config(hp: HP):
    var1 = hp.select(["a", "b"], default="b")
    var2 = hp.select({"c": 5, "d": 7}, default="d")
    var3 = hp.text_input("hello")
    var4 = hp.number_input(10)


my_config(overrides={"var1": "hey there", "var4": 5})
# {'var1': 'hey there', 'var2': 7, 'var3': 'hello', 'var4': 5}
```
```python
my_config(selections={"var1": "a"}, overrides={"var1": "hey there", "var4": 5})
# {'var1': 'hey there', 'var2': 7, 'var3': 'hello', 'var4': 5}
```

Note how the override takes precedence in the second example.

## Defaults

- In `hp.select`, you need to specify the defaults explicitly.
- For `text_input` and `number_input` methods, the value itself serves as the default.

### Common Use Case: Empty Call

Here's a common use case demonstrating how defaults work with an empty call:
```python
import hypster
from hypster import HP


@hypster.config
def my_config(hp: HP):
    var1 = hp.select(["a", "b"], default="b")
    var2 = hp.select({"c": 5, "d": 7}, default="d")
    var3 = hp.text_input("hello")
    var4 = hp.number_input(10)


my_config()
{"var1": "b", "var2": 7, "var3": "hello", "var4": 10}
```
- If no `final_vars` are defined (empty list), it will output all the variables in the function.
- If no selections and overrides are defined, it will output the default values.
- If there are no defaults specified, it will raise an error.
# The Pythonic API


# The Pythonic API

Hypster's API is designed to be intuitive and expressive, allowing you to use familiar Python constructs in your configuration functions.

## Conditional Statements for Dependent Variables

You can use conditional statements to define dependent variables:```python
from hypster import HP, config


@config
def conditional_config(hp: HP):
    model_type = hp.select(["CNN", "RNN", "Transformer"], default="CNN")

    if model_type == "CNN":
        num_layers = hp.select([3, 5, 7], default=5)
    elif model_type == "RNN":
        cell_type = hp.select(["LSTM", "GRU"], default="LSTM")
    else:  # Transformer
        num_heads = hp.select([4, 8, 16], default=8)
```
## Loops

You can use loops to define repetitive configurations:```python
from hypster import HP, config


@config
def loop_config(hp: HP):
    num_layers = hp.select([3, 5, 7], default=5)
    layer_sizes = []

    for i in range(num_layers):
        layer_sizes.append(hp.select([32, 64, 128], default=64, name=f"layer_{i}_size"))
```
## Changing Options Conditionally

You can dynamically change the options based on other selections:```python
from hypster import HP, config


@config
def dynamic_options_config(hp: HP):
    dataset_size = hp.select(["small", "medium", "large"], default="medium")

    if dataset_size == "small":
        model_options = ["simple_cnn", "small_rnn"]
    elif dataset_size == "medium":
        model_options = ["resnet", "lstm"]
    else:
        model_options = ["transformer", "large_cnn"]

    model = hp.select(model_options)
```
## Summary

By allowing pythonic configuration spaces you can:
- Use conditional statements to define dependent variables
- Utilize loops for repetitive configurations
- Dynamically change options based on other selections
- And much more! :)
# Automatic Parameter Naming


# Variable Naming
Hypster provides sensible defaults for naming your variables to keep your code **DRY** (**D**on't **R**epeat **Y**ourself)## Explicit Naming
You can explicitly name your variables using the `name` parameter:```python
from hypster import HP, config


@config
def explicit_naming(hp: HP):
    var = hp.select(["o1", "o2"], name="my_explicit_variable")
```
## Automatic Naming

Hypster uses a name injection process to automatically name your hyperparameters. It's important to understand how this works, especially if you have security concerns about code modification:

1. **Source Code Modification**: Hypster analyzes your configuration function's source code and injects `name` keyword arguments into the hyperparameter calls (`hp.select()`, `hp.number_input()`, etc.).

2. **AST Transformation**: This process uses Python's Abstract Syntax Tree (AST) to modify the source code without changing its functionality.

3. **Security Implications**: While this process is designed to be safe, users with strict security requirements should be aware that it involves modifying and re-executing the source code.

4. **Disabling Name Injection**: If you prefer to avoid automatic name injection, you can disable it by using `@config(inject_names=False)` or `load(..., inject_names=False)`. When disabled, you must provide explicit names for all hyperparameters.

Example of how name injection modifies your code:
```python
from hypster import HP, config


# Original code
@config
def my_config(hp: HP):
    model = hp.select(["cnn", "rnn"])


# Modified code (internal representation)
def my_config(hp: HP):
    model = hp.select(["cnn", "rnn"], name="model")
```
### Automatic Naming Rules
Hypster automatically infers variable names by utilizing the variable names, dictionary keys, and keyword arguments:

1. Variable Names
   - Example: `a = hp.select(['option1', 'option2'])`
   - Result: 'a' will be the name of this parameter

2. Dictionary Keys
   - Example: `config = {'learning_rate': hp.number_input(0.001)}`
   - Result: The dictionary key 'learning_rate' will be the name of this parameter

3. Class and Function Keyword Arguments
   - Example: `Model(hidden_size=hp.select([64, 128, 256]))`
   - Result: The keyword argument 'hidden_size' will be the name of this parameter

For nested structures, Hypster uses dot notation `(key.nested_key)` to represent the hierarchy. For example:
```python
model = Model(model_type=hp.select(['cnn', 'rnn']), # Automatically named 'model.model_type'
              model_kwargs={'lr' : hp.number_input(0.1)} # Automatically named 'model.model_kwargs.lr'
             )
``````{warning}
- Parameters are named based on the variable they're assigned to, **not the function or class name** they're associated with.
- For example, `result = some_func(a = hp.select(...))` will be accessible as `result.a`, not `some_func.a`.
```### Example Use-Cases:
1. Variable Assignment```python
@config
def automatic_naming(hp: HP):
    # This will be automatically named 'var'
    var = hp.select(["o1", "o2"])
    # This will be automatically named 'model_type'
    model_type = hp.select(["cnn", "rnn"])
```
2. Dictionary Keys:```python
@config
def dict_naming(hp: HP):
    config = {
        "model_type": hp.select(["cnn", "rnn"]),  # Automatically named 'config.model_type'
        "learning_rate": hp.number_input(0.001),  # Automatically named 'config.learning_rate'
    }
```
3. Class and function Keyword Arguments:```python
from hypster import HP, config


@config
def class_kwargs_naming(hp: HP):
    # Note new class definitions (or imports) need to be inside the config function
    class ModelConfig:
        def __init__(self, model_type, learning_rate):
            self.model_type = model_type
            self.learning_rate = learning_rate

    def func(param):
        return

    model = ModelConfig(
        model_type=hp.select(["cnn", "rnn"]),  # Automatically named 'model.model_type'
        learning_rate=hp.number_input(0.001),  # Automatically named 'model.learning_rate'
    )

    var = func(param=hp.select(["option1", "option2"]))  # Automatically named 'var.param'
```
```python
results = class_kwargs_naming(selections={"model.model_type": "cnn", "var.param": "option1"})
print(results["model"].model_type)
print(results["model"].learning_rate)
```
## Disabling Automatic Naming
In case you want to disable automatic naming and rely solely on explicit naming, you can do so by setting `inject_names=False`:```python
from hypster import HP, config


@config(inject_names=False)
def class_kwargs_naming(hp: HP):
    # Note new class definitions (or imports) need to be inside the config function
    class ModelConfig:
        def __init__(self, model_type, learning_rate):
            self.model_type = model_type
            self.learning_rate = learning_rate

    def func(param):
        return

    model = ModelConfig(
        model_type=hp.select(["cnn", "rnn"], name="model_type"),
        learning_rate=hp.number_input(0.001, name="learning_rate"),
    )
    var = func(param=hp.select(["option1", "option2"], name="param"))
```
When automatic naming is disabled, you must provide explicit names for all hyperparameters. Failing to do so will result in an error:
```python
@config(inject_names=False)
def no_injection_config(hp: HP):
    a = hp.select(["a", "b"])  # This will raise an error because no name is provided
```
```python
import logging

# Disable logging to prevent verbose output
logging.disable(logging.CRITICAL)

try:
    no_injection_config()
    print("If you see this, the error didn't occur as expected.")
except ValueError as e:
    assert "`name` argument is missing" in str(e)
    print("ValueError occurred as expected: `name` argument is missing")
```
Disabling automatic naming can be useful in scenarios where you want full control over parameter names, when the automatic naming might lead to ambiguities in your configuration, or when you have security concerns about source code modification.
# Portability


# Portability, Imports & DefinitionsHypster requires all of your configuration code to be encapsulated within the function itself to ensure portability. This means you must include any necessary imports and class definitions inside the function.```python
from hypster import HP, config


@config
def portable_config(hp: HP):
    import torch

    device = hp.select(["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    # Rest of your configuration...
```
This approach ensures that the configuration function can be easily shared or saved without dependency issues.## Examples of What Works and What Doesn't#### ❌ This will not work:```python
import os

from hypster import HP, config


@config
def non_portable_config(hp: HP):
    device = hp.select(["cpu", "cuda"], default="cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu")
```
```python
import logging

# Disable logging to prevent verbose output
logging.disable(logging.CRITICAL)

try:
    non_portable_config()
    print("If you see this, the error didn't occur as expected.")
except NameError as e:
    assert "name 'os' is not defined" in str(e)
    print("NameError occurred as expected: 'os' is not defined")
```
#### ✅ This will work:```python
from hypster import HP, config


@config
def portable_config(hp: HP):
    import torch

    device = hp.select(["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    # Rest of your configuration...
```
The same principle applies to class definitions and function definitions:```python
from hypster import HP, config


@config
def class_kwargs_naming(hp: HP):
    # Note: New class definitions (or imports) need to be inside the config function
    class ModelConfig:
        def __init__(self, model_type, learning_rate):
            self.model_type = model_type
            self.learning_rate = learning_rate

    def func(param):
        return param

    model = ModelConfig(
        model_type=hp.select(["cnn", "rnn"]),  # Automatically named 'model.model_type'
        learning_rate=hp.number_input(0.001),
    )  # Automatically named 'model.learning_rate'

    var = func(param=hp.select(["option1", "option2"]))  # Automatically named 'var.param'
```

## Advanced Usage


# Propagation


# Configuration Propagation

Hypster supports nested configurations by providing a `hp.propagate` API to select and override values at different levels.

## Using `hp.propagate()`

The `hp.propagate()` function allows you to include one configuration within another, propagating selections and overrides.

Example:We first define a hypster config function. It can be in the same Jupyter Notebook/Python Module or in a different one:```python
import hypster
from hypster import HP


@hypster.config
def my_config(hp: HP):
    llm_model = hp.select(
        {"haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-5-sonnet-20240620", "gpt-4o-mini": "gpt-4o-mini"},
        default="gpt-4o-mini",
    )

    llm_config = {"temperature": hp.number_input(0), "max_tokens": hp.number_input(64)}

    system_prompt = hp.text_input("You are a helpful assistant. Answer with one word only")
```
```python
hypster.save(my_config, "my_config.py")
```
- We can then `load` it from its path and have it be part of the parent configuration.
- We can select & override values within our nested configuration by using dot notation```python
from hypster import HP, config


@config
def my_config_parent(hp: HP):
    import hypster

    my_config = hypster.load("my_config.py")
    my_conf = hp.propagate(my_config)
    a = hp.select(["a", "b", "c"], default="a")


my_config_parent(
    selections={"my_conf.llm_model": "haiku"},
    overrides={"a": "d", "my_conf.system_prompt": "You are a helpful assistant. Answer with *two words* only"},
)
```

# Saving and Loading Configurations


# Advanced Hypster Features

## Saving and Loading Configurations

Hypster provides functionality to save and load configurations, making it easy to persist and reuse your setups.

### Saving Configurations

You can save a Hypster configuration using the `hypster.save()` function. This function cleans the decorator and imports, making the saved file standalone.

Example:
```python
import hypster
from hypster import HP


@hypster.config
def my_config(hp: HP):
    chunking_strategy = hp.select(["paragraph", "semantic", "fixed"], default="paragraph")

    llm_model = hp.select(
        {"haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-5-sonnet-20240620", "gpt-4o-mini": "gpt-4o-mini"},
        default="gpt-4o-mini",
    )

    llm_config = {"temperature": hp.number_input(0), "max_tokens": hp.number_input(64)}

    system_prompt = hp.text_input("You are a helpful assistant. Answer with one word only")
```
```python
hypster.save(my_config, "my_config.py")
```
This will:
1. save the configuration to a file named `my_config.py`
1. remove the @config decorator from the function definition
1. Adding necessary imports, namely: `from hypster import HP`

These allow portability for future usage of `hypster.load()`
### Loading Configurations

To load a saved configuration, use the `hypster.load()` function:
```python
import hypster

my_config = hypster.load("my_config.py")
```
```python
my_config()
```

This loads the configuration from `my_config.py` and allows you to use it in your current setup.

# Configuration Snapshots


# Configuration Snapshots for Reproducibility

- Hypster provides a way to capture a snapshot of the configuration for reproducibility purposes.
- This is especially useful for reproducibility purposes in Machine Learning & AI projects or any scenario where you need to recreate exact configurations.
- When using `hp.propagate`, the resulting snapshot also returns values from nested configurations.

## Using `return_config_snapshot=True`

When calling a Hypster configuration, you can set `return_config_snapshot=True` to get a dictionary of all instantiated values.

Example:```python
%%writefile llm_model.py

#This is a mock class for demonstration purposes
class LLMModel:
    def __init__(self, chunking, model, config):
        self.chunking = chunking
        self.model = model
        self.config = config

    def __eq__(self, other):
        return (self.chunking == other.chunking and
                self.model == other.model and
                self.config == other.config)
```
```python
from hypster import HP, config


@config
def my_config(hp: HP):
    from llm_model import LLMModel

    chunking_strategy = hp.select(["paragraph", "semantic", "fixed"], default="paragraph")

    llm_model = hp.select(
        {"haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-5-sonnet-20240620", "gpt-4o-mini": "gpt-4o-mini"},
        default="gpt-4o-mini",
    )

    llm_config = {"temperature": hp.number_input(0), "max_tokens": 64}

    model = LLMModel(chunking_strategy, llm_model, llm_config)


results, snapshot = my_config(selections={"llm_model": "haiku"}, return_config_snapshot=True)
```
```python
results
```
```python
snapshot
```
The difference between the `results` and `snapshot` are subtle, but important:
- `results` contains the instantiated results from the selections & overrides of the config function.
    - Notice the `'model'` output in the `results` dictionary
- `snapshot` contains the values that are necessary to get the exact output by using overrides=snapshot
    - Notice that `'model'` isn't found in the snapshot since it is a byproduct of the previous selected parameters (`chunking_strategy`, `llm_model`, etc...)
    - Notice that we have `llm_config.temperature` only, since this `max_tokens` isn't a configurable parameter.

### Example Usage:```python
reproduced_results = my_config(overrides=snapshot)
assert reproduced_results == results  # This should be True
```
This ensures that you can recreate the exact configuration state, which is crucial for reproducibility in machine learning experiments, ensuring consistent results across multiple runs or different environments.

## Nested Configurations

When using `hp.propagate`, the snapshot captures the entire hierarchy of configurations:```python
from hypster import HP, config, save


@config
def my_config(hp: HP):
    llm_model = hp.select(
        {"haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-5-sonnet-20240620", "gpt-4o-mini": "gpt-4o-mini"},
        default="gpt-4o-mini",
    )

    llm_config = {"temperature": hp.number_input(0), "max_tokens": hp.number_input(64)}
```
```python
save(my_config, "my_config.py")
```
- We can then `load` it from its path and have it be part of the parent configuration.
- We can select & override values within our nested configuration by using dot notation```python
@config
def my_config_parent(hp: HP):
    import hypster

    my_config = hypster.load("my_config.py")
    my_conf = hp.propagate(my_config)
    a = hp.select(["a", "b", "c"], default="a")
```
```python
final_vars = ["my_conf", "a"]

results, snapshot = my_config_parent(
    final_vars=final_vars, selections={"my_conf.llm_model": "haiku"}, overrides={"a": "d"}, return_config_snapshot=True
)
```
```python
results
```
```python
snapshot
```
```python
reproduced_results = my_config_parent(final_vars=final_vars, overrides=snapshot)
assert reproduced_results == results
```
