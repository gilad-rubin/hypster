# README.md

---
layout:
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

# 👋 Welcome

<div data-full-width="false">

<figure><picture><source srcset=".gitbook/assets/hypster_text_white_text.png" media="(prefers-color-scheme: dark)"><img src=".gitbook/assets/hypster_with_text (1).png" alt=""></picture><figcaption></figcaption></figure>

</div>

### **Hypster is a lightweight framework for defining configurations spaces to optimize ML & AI workflows.**

### Key Features

* :snake: **Pythonic API**: Intuitive & minimal syntax that feels natural to Python developers
* :nesting\_dolls: **Hierarchical Configurations**: Support for nested and swappable configurations
* :triangular\_ruler: **Type Safety**: Built-in type hints and validation using [`Pydantic`](https://github.com/pydantic/pydantic)
* :package: **Portability**: Easy serialization and loading of configurations
* :test\_tube: **Experiment Ready**: Built-in support for hyperparameter optimization
* :video\_game: **Interactive UI**: Jupyter widgets integration for interactive parameter selection

> Show your support by giving us a [star](https://github.com/gilad-rubin/hypster)! ⭐&#x20;

### How Does it work?

{% stepper %}
{% step %}
#### Install Hypster

Using uv (recommended):
```bash
uv add hypster
```

Or using pip:
```bash
pip install hypster
```
{% endstep %}

{% step %}
#### Define a configuration space

```python
from hypster import config, HP


@config
def llm_config(hp: HP):
    model_name = hp.select(["gpt-4o-mini", "gpt-4o"])
    temperature = hp.number(0.0, min=0.0, max=1.0)
```
{% endstep %}

{% step %}
#### Instantiate your configuration

```python
results = my_config(values={"model" : "gpt-4o", "temperature" : 1.0})
```
{% endstep %}

{% step %}
#### Define an execution function

```python
def generate(prompt: str, model_name: str, temperature: float) -> str:
    model = llm.get_model(model_name)
    response = model.prompt(prompt, temperature=temperature)
    return response
```
{% endstep %}

{% step %}
#### Execute!

```python
generate(prompt="What is Hypster?", **results)
```
{% endstep %}
{% endstepper %}

## Discover Hypster

<table data-view="cards"><thead><tr><th></th><th></th><th></th><th data-hidden data-card-cover data-type="files"></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Getting Started</strong></td><td>How to create &#x26; instantiate Hypster configs</td><td></td><td><a href=".gitbook/assets/Group 4 (5).png">Group 4 (5).png</a></td><td><a href="getting-started/installation.md">installation.md</a></td></tr><tr><td><strong>Tutorials</strong></td><td>Step-by-step guides for ML &#x26; Generative AI use-cases </td><td></td><td><a href=".gitbook/assets/Group 53.png">Group 53.png</a></td><td><a href="getting-started/usage-examples/">usage-examples</a></td></tr><tr><td><strong>Best Practices</strong></td><td>How to make the most out of Hypster</td><td></td><td><a href=".gitbook/assets/Group 26.png">Group 26.png</a></td><td><a href="in-depth/basic-best-practices.md">basic-best-practices.md</a></td></tr></tbody></table>

## Why Use Hypster?

In modern AI/ML development, we often need to handle **multiple configurations across different scenarios**. This is essential because:

1. We don't know in advance which **hyperparameters** will best optimize our performance metrics and satisfy our constraints.
2. We need to support multiple **"modes"** for different scenarios. For example:
   1. Local vs. Remote Environments, Development vs. Production Settings
   2. Different App Configurations for specific use-cases and populations

Hypster takes care of these challenges by providing a simple way to define configuration spaces and instantiate them into concrete workflows. This enables you to easily manage and optimize multiple configurations in your codebase.

## Additional Reading

* [Introducing Hypster](https://medium.com/@giladrubin/introducing-hypster-a-pythonic-framework-for-managing-configurations-to-build-highly-optimized-ai-5ee004dbd6a5) - A comprehensive introduction to Hypster's core concepts and design philosophy.
* [Implementing Modular RAG With Haystack & Hypster](https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f) - A practical guide to building modular, LEGO-like reconfigurable RAG systems.
* [5 Pillars for Hyper-Optimized AI Workflows](https://medium.com/@giladrubin/5-pillars-for-a-hyper-optimized-ai-workflow-21fcaefe48ca) - Key principles for designing optimized AI systems. The process behind this article gave rise to hypster's design.


# getting-started/installation.md

# 🖥️ Installation

Hypster is a lightweight package, mainly dependent on `Pydantic` for type-checking.

{% tabs %}
{% tab title="Basic Installation" %}
Using uv (recommended):
```bash
uv add hypster
```

Or using pip:
```bash
pip install hypster
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
{% endtab %}

{% tab title="Interactive Jupyter UI" %}
Hypster comes with an interactive **Jupyter Notebook UI** to make instantiation as easy as :pie:

Using uv:
```bash
uv add 'hypster[jupyter]'
```

Or using pip:
```bash
pip install hypster[jupyter]
```

Dependencies:

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
{% endtab %}

{% tab title="Development" %}
Interested in **contributing to Hypster?** Go ahead and install the full development suite using:

Using uv:
```bash
uv add 'hypster[dev]'
```

Or using pip:
```bash
pip install hypster[dev]
```

* [Pydantic](https://github.com/pydantic/pydantic)
* [ipywidgets](https://github.com/jupyter-widgets/ipywidgets)
* [ruff](https://github.com/astral-sh/ruff)
* [mypy](https://github.com/python/mypy)
* [pytest](https://github.com/pytest-dev/pytest)
{% endtab %}
{% endtabs %}

## Verification

After installation, you can verify your setup by running:

```python
import hypster
print(hypster.__version__)
```

## System Requirements

* Python 3.10 or higher
* Optional: Jupyter Notebook/Lab for interactive features

## Troubleshooting

If you encounter any installation issues:

1. Ensure your package manager is up to date:

Using uv:
```bash
uv self update
```

Or using pip:
```bash
pip install -U pip
```

2. Ensure `hypster` is up to date

Using uv:
```bash
uv add --upgrade hypster
```

Or using pip:
```bash
pip install -U hypster
```

3. For Jupyter-related issues, make sure Jupyter is properly installed:

Using uv:
```bash
# For JupyterLab
uv add jupyterlab

# Or 'classic' Jupyter Notebook
uv add notebook
```

Or using pip:
```bash
# For JupyterLab
pip install -U jupyterlab

# Or 'classic' Jupyter Notebook
pip install -U notebook
```

3. If you're still having problems, please [open an issue](https://github.com/gilad-rubin/hypster/issues) on our GitHub repository.


# getting-started/defining-a-configuration-space.md

# 🚀 Defining of A Config Function

A hypster `config` function is the heart of this framework. It requires of 3 main components:

{% stepper %}
{% step %}
### Imports

```python
from hypster import config, HP
```

This makes sure we have the `@config` decorater and `HP` class for type annotation.
{% endstep %}

{% step %}
### Signature

```python
@config
def my_config(hp: HP):
```

The function definition consists of the `@config` decorator and the signature. Including the `HP` (HyperParameter) type hint will enable IDE features like code suggestions and type checking.&#x20;
{% endstep %}

{% step %}
### Body

```python
@config
def my_config(hp: HP):
    from package import Class

    var = hp.select(["a", "b", "c"], default="a")
    num = hp.number(10)
    text = hp.text("Hey!")

    instance = Class(var=var, num=num, text=text)
```

Hypster comes with the following HP calls:

* `hp.select()` and `hp.multi_select()` for [categorical choices](../in-depth/hp-call-types/select-and-multi-select.md)
* `hp.int()` and `hp.multi_int()` for [integer values](../in-depth/hp-call-types/int-and-multi-int.md)
* `hp.number()`and `hp.multi_number()` for [numeric values](../in-depth/hp-call-types/int-and-multi-int.md)
* `hp.text()` and `hp.multi_text()` for [string values](../in-depth/hp-call-types/text-and-multi-text.md)
* `hp.bool()` and `hp.multi_bool()` for [boolean values](../in-depth/hp-call-types/bool-and-multi-bool.md)

Please note:

{% hint style="warning" %}
**All imports must be defined inside the body of the function.** This enables the portability of hypster's configuration object.
{% endhint %}

{% hint style="info" %}
**No return statement is allowed (nor needed)**. This enables [selecting the variables](selecting-output-variables.md) we want to retrieve upon instantiation using `final_vars` and `exclude_vars`
{% endhint %}
{% endstep %}

{% step %}
### Instantiation

Now that we've created a configuration space/function - we can instantiate it using:

```python
my_config(final_vars=["instance"], values={"var" : "b"})
```

Congratulations! :tada: You've created and instantiated your first Hypster config.
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


# getting-started/instantiating-a-configuration-space.md

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

* [**select & multi\_select**](../in-depth/hp-call-types/select-and-multi-select.md) - For categorical choices
* [**int, number & multi\_int, multi\_number**](../in-depth/hp-call-types/int-and-multi-int.md) - For numeric values
* [**bool & multi\_bool**](../in-depth/hp-call-types/bool-and-multi-bool.md) - For boolean values
* [**text & multi\_text**](../in-depth/hp-call-types/text-and-multi-text.md) - For string values
* [**nest**](../in-depth/hp-call-types/nest.md) - For nested configurations


# getting-started/selecting-output-variables.md

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


# getting-started/usage-examples/README.md

# 🪄 Usage Examples

Hypster's configuration system can be applied to various domains. Here are practical examples showing how to create type-safe, modular configurations for different use cases.

Check out the following tutorials for basic examples:

1. [Machine Learning](basic-example.md)
2. [LLM Generation](llms-and-generative-ai.md)


# getting-started/usage-examples/basic-example.md

# Machine Learning

Let's walk through a simple example to understand how Hypster works. We'll create a basic ML classifier configuration.

Prerequisites:

```bash
pip install scikit-learn
```

## Configurable Machine Learning Classifier

```python
from hypster import HP, config


@config
def classifier_config(hp: HP):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    # Define the model type choice
    model_type = hp.select(["random_forest", "hist_boost"],
                           default="hist_boost")

    # Create the classifier based on selection
    if model_type == "hist_boost":
        learning_rate = hp.number(0.01, min=0.001, max=0.1)
        max_depth = hp.int(10, min=3, max=20)

        classifier = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
    else:  # model_type == "random_forest"
        n_estimators = hp.int(100, min=10, max=500)
        max_depth = hp.int(5, min=3, max=10)
        bootstrap = hp.bool(default=True)

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap
        )
```

{% code overflow="wrap" %}
```python
# Instantiate with histogram gradient boosting
hist_config = classifier_config(values={
    "model_type": "hist_boost",
    "learning_rate": 0.05,
    "max_depth": 3
})

# Instantiate with random forest
rf_config = classifier_config(values={
    "model_type": "random_forest",
    "n_estimators": 200,
    "bootstrap": False
})
```
{% endcode %}

This example demonstrates several key features of Hypster:

1. **Configuration Definition**: Using the `@config` decorator to define a configuration space
2. **Parameter Types**: Using different HP call types (`select`, `number`, `int`, `bool`)
3. **Default Values**: Setting sensible defaults for all parameters
4. **Conditional Logic**: Different parameters based on model selection
5. **Multiple Instantiations**: Creating different configurations from the same space

## Understanding the Code

1. We define a configuration space using the `@config` decorator
2. The configuration function takes an `hp` parameter of type `HP`
3. We use various HP calls to define our parameter space:
   * `hp.select()` for categorical choices
   * `hp.number()` for floating-point & integer values
   * `hp.int()` for integer values only
   * `hp.bool()` for boolean values
4. The configuration returns a dictionary with our instantiated objects
5. We can create multiple instances with different configurations

## Training and Evaluating

{% code overflow="wrap" %}
```python
# Train a model using the configuration
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use the configured classifier
for model_type in ["random_forest", "hist_boost"]:
    results = classifier_config(values={"model_type": model_type})
    classifier = results["classifier"]

    # Train and evaluate
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(f"Model: {model_type}, accuracy: {score:.3f}")
```
{% endcode %}

This basic example shows how Hypster makes it easy to:

* Define configuration spaces with type-safe parameters
* Set reasonable defaults and parameter ranges
* Create multiple configurations from the same space
* Integrate with existing ML libraries seamlessly


# getting-started/usage-examples/llms-and-generative-ai.md

# LLM Generation

This tutorial demonstrates how to use Hypster with the `llm` package for managing different LLM configurations. We'll create a simple example showing how to switch between models and adjust generation parameters.

Prerequisites:

```bash
pip install llm
```

## Configurable LLM

```python
import os
import llm
from hypster import HP, config


@config
def llm_config(hp: HP):
    model_name = hp.select(["gpt-4o-mini", "gpt-4o"])
    temperature = hp.number(0.0, min=0.0, max=1.0)
    max_tokens = hp.int(256, max=2048)

def generate(prompt: str,
             model_name: str,
             temperature: float,
             max_tokens: int) -> str:
    model = llm.get_model(model_name)
    return model.prompt(prompt, temperature=temperature, max_tokens=max_tokens)


os.environ["OPENAI_API_KEY"] = "..."

# Create configurations for different use cases
final_vars = ["model_name", "temperature", "max_tokens"]
default_config = llm_config(final_vars=final_vars)
creative_config = llm_config(values={"model_name": "gpt-4o",
                                     "temperature": 1.0},
                             final_vars=final_vars)

# Example prompts
prompt1 = "Explain what machine learning is in 5 words."
prompt2 = "Write a haiku about AI in 17 syllables."
# Generate responses with different configurations
print("Default Configuration (Balanced):")
print(generate(prompt1, **default_config))
print("Creative Configuration (Higher Temperature):")
print(generate(prompt2, **creative_config))
```

This example demonstrates:

1. Simple model configuration with Hypster
2. Easy model switching using `llm`
3. Adjustable generation parameters (temperature, max\_tokens)
4. Different configurations for different use cases


# in-depth/automatic-naming.md

# 🤖 Automatic Naming

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

1.  **Variable Assignment**

    ```python
    # Name becomes "model_type"
    model_type = hp.select(["cnn", "rnn"])

    # Name becomes "learning_rate"
    learning_rate = hp.number(0.001)
    ```
2.  **Dictionary Keys**

    ```python
    config = {
        "learning_rate": hp.number(0.001), # "config.learning_rate"
        "model_params": {
            "layers": hp.int(3)            # "config.model_params.layers"
        }
    }
    ```
3.  **Function/Class Keywords**

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

1.  **Assignment Priority**

    ```python
    # Names are based on assignment targets, not function names
    result = some_func(param=hp.select([1, 2]))  # Creates "result.param, not some_func.param"
    ```
2.  **Nested Naming**

    ```python
    model = Model(
        type=hp.select(["cnn", "rnn"]),         # "model.type"
        params={
            "lr": hp.number(0.1),               # "model.params.lr"
            "layers": hp.int(3)                 # "model.params.layers"
        }
    )
    ```
3.  **Warning**: Avoid ambiguous assignments

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

1.  **Use Descriptive Variables**

    ```python
    # Good: Clear variable names
    learning_rate = hp.number(0.001)
    model_type = hp.select(["cnn", "rnn"])

    # Bad: Unclear names
    x = hp.number(0.001)
    y = hp.select(["cnn", "rnn"])
    ```
2.  **Consistent Naming**

    ```python
    # Good: Consistent structure
    model_config = {
        "type": hp.select(["cnn", "rnn"]),
        "params": {
            "learning_rate": hp.number(0.001)
        }
    }
    ```
3.  **Explicit Names for Clarity**

    ```python
    # Use explicit names when auto-naming might be ambiguous
    result = complex_function(
        param=hp.select([1, 2], name="specific_param_name")
    )
    ```


# in-depth/hp-call-types/README.md

# 🍱 HP Call Types

Hypster provides several parameter types to handle different configuration needs. Each type includes built-in validation and supports both single and multiple values.

## Available Types

### Selectable Types

**select & multi\_select**

* Categorical choices with optional value mapping
* Supports both list and dictionary forms

### Value Types

* **number & multi\_number**
  * Floating-point numbers with optional bounds
  * Accepts both integers and floats
* **int & multi\_int**
  * Integer values with optional bounds
  * Strict integer validation
* **text & multi\_text**
  * String values without validation
  * Useful for prompts, paths, and identifiers
* **bool & multi\_bool**
  * Boolean values
  * Simple true/false choices

### Advanced Types

* **nest**
  * Nested configuration management
  * Enables modular, reusable configs

### Common Features

All selectable & value-based types support:

* Automatic name inference
* Interactive UI widgets
* Type validation
* Default values

For detailed usage and examples, click through to the specific parameter type documentation.


# in-depth/hp-call-types/select-and-multi-select.md

# Selectable Types

The `select` and `multi_select` methods enable categorical parameter configuration. These methods support both single and multiple value selection with flexible validation options.

I'll help improve the documentation for the select and multi\_select parameters by adding the requested information. Here's the enhanced version:

## Select & Multi\_select Parameters

The `select` and `multi_select` methods enable categorical parameter configuration using either lists or dictionaries. These methods support both single and multiple value selection with flexible validation options.

### Function Signatures

#### select

```python
def select(
    options: Union[Dict[ValidKeyType, Any], List[ValidKeyType]],
    *,
    name: Optional[str] = None,
    default: Optional[ValidKeyType] = None,
    options_only: bool = False
) -> Any
```

#### multi\_select

```python
def multi_select(
    options: Union[Dict[ValidKeyType, Any], List[ValidKeyType]],
    *,
    name: Optional[str] = None,
    default: List[ValidKeyType] = None,
    options_only: bool = False
) -> List[Any]
```

### Parameters

* `options`: Either a list of valid values or a dictionary mapping keys to values
* `default`: Default value(s) if none provided (single value for select, list for multi\_select)
* `options_only`: When True, only allows values from the predefined options

## Pre-defined Parameter Forms

### List Form

Use when the parameter keys and values are identical:

```python
# Single selection
model_type = hp.select(["haiku", "sonnet"], default="haiku")

# Multiple selection
features = hp.multi_select(["price", "size", "color"], default=["price", "size"])
```

### Dictionary Form

Use when parameter keys need to map to different values:

```python
# Single selection with simple values
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="haiku")

# Multiple selection with complex values
callbacks = hp.multi_select({
    "cost": cost_callback,
    "runtime": runtime_callback
}, default=["cost"])
```

### Value Resolution

When using dictionary form, the configuration system maps input keys to their corresponding values:

```python
# Configuration definition
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="haiku")

# Usage
config = my_config(values={"model": "haiku"})
# Returns: "claude-3-haiku-20240307"

config = my_config(values={"model": "sonnet"})
# Returns: "claude-3-sonnet-20240229"
```

#### When to Use Dictionary Form?

Dictionary form is recommended when working with:

* Long string values: `{"haiku": "claude-3-haiku-20240307"}`
* Precise numeric values: `{"small": 1.524322}`
* Object references: `{"rf": RandomForest(n_estimators=100)}`

## Default Values

The `default` parameter must be a valid option from the predefined choices. For dictionary form, the default must be one of the keys (not values).

### List Form Defaults

When using list form, the default must be one of the items in the list:

```python
# Valid defaults
model = hp.select(["haiku", "sonnet"], default="haiku")  # OK: "haiku" is in list
features = hp.multi_select(["price", "size"], default=["price", "size"])  # OK: "price" and "size" are in list

# Invalid defaults - will raise errors
model = hp.select(["haiku", "sonnet"], default="opus")  # Error: "opus" not in list
features = hp.multi_select(["price", "size"], default=["color"])  # Error: "color" not in list
```

### Dictionary Form Defaults

When using dictionary form, the default must be one of the dictionary keys:

```python
# Valid defaults
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="haiku")  # OK: "haiku" is a key

callbacks = hp.multi_select({
    "cost": cost_callback,
    "runtime": runtime_callback
}, default=["cost"])  # OK: "cost" is a key

# Invalid defaults - will raise errors
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
}, default="claude-3-haiku-20240307")  # Error: using value instead of key

callbacks = hp.multi_select({
    "cost": cost_callback,
    "runtime": runtime_callback
}, default=["timing"])  # Error: "timing" is not a key
```

### Instantiating With Missing Default Values

If no default is provided, a value must be specified during configuration:

```python
# No default provided
model = hp.select(["haiku", "sonnet"])

# Must provide value during configuration
config = my_config(values={"model": "haiku"})  # OK
config = my_config()  # Error: no default and no value provided
```

## Value Validation

The `options_only` parameter determines how strictly values are validated:

```python
# Flexible validation - allows any value (default)
model = hp.select(["haiku", "sonnet"], options_only=False)

# Strict validation - only predefined options allowed
model = hp.select(["haiku", "sonnet"], options_only=True)
```

### Valid Instantiation Examples

```python
# Using predefined values
my_config(values={"model_type": "haiku"})

# Using custom values (when options_only=False)
my_config(values={"model_type": "claude-3-opus-20240229"})
```

### Invalid Instantiation Examples

```python
# Using a value not in the options list with options_only=True
my_config(values={"model_type": "claude-3-opus-20240229"})
```

I'll improve the Reproducibility and Value History section by adding clear examples:

## Reproducibility and Value History

Hypster maintains a historical record of parameter values to ensure configuration reproducibility across different runs. This history can be accessed using `my_config.get_last_snapshot()`, allowing you to view and reuse previous configurations.

### Value Serialization

When instantiating parameters with values outside the predefined options, Hypster handles serialization in two ways:

#### Simple types (str, int, float, bool)&#x20;

are properly logged and reproducible, regardless of if they were originally in the pre-defined options or not.

#### Complex objects

{% hint style="warning" %}
Complex object (classes, functions, anything outside of str, int, float, bool) will be serialized as strings using `str(value)` and will **not be reproducible for future runs**.
{% endhint %}

### Examples

```python
# Define configuration with options
model = hp.select({
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229"
})

# Example 1: Serializable value not in options
my_config(values={"model": "claude-3-opus"})
# Stored in history as "claude-3-opus"
# Fully reproducible since it's a simple string

# Example 2: Non-serializable value
class ModelClass:
    def __init__(self, model: str):
        self.model = model

    def __str__(self):
        return f"ModelClass(model={self.model})"

my_config(values={"model": ModelClass(model="haiku")})
# Stored in history as a string: "ModelClass(model=haiku)"
# Not reproducible since it's converted to string representation
```


# in-depth/hp-call-types/int-and-multi-int.md

# Numeric Types

Hypster provides flexible numeric parameter configuration through four methods: `number`, `multi_number`, `int`, and `multi_int`. These methods support automatic validation with optional bounds checking.

I'll create a comprehensive documentation for the numeric parameters:

## Numeric Parameters

Hypster provides flexible numeric parameter configuration through four methods: `number`, `multi_number`, `int`, and `multi_int`. These methods support automatic validation with optional bounds checking.

### Function Signatures

#### Single Value Methods

```python
NumericType = Union[StrictInt, StrictFloat] #pydantic types

def number(
    default: NumericType,
    *,
    name: Optional[str] = None,
    min: Optional[NumericType] = None,
    max: Optional[NumericType] = None
) -> NumericType

def int(
    default: int,
    *,
    name: Optional[str] = None,
    min: Optional[int] = None,
    max: Optional[int] = None
) -> int
```

#### Multiple Value Methods

```python
def multi_number(
    default: List[NumericType] = [],
    *,
    name: Optional[str] = None,
    min: Optional[NumericType] = None,
    max: Optional[NumericType] = None
) -> List[NumericType]

def multi_int(
    default: List[int] = [],
    *,
    name: Optional[str] = None,
    min: Optional[int] = None,
    max: Optional[int] = None
) -> List[int]
```

## Type Flexibility

### Number vs Integer

* `number`/`multi_number`: Accepts both integers and floating-point values
* `int`/`multi_int`: Accepts only integer values

The first argument in both methods is the default value:

```python
# Number parameters accept both types
temperature = hp.number(0.7, min=0, max=1) # min and max are optional
config(values={"temperature": 0.5})  # Float OK
config(values={"temperature": 1})    # Integer OK

# Integer parameters only accept integers
max_tokens = hp.int(256)
config(values={"max_tokens": 1024})  # Integer OK
config(values={"max_tokens": 512.5}) # Error: Float not allowed
```

## Bounds Validation

All numeric parameters support optional minimum and maximum bounds:

```python
# Single value with bounds
learning_rate = hp.number(0.001, min=0.0001)
batch_size = hp.int(32, max=256)

# Multiple values with bounds
layer_sizes = hp.multi_int(
    default=[128, 64, 32],
    min=16,
    max=512
)

learning_rates = hp.multi_number(
    default=[0.1, 0.01, 0.001],
    min=0.0001,
    max=1.0
)
```

### Valid Examples

```python
# Single values
config(values={"learning_rate": 0.01})    # Within bounds
config(values={"batch_size": 128})        # Within bounds

# Multiple values
config(values={"layer_sizes": [256, 128, 64]})           # All within bounds
config(values={"learning_rates": [0.1, 0.05, 0.001]})    # All within bounds
```

### Invalid Examples

```python
# Out of bounds errors
config(values={"learning_rate": 0.2})     # Exceeds max
config(values={"batch_size": 0})          # Below min

# Type errors
config(values={"batch_size": 64.5})       # Float not allowed for int
config(values={"layer_sizes": [32.5]})    # Float not allowed for multi_int

# Multiple values with invalid elements
config(values={"layer_sizes": [1024, 8, 64]})  # 1024 exceeds max, 8 below min
```

## Reproducibility

All numeric parameters are fully serializable and reproducible:

```python
# Configuration will be exactly reproduced
snapshot = config.get_last_snapshot()
restored_config = config(values=snapshot)
```


# in-depth/hp-call-types/bool-and-multi-bool.md

# Boolean Types

Hypster provides boolean parameter configuration through `bool` and `multi_bool` methods. These methods handle boolean values without additional validation.

## Function Signatures

```python
def bool(
    default: bool,
    *,
    name: Optional[str] = None
) -> bool

def multi_bool(
    default: List[bool] = [],
    *,
    name: Optional[str] = None
) -> List[bool]
```

## Usage Examples

### Single Boolean Values

```python
# Single boolean parameter with default
stream = hp.bool(True)
use_cache = hp.bool(False)

# Usage
config(values={"stream": False})
config(values={"use_cache": True})
```

### Multiple Boolean Values

```python
# Multiple boolean parameters with defaults
layer_trainable = hp.multi_bool([True, True, False])
feature_flags = hp.multi_bool([False, False])

# Usage
config(values={"layer_trainable": [False, True, True]})
config(values={"feature_flags": [True, False, True]})
```

### Invalid Values

```python
# These will raise errors
config(values={"stream": "true"})  # String instead of boolean
config(values={"layer_trainable": [1, 0]})  # Numbers instead of booleans
```

## Reproducibility

All boolean parameters are fully serializable and reproducible:

```python
# Configuration will be exactly reproduced
snapshot = config.get_last_snapshot()
restored_config = config(values=snapshot)
```


# in-depth/hp-call-types/text-and-multi-text.md

# Textual Types

Hypster provides string parameter configuration through `text` and `multi_text` methods. These methods handle string values without additional validation.

## Function Signatures

```python
def text(
    default: str,
    *,
    name: Optional[str] = None
) -> str

def multi_text(
    default: List[str] = [],
    *,
    name: Optional[str] = None
) -> List[str]
```

## Usage Examples

### Single Text Values

```python
# Single text parameter with default
model_name = hp.text("gpt-4")
prompt_prefix = hp.text("You are a helpful assistant.")

# Usage
config(values={"model_name": "claude-3"})
config(values={"prompt_prefix": "You are an expert programmer."})
```

### Multiple Text Values

```python
# Multiple text parameters with defaults
stop_sequences = hp.multi_text(["###", "END"])
system_prompts = hp.multi_text([
    "You are a helpful assistant.",
    "Answer concisely."
])

# Usage
config(values={"stop_sequences": ["STOP", "END", "DONE"]})
config(values={"system_prompts": ["Be precise.", "Show examples."]})
```

## Reproducibility

All textual parameters are fully serializable and reproducible:

```python
# Configuration will be exactly reproduced
snapshot = config.get_last_snapshot()
restored_config = config(values=snapshot)
```


# in-depth/hp-call-types/nest.md

# Nested Configurations

Hypster enables hierarchical configuration management through the `hp.nest()` method, allowing you to compose complex configurations from smaller, reusable components.

> For an in depth tutorial, please check out the article on Medium: [**Implementing Modular-RAG using Haystack and Hypster**](https://towardsdatascience.com/implementing-modular-rag-with-haystack-and-hypster-d2f0ecc88b8f)

## `nest` Function Signature

```python
def nest(
    config_func: Union[str, Path, "Hypster"],
    *,
    name: Optional[str] = None,
    final_vars: List[str] = [],
    exclude_vars: List[str] = [],
    values: Dict[str, Any] = {}
) -> Dict[str, Any]
```

#### Parameters

* `config_func`: Either a path to a saved configuration or a Hypster config object
* `name`: Optional name for the nested configuration (used in dot notation)
* `final_vars`: List of variables that cannot be modified by parent configs
* `exclude_vars`: List of variables to exclude from the configuration
* `values`: Dictionary of values to override in the nested configuration

## Steps for nesting

{% stepper %}
{% step %}
### Define a reusable config

```python
from hypster import config, HP

@config
def llm_config(hp: HP):
    model = hp.select({
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229"
    }, default="haiku")
    temperature = hp.number(0.7, min=0, max=1)
```
{% endstep %}

{% step %}
### Save it

```python
llm_config.save("configs/llm.py")
```
{% endstep %}

{% step %}
### Define a parent config and use `hp.nest`

```python
@config
def qa_config(hp: HP):
    # Load and nest LLM configuration
    llm = hp.nest("configs/llm.py")

    # Add QA-specific parameters
    max_context_length = hp.int(1000, min=100, max=2000)

    # Combine LLM and QA parameters
    qa_pipeline = QAPipeline(
        model=llm["model"],
        temperature=llm["temperature"],
        max_context_length=max_context_length
    )
```
{% endstep %}

{% step %}
### Instantiate using dot notation

```python
qa_config(values={
    "llm.model": "sonnet",
    "llm.temperature": 0.5,
    "max_context_length": 1500
})
```
{% endstep %}
{% endstepper %}

## Configuration Sources

`hp.nest()` accepts two types of sources:

### Path to Configuration File

```python
llm = hp.nest("configs/llm.py")
```

### Direct Configuration Object

```python
from hypster import load

# Load the configuration
llm_config = load("configs/llm.py")

# Use the loaded config
qa_config = hp.nest(llm_config)
```

## Value Assignment

Values for nested configurations can be set using either dot notation or nested dictionaries:

```python
# Using dot notation
qa_config(values={
    "llm.model": "sonnet",
    "llm.temperature": 0.5,
    "max_context_length": 1500
})

# Using nested dictionary
qa_config(values={
    "llm": {
        "model": "sonnet",
        "temperature": 0.5
    },
    "max_context_length": 1500
})
```

## Hierarchical Nesting

Configurations can be nested multiple times to create modular, reusable components:

```python
@config
def indexing_config(hp: HP):
    # Reuse LLM config for document processing
    llm = hp.nest("configs/llm.py")

    # Indexing-specific parameters
    embedding_dim = hp.int(512, min=128, max=1024)

    # Process documents with LLM
    enriched_docs = process_documents(
        llm=llm["model"],
        temperature=llm["temperature"],
        embedding_dim=embedding_dim
    )

@config
def rag_config(hp: HP):
    # Reuse indexing config (which includes LLM config)
    indexing = hp.nest("configs/indexing.py")

    # Add retrieval configuration
    retrieval = hp.nest("configs/retrieval.py")
```

## Passing Values to Nested Configs

Use the `values` parameter to pass dependent values to nested configuration values:

```python
retrieval = hp.nest(
    "configs/retrieval.py",
    values={
        "embedding_dim": indexing["embedding_dim"],
        "top_k": 5
    }
)
```

`final_vars` and `exclude_vars` are also supported.

## Best Practices

1. **Modular Design**
   * Create small, focused configurations for specific components
   * Combine configurations only when there are clear dependencies
   * Keep configurations reusable across different use cases
2.  **Clear Naming**

    ```python
    # Use descriptive names for nested configs
    llm = hp.nest("configs/llm.py", name="llm")
    indexer = hp.nest("configs/indexer.py", name="indexer")
    ```
3.  **Value Dependencies**

    ```python
    # Explicitly pass dependent values
    retriever = hp.nest(
        "configs/retriever.py",
        values={"embedding_dim": embedder["embedding_dim"]}
    )
    ```
4.  **File Organization**

    ```python
    # Keep related configs in a dedicated directory
    configs/
    ├── llm.py
    ├── indexing.py
    ├── retrieval.py
    └── rag.py
    ```


# in-depth/basic-best-practices.md

# 🧠 Best Practices

## "Shift Left" - Move Complexity to your Configs

Hypster encourages moving complexity into the configuration phase ("shifting left") rather than the execution phase:

```python
@config
def model_config(hp: HP):
    # Complex logic in configuration
    model_type = hp.select(["lstm", "transformer"], default="lstm")

    if model_type == "lstm":
        hidden_size = hp.int(128, min=64, max=512)
        num_layers = hp.int(2, min=1, max=4)
        bidirectional = hp.bool(True)

        model = LSTMModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
    else:  # transformer
        num_heads = hp.int(8, min=4, max=16)
        num_layers = hp.int(6, min=2, max=12)
        dropout = hp.number(0.1, min=0, max=0.5)

        model = TransformerModel(
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    # Common training parameters
    model.optimizer = hp.select(["adam", "sgd"], default="adam", name="optimizer")
    model.learning_rate = hp.number(0.001, min=1e-5, max=0.1, name="learning_rate")


# Simple execution code (outside config)
config = model_config(values={"model_type": "transformer"})
model = config["model"]
model.fit(X_train, y_train)  # All complexity handled in config
```

## Performance Guidelines

* Keep configuration execution under 1ms
* Never make API calls or database requests during configuration
* Avoid any operations that incur costs
* Defer resource initialization to execution phase

## Pythonic Configuration

### Use Native Python Features

```python
@config
def model_config(hp: HP):
    # Conditional logic
    model_type = hp.select(["cnn", "rnn", "transformer"])

    # Match-case statement
    match model_type:
        case "cnn":
            layers = hp.int(3, min=1, max=10)
            kernel = hp.select([3, 5, 7])
        case "rnn":
            cell = hp.select(["lstm", "gru"])
            hidden = hp.int(128)
        case _:
            heads = hp.int(8)

    # List comprehension
    layer_sizes = [hp.int(64, name=f"layer_{i}") for i in range(layers)]

    # For loop
    activations = {}
    for layer in range(layers):
        activations[f"layer_{layer}"] = hp.select(["relu", "tanh"], name=f"activation_{layer}")

    # One-liner conditional
    dropout = hp.number(0.5, name="dropout") if model_type == "transformer" else hp.number(0.3, name="dropout")
```

## Utilize Hypster's built-in Type Safety

### Use Built-in Type Checking

```python
@config
def typed_config(hp: HP):
    # Automatic type validation
    batch_size = hp.int(32)          # Only accepts integers
    learning_rate = hp.number(0.001)  # Accepts floats and ints
    model_name = hp.text("gpt-4")    # Accepts strings
```

### Value Validation

```python
@config
def validated_config(hp: HP):
    # Numeric bounds
    epochs = hp.int(100, min=1, max=1000)

    # Categorical options
    model = hp.select(["a", "b"], options_only=True)  # Only allows listed values
```


# advanced/nesting.md

# Nesting


# advanced/usage-tips-best-practices.md

# Usage tips / Best Practices



# advanced/saving-and-loading-configs.md

# Saving & Loading Configs

<figure><img src="../.gitbook/assets/image (24).png" alt=""><figcaption></figcaption></figure>


# advanced/performing-hyperparameter-optimization.md

# Performing Hyperparameter Optimization



# reproducibility/observing-past-runs.md

# Observing Past Runs



# reproducibility/experiment-tracking.md

# Experiment Tracking



# reproducibility/serialization.md

# Serialization



# reproducibility/cards.md

---
hidden: true
icon: acorn
---

# Cards

<figure><img src="../.gitbook/assets/hypster_with_text (1).png" alt=""><figcaption></figcaption></figure>

<table data-view="cards"><thead><tr><th></th><th></th><th></th><th data-hidden data-card-cover data-type="files"></th></tr></thead><tbody><tr><td><strong>Getting Started</strong></td><td>How to create &#x26; instantiate Hypster configs</td><td></td><td></td></tr><tr><td><strong>Best Practices</strong></td><td>How to make the most out of Hypster in your AI/ML project</td><td></td><td></td></tr><tr><td><strong>Tutorials</strong></td><td>Learn step-by-step tutorials for ML &#x26; Generative AI use-cases </td><td></td><td></td></tr><tr><td><strong>Introduction</strong></td><td>Motivation, Use cases &#x26; Unique Features</td><td></td><td></td></tr><tr><td><strong>Advanced Usage</strong></td><td>Working with nested configurations &#x26; serialization</td><td></td><td></td></tr><tr><td><strong>Hyperparameter Optimization</strong></td><td>How to utilize Hypster to perform &#x26; track HPO "Sweeps"</td><td></td><td></td></tr></tbody></table>


# reproducibility/deploying-to-production.md

# Deploying to production



# integrations/hamilton.md

# Hamilton



# integrations/haystack.md

# Haystack



# philosophy/origin-story.md

# Origin Story



# philosophy/articles.md

# Articles

{% embed url="https://medium.com/@giladrubin/introducing-hypster-a-pythonic-framework-for-managing-configurations-to-build-highly-optimized-ai-5ee004dbd6a5" %}

{% embed url="https://medium.com/@giladrubin/5-pillars-for-a-hyper-optimized-ai-workflow-21fcaefe48ca" %}


# philosophy/use-cases.md

---
icon: list-check
---

# Use Cases

<figure><img src="../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

<pre class="language-markdown"><code class="lang-markdown"><strong>* Use Cases
</strong>  * Enabling different "Modes"
    * Development
      * Local/Remote
        * Dev/Prod
        * Test/Actual
      * App
        * Different "Modes" for an app
        * A/B Testing
      * Agentic AI
        * Tool Use
        * Function Calling
</code></pre>


# philosophy/unique-features.md

---
icon: alicorn
---

# Unique Features

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>


# philosophy/hypster-vs.-alternatives.md

# Hypster vs. Alternatives

<figure><img src="../.gitbook/assets/image (20).png" alt=""><figcaption></figcaption></figure>
