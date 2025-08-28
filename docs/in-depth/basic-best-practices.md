# 🧠 Best Practices

## "Shift Left" - Move Complexity to your Configs

Hypster encourages moving complexity into the configuration phase ("shifting left") rather than the execution phase:

```python
from hypster import HP, instantiate

def model_config(hp: HP):
    # Complex logic in configuration
    model_type = hp.select(["lstm", "transformer"], name="model_type", default="lstm")

    if model_type == "lstm":
        hidden_size = hp.int(128, name="hidden_size", min=64, max=512)
        num_layers = hp.int(2, name="num_layers", min=1, max=4)
        bidirectional = hp.bool(True, name="bidirectional")

        model = LSTMModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
    else:  # transformer
        num_heads = hp.int(8, name="num_heads", min=4, max=16)
        num_layers = hp.int(6, name="num_layers", min=2, max=12)
        dropout = hp.float(0.1, name="dropout", min=0, max=0.5)

        model = TransformerModel(
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    # Common training parameters
    optimizer = hp.select(["adam", "sgd"], name="optimizer", default="adam")
    learning_rate = hp.float(0.001, name="learning_rate", min=1e-5, max=0.1)

    return {
        "model": model,
        "optimizer": optimizer,
        "learning_rate": learning_rate
    }


# Simple execution code (outside config)
config = instantiate(model_config, values={"model_type": "transformer"})
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
def model_config(hp: HP):
    # Conditional logic
    model_type = hp.select(["cnn", "rnn", "transformer"], name="model_type")

    # Match-case statement
    match model_type:
        case "cnn":
            layers = hp.int(3, name="layers", min=1, max=10)
            kernel = hp.select([3, 5, 7], name="kernel")
        case "rnn":
            cell = hp.select(["lstm", "gru"], name="cell")
            hidden = hp.int(128, name="hidden")
        case _:
            heads = hp.int(8, name="heads")

    # List comprehension
    layer_sizes = [hp.int(64, name=f"layer_{i}") for i in range(layers)]

    # For loop
    activations = {}
    for layer in range(layers):
        activations[f"layer_{layer}"] = hp.select(["relu", "tanh"], name=f"activation_{layer}")

    # One-liner conditional
    dropout = hp.float(0.5, name="dropout") if model_type == "transformer" else hp.float(0.3, name="dropout_alt")

    return {
        "model_type": model_type,
        "layer_sizes": layer_sizes,
        "activations": activations,
        "dropout": dropout
    }
```

## Utilize Hypster's built-in Type Safety

### Use Built-in Type Checking

```python
def typed_config(hp: HP):
    # Automatic type validation
    batch_size = hp.int(32, name="batch_size")          # Only accepts integers
    learning_rate = hp.float(0.001, name="learning_rate")  # Accepts floats
    model_name = hp.text("gpt-4", name="model_name")    # Accepts strings

    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_name": model_name
    }
```

### Value Validation

```python
def validated_config(hp: HP):
    # Numeric bounds
    epochs = hp.int(100, name="epochs", min=1, max=1000)

    # Categorical options
    model = hp.select(["a", "b"], name="model", options_only=True)  # Only allows listed values

    return {
        "epochs": epochs,
        "model": model
    }
```

## Required Naming Convention

{% hint style="warning" %}
All `hp.*` calls that you want to be overrideable must include an explicit `name="..."` argument. An `hp.*` call without a name will raise an error upon execution.
{% endhint %}

```python
def proper_naming(hp: HP):
    # Correct - explicit names
    model_type = hp.select(["a", "b"], name="model_type")
    learning_rate = hp.float(0.01, name="learning_rate")

    # Incorrect - missing names (will raise error)
    # model_type = hp.select(["a", "b"])  # Error!
    # learning_rate = hp.float(0.01)     # Error!

    return {"model_type": model_type, "learning_rate": learning_rate}
```

## Configuration Function Structure

```python
def well_structured_config(hp: HP):
    # 1. Define parameters with explicit names
    model_type = hp.select(["a", "b"], name="model_type")
    learning_rate = hp.float(0.01, name="learning_rate")

    # 2. Apply conditional logic
    if model_type == "a":
        param_a = hp.int(100, name="param_a")
    else:
        param_b = hp.float(0.5, name="param_b")

    # 3. Build objects/models
    model = SomeModel(type=model_type, lr=learning_rate)

    # 4. Return explicitly what downstream code needs
    return {
        "model": model,
        "learning_rate": learning_rate
    }
```
