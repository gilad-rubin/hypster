I'll help improve the documentation with your suggestions:

# Best Practices

## Shift Left Philosophy

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

### Performance Guidelines
- **Keep configuration execution under 1ms**
- **Never make API calls or database requests during configuration**
- **Avoid any operations that incur costs**
- **Defer resource initialization to execution phase**

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

## Type Safety

### Built-in Type Checking
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
