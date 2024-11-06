# 🎮 Interactive Configuration (UI)

As configuration spaces become more complex, manual instantiation can become challenging. Hypster provides an interactive UI for Jupyter notebooks that makes configuration management intuitive and error-free.

## Manual vs Interactive Configuration

### Manual Configuration
```python
# Manual configuration requires knowing all valid parameter combinations
results = modular_rag(
    values={
        "embedder_type": "fastembed",
        "embedder.model": "mini-lm",
        "embedder.parallel": 4
    }
)
```

### Interactive Configuration
```python
from hypster.ui import interactive_config

# Create an interactive UI
results = interactive_config(modular_rag)
```

## VS Code Setup
For VS Code users with dark theme:
```python
from hypster.ui import apply_vscode_theme
apply_vscode_theme()  # Enables dark mode compatibility
```

## Example: Conditional Configuration

Here's an example showing how the UI handles conditional parameters:

```python
@config
def conditional_config(hp: HP):
    model = hp.select(["cnn", "rnn", "transformer"], default="rnn")

    if model == "cnn":
        layers = hp.select([3, 5, 7], default=5)
        kernel_size = hp.select([3, 5], default=3)
    elif model == "rnn":
        cell_type = hp.select(["lstm", "gru"], default="lstm")
        num_layers = hp.int(5, min=1, max=100)
    else:  # transformer
        num_heads = hp.select([4, 8, 16], default=8)
        num_layers = hp.int(2, min=1, max=10)

# Create interactive UI
results = interactive_config(conditional_config)
```

The UI automatically:
- Shows/hides parameters based on conditions
- Validates parameter values
- Updates dependent parameters

## Example: Nested Configurations

The UI also handles nested configurations elegantly:

```python
@config
def modular_rag(hp: HP):
    embedder_type = hp.select(["fastembed", "jina"], default="fastembed")

    match embedder_type:
        case "fastembed":
            embedder = hp.propagate("configs/fast_embed.py")
        case "jina":
            embedder = hp.propagate("configs/jina_embed.py")

# Create interactive UI
results = interactive_config(modular_rag)
```

## Working with Results

The `results` object from `interactive_config` is dynamic and always reflects the current UI state:

## Key Features

- **Real-time Updates**: UI components update automatically based on conditions
- **Validation**: Prevents invalid parameter combinations
- **Nested Support**: Handles complex nested configurations
- **Type-specific Inputs**: Provides appropriate input widgets for each parameter type
- **VS Code Integration**: Seamless integration with VS Code's Jupyter extension
