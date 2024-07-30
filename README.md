# Hypster

Hypster is a flexible configuration system for Python projects, allowing for dynamic configuration management and visualization using Streamlit.

## Features

- Dynamic configuration tree building
- Nested configuration support
- Streamlit-based configuration visualization and editing
- Easy integration with existing Python projects

## Installation

You can install Hypster using pip:

```bash
pip install hypster
```

Or using Poetry:

```bash
poetry add hypster
```

## Quick Start

Here's a simple example of how to use Hypster:

```python
from hypster import Builder, Select, prep

class CacheConfig:
    def __init__(self, type: str, size: int):
        self.type = type
        self.size = size

cache_config = prep(CacheConfig(type=Select("cache_type"), size=1000))
cache_type__memory = "memory"
cache_type__disk = "disk"

builder = Builder().with_modules(globals())
driver = builder.build()
config = driver.instantiate(["cache_config"])

print(config)
```

For more examples and detailed usage, check out the [documentation](https://hypster.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.