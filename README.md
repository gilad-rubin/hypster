<p align="center">
  <img src="assets/hypster_with_text.png" alt="Hypster Logo" width="600"/>
</p>

Hypster is a lightweight configuration system for AI & Machine Learning projects. 
It offers minimal, intuitive syntax, supporting hierarchical and swappable configurations with lazy instantiation - making it both powerful and easy to integrate with existing projects.

## Installation

You can install Hypster using pip:

```bash
pip install hypster
```

## Quick Start

Here's a simple example of how to use Hypster:

```python
%%writefile configs.py
from hypster import lazy, Options

class Database:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def connect(self):
        print(f"Connecting to {self.host}:{self.port}")

class Cache:
    def __init__(self, type: str):
        self.type = type

    def initialize(self):
        print(f"Initializing {self.type} cache")

# Make classes lazy and update the global namespace
lazy([Database, Cache], update_globals=True)

# Define configuration options
db_host = Options({"production": "prod.example.com", "staging": "staging.example.com"}, default="staging")
cache_type = Options(["memory", "redis"], default="memory")
db_port = Options({"main": 5432, "alt": 5433}, default="main")

# Create lazy instances
db = Database(host=db_host, port=db_port)
cache = Cache(type=cache_type)
```

Now, in another cell or module, you can instantiate the configuration:

```python
from hypster import Composer
import configs

config = Composer().with_modules(configs).compose()

result = config.instantiate(
    final_vars=["db", "cache"],
    selections={"db.host": "production", "cache.type": "redis"},
    overrides={"db.port": 8000}
)

db.connect()  # Outputs: Connecting to prod.example.com:5434
cache.initialize()  # Outputs: Initializing redis cache
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.