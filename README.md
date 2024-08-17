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

@dataclass
class DatabaseConfig:
    host: str
    port: int

@dataclass
class CacheConfig:
    type: str

# "lazy" defers instantiation
lazy([Database, Cache], update_globals=True)

# Define configuration options
db_host = Options({"production": "prod.example.com", 
                   "staging": "staging.example.com"}, default="staging")
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
## Inspiration
Hypster draws inspiration from [Meta's Hydra](https://github.com/facebookresearch/hydra) and [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) packages, combining their powerful configuration management with a minimalist approach. 

The API design is also influenced by the elegant simplicity of [Hamilton's API](https://github.com/DAGWorks-Inc/hamilton).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.