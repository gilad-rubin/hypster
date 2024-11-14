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
