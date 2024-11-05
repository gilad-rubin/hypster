# Basic Example

Let's walk through a simple example to understand how Hypster works. We'll create a basic text classifier configuration.

```python
from hypster import config, HP
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

@config
def classifier_config(hp: HP):
    # Define the model type choice
    model_type = hp.select(["random_forest", "hist_gradient_boosting"], default="random_forest")

    # Define common parameters
    random_state = hp.int(42)

    # Create the classifier based on selection
    if model_type == "hist_gradient_boosting":
        learning_rate = hp.number_input(0.01, min=0.001, max=0.1)
        max_depth = hp.int(10, min=3, max=20)

        classifier = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    else: # model_type == "random_forest"
        n_estimators = hp.int(100, min=10, max=500)
        max_depth = hp.int(5, min=3, max=10)

        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
```

```python
# Instantiate with default values
config_instance = classifier_config()
classifier = config_instance["classifier"]

# Instantiate with custom values
hist_config = classifier_config(values={
    "model_type": "hist_gradient_boosting",
    "learning_rate": 0.05,
    "max_depth": 3
})

rf_config = classifier_config(values={
    "model_type": "random_forest",
    "n_estimators": 200,
    "max_depth": 3
})
```

This example demonstrates several key features of Hypster:

1. **Configuration Definition**: Using the `@config` decorator to define a configuration space
2. **Parameter Types**: Using different HP call types (`select`, `number_input`, `int`)
3. **Default Values**: Setting sensible defaults for all parameters
4. **Conditional Logic**: Different parameters based on model selection
5. **Multiple Instantiations**: Creating different configurations from the same space

## Understanding the Code

1. We define a configuration space using the `@config` decorator
2. The configuration function takes an `hp` parameter of type `HP`
3. We use various HP calls to define our parameter space:
   - `hp.select()` for categorical choices
   - `hp.number_input()` for floating-point numbers
   - `hp.int()` for integer values
4. The configuration returns a dictionary with our instantiated objects
5. We can create multiple instances with different parameter values

## Training and Evaluating

```python
# Train a model using the configuration
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Use the configured classifier
config_instance = classifier_config()
classifier = config_instance["classifier"]

# Train and evaluate
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(f"Model accuracy: {score:.3f}")
```

This basic example shows how Hypster makes it easy to:
- Define configuration spaces with type-safe parameters
- Set reasonable defaults and parameter ranges
- Create multiple configurations from the same space
- Integrate with existing ML libraries seamlessly
