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
