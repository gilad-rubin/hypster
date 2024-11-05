# Machine Learning Tutorial

This tutorial demonstrates how to use Hypster for a complete machine learning pipeline, including data preprocessing, model selection, and evaluation.

## Complete Example: Iris Classification

```python
from hypster import config, HP
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the preprocessing configuration
@config
def preprocess_config(hp: HP):
    return {
        "scale_features": hp.bool(True, description="Whether to scale features"),
        "test_size": hp.number_input(0.2, min=0.1, max=0.4),
        "random_state": hp.int(42)
    }

# Define the model configuration
@config
def model_config(hp: HP):
    model_type = hp.select(
        ["svm", "random_forest"],
        default="svm",
        description="Model type to use"
    )

    if model_type == "svm":
        model = SVC(
            C=hp.number_input(1.0, min=0.1, max=10.0),
            kernel=hp.select(["linear", "rbf"], default="rbf"),
            random_state=hp.propagate("random_state")
        )
    else:
        model = RandomForestClassifier(
            n_estimators=hp.int(100, min=50, max=300),
            max_depth=hp.int(10, min=3, max=20),
            random_state=hp.propagate("random_state")
        )

    return {"model": model}

# Define the complete pipeline configuration
@config
def pipeline_config(hp: HP):
    preprocess = preprocess_config(hp)
    model = model_config(hp)
    return {**preprocess, **model}

# Function to run the pipeline
def run_pipeline(config_values=None):
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Get configuration
    config = pipeline_config(values=config_values)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )

    # Scale features if specified
    if config["scale_features"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train model
    model = config["model"]
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "model": model,
        "config": config
    }

# Example usage
if __name__ == "__main__":
    # Try different configurations
    configs_to_try = [
        {
            "model_type": "svm",
            "C": 5.0,
            "kernel": "rbf"
        },
        {
            "model_type": "random_forest",
            "n_estimators": 200,
            "max_depth": 15
        }
    ]

    for idx, config_values in enumerate(configs_to_try):
        results = run_pipeline(config_values)
        print(f"\nConfiguration {idx + 1}")
        print(f"Model type: {config_values['model_type']}")
        print(f"Accuracy: {results['accuracy']:.3f}")
```

This example demonstrates:

1. **Modular Configurations**: Separate configs for preprocessing and model
2. **Parameter Propagation**: Using `hp.propagate()` to share parameters
3. **Pipeline Integration**: Combining configs into a complete ML pipeline
4. **Practical Usage**: Real-world example with the Iris dataset

## Running Hyperparameter Optimization

```python
from hypster.optimization import optimize

# Define the objective function
def objective(config_values):
    results = run_pipeline(config_values)
    return results["accuracy"]

# Run optimization
best_config = optimize(
    pipeline_config,
    objective,
    n_trials=20,
    direction="maximize"
)

print("Best configuration found:")
print(best_config)
```

This tutorial shows how Hypster can help organize and optimize machine learning workflows while maintaining clean, modular code.
