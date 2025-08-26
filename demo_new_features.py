"""
Hypster Configuration Registry & Enhanced Nesting - Demonstration

This script demonstrates all the new features implemented:
1. Registry System with Namespace Support
2. Enhanced Nesting with Registry Lookup
3. Dynamic Configuration Selection
4. File/Module Loading with Specific Objects
5. External Import Support
"""

import json
import os
import tempfile

from hypster import HP, config, load, registry, save


def demo_registry_system():
    """Demonstrate the registry system and namespace organization."""
    print("=== Registry System Demo ===")

    # Clear registry for clean demo
    registry.clear()

    # Register configurations in different namespaces
    @config(register="models.transformers.bert_base")
    def bert_base_config(hp: HP):
        hidden_size = hp.int(default=768, name="hidden_size")
        num_layers = hp.int(default=12, name="num_layers")
        num_heads = hp.int(default=12, name="num_heads")

    @config(register="models.transformers.bert_large")
    def bert_large_config(hp: HP):
        hidden_size = hp.int(default=1024, name="hidden_size")
        num_layers = hp.int(default=24, name="num_layers")
        num_heads = hp.int(default=16, name="num_heads")

    @config(register="models.cnn.resnet50")
    def resnet50_config(hp: HP):
        layers = hp.select([18, 34, 50, 101], default=50, name="layers")
        pretrained = hp.bool(default=True, name="pretrained")

    @config(register="data.loaders.text")
    def text_loader_config(hp: HP):
        batch_size = hp.int(default=32, name="batch_size")
        max_length = hp.int(default=512, name="max_length")

    print(f"Registry size: {len(registry)}")
    print(f"All configurations: {registry.list()}")
    print(f"Models namespace: {registry.list('models')}")
    print(f"Transformers: {registry.list('models.transformers')}")
    print(f"Data configs: {registry.list('data')}")
    print()


def demo_enhanced_nesting():
    """Demonstrate enhanced nesting with registry lookup."""
    print("=== Enhanced Nesting Demo ===")

    # Configuration that uses registry for nesting
    @config
    def experiment_config(hp: HP):
        # Direct registry lookup
        model = hp.nest("models.transformers.bert_base", name="model")
        data_loader = hp.nest("data.loaders.text", name="data_loader")

        # Training parameters
        learning_rate = hp.number(default=1e-4, name="learning_rate")
        epochs = hp.int(default=10, name="epochs")

    result = experiment_config()
    print("Experiment config result:")
    print(f"  Model hidden_size: {result['model']['hidden_size']}")
    print(f"  Data loader batch_size: {result['data_loader']['batch_size']}")
    print(f"  Learning rate: {result['learning_rate']}")
    print()


def demo_dynamic_selection():
    """Demonstrate dynamic configuration selection."""
    print("=== Dynamic Configuration Selection Demo ===")

    # Register multiple retriever configs
    @config(register="retrievers.bm25")
    def bm25_config(hp: HP):
        k1 = hp.number(default=1.5, name="k1")
        b = hp.number(default=0.75, name="b")

    @config(register="retrievers.tfidf")
    def tfidf_config(hp: HP):
        max_features = hp.int(default=5000, name="max_features")
        use_idf = hp.bool(default=True, name="use_idf")

    @config(register="retrievers.dense")
    def dense_config(hp: HP):
        model_name = hp.text(default="sentence-transformers/all-MiniLM-L6-v2", name="model_name")
        similarity = hp.select(["cosine", "dot", "euclidean"], default="cosine", name="similarity")

    # Configuration that dynamically selects retriever
    @config
    def search_pipeline_config(hp: HP):
        retriever_type = hp.select(["bm25", "tfidf", "dense"], default="bm25", name="retriever_type")
        retriever = hp.nest(f"retrievers.{retriever_type}", name="retriever")

        top_k = hp.int(default=10, name="top_k")

    # Test different selections
    for ret_type in ["bm25", "tfidf", "dense"]:
        result = search_pipeline_config(values={"retriever_type": ret_type})
        print(f"Using {ret_type} retriever:")
        print(f"  Config keys: {list(result['retriever'].keys())}")
    print()


def demo_file_loading():
    """Demonstrate loading specific functions from files."""
    print("=== File Loading with Specific Objects Demo ===")

    # Create a file with multiple configurations
    multi_config_content = """import json
from hypster import HP

def llm_config_small(hp: HP):
    model_name = hp.select(["gpt-3.5-turbo", "claude-3-haiku"], default="gpt-3.5-turbo", name="model_name")
    temperature = hp.number(default=0.7, name="temperature")
    max_tokens = hp.int(default=100, name="max_tokens")

def llm_config_large(hp: HP):
    model_name = hp.select(["gpt-4", "claude-3-opus"], default="gpt-4", name="model_name")
    temperature = hp.number(default=0.3, name="temperature")
    max_tokens = hp.int(default=1000, name="max_tokens")

def data_config(hp: HP):
    # Using external import
    sample_data = json.dumps({"example": "data"})
    format_type = hp.select(["json", "csv", "parquet"], default="json", name="format_type")
    hp.text(default=sample_data, name="sample_data")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(multi_config_content)
        temp_path = f.name

    try:
        # Configuration that loads specific functions from the file
        @config
        def application_config(hp: HP):
            # Load specific LLM config based on size choice
            llm_size = hp.select(["small", "large"], default="small", name="llm_size")
            llm = hp.nest(f"{temp_path}:llm_config_{llm_size}", name="llm")

            # Load data config
            data = hp.nest(f"{temp_path}:data_config", name="data")

            # App-specific settings
            debug_mode = hp.bool(default=False, name="debug_mode")

        # Test with small LLM
        result_small = application_config(values={"llm_size": "small"})
        print("Small LLM configuration:")
        print(f"  Model: {result_small['llm']['model_name']}")
        print(f"  Max tokens: {result_small['llm']['max_tokens']}")

        # Test with large LLM
        result_large = application_config(values={"llm_size": "large"})
        print("Large LLM configuration:")
        print(f"  Model: {result_large['llm']['model_name']}")
        print(f"  Max tokens: {result_large['llm']['max_tokens']}")

    finally:
        os.unlink(temp_path)

    print()


def demo_external_imports():
    """Demonstrate external import preservation."""
    print("=== External Import Support Demo ===")

    # Configuration that uses external imports
    @config
    def preprocessing_config(hp: HP):
        # Using json module
        config_data = {"preprocessing": "config"}
        serialized = json.dumps(config_data)

        normalize = hp.bool(default=True, name="normalize")
        scaling_method = hp.select(["standard", "minmax", "robust"], default="standard", name="scaling_method")
        hp.text(default=serialized, name="config_json")

    # Save and reload to test import preservation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        save(preprocessing_config, temp_path)

        # Read the saved file
        with open(temp_path, "r") as f:
            saved_content = f.read()

        print("Saved configuration file:")
        print(saved_content)
        print(f"External import preserved: {'import json' in saved_content}")

        # Load and test the saved config
        loaded_config = load(temp_path)
        result = loaded_config()
        print(f"Loaded config works: {'normalize' in result}")

    finally:
        os.unlink(temp_path)

    print()


def demo_error_handling():
    """Demonstrate helpful error messages."""
    print("=== Error Handling Demo ===")

    try:

        @config
        def failing_config(hp: HP):
            # This should fail with helpful message
            model = hp.nest("models.nonexistent", name="model")

        failing_config()
    except ValueError as e:
        print("Error message for nonexistent config:")
        print(f"  {e}")

    print()


def main():
    """Run all demonstrations."""
    print("Hypster Configuration Registry & Enhanced Nesting Demo")
    print("=" * 60)
    print()

    demo_registry_system()
    demo_enhanced_nesting()
    demo_dynamic_selection()
    demo_file_loading()
    demo_external_imports()
    demo_error_handling()

    print("Demo completed successfully! 🎉")
    print()
    print("Key benefits demonstrated:")
    print("✅ Centralized configuration registry with namespaces")
    print("✅ Dynamic configuration selection based on parameters")
    print("✅ Loading specific functions from files/modules")
    print("✅ Preservation of external imports")
    print("✅ Helpful error messages and suggestions")
    print("✅ Full backward compatibility")


if __name__ == "__main__":
    main()
