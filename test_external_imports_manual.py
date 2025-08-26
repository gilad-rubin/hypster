"""Test external imports functionality."""

import json
import os

from hypster import HP, config


@config
def config_with_json(hp: HP):
    """Config that uses json module."""
    data_format = hp.select(["json", "yaml"], default="json", name="data_format")
    if data_format == "json":
        test_data = json.dumps({"test": "value"})
        # Store as a parameter for the config system
        hp.text(default=test_data, name="test_data")


if __name__ == "__main__":
    import tempfile

    from hypster import save

    print("Testing external imports...")

    # Save the config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_path = f.name

    try:
        save(config_with_json, temp_path)
        print(f"Saved config to: {temp_path}")

        # Read the saved file content
        with open(temp_path, "r") as f:
            content = f.read()

        print("Saved file content:")
        print(content)
        print("Contains json import:", "import json" in content)

        # Test that the saved config can be loaded and works
        from hypster import load

        loaded_config = load(temp_path)
        result = loaded_config()
        print("Loaded config works:", "data_format" in result)

    finally:
        os.unlink(temp_path)
        print("Cleaned up temp file")
