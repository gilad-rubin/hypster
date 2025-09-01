"""Test current behavior of dict options in select."""

from hypster import HP, instantiate


def test_dict_primitive():
    """Test dictionary with primitive values."""

    def config(hp: HP):
        options = {"fast": "gpt-4o-mini", "smart": "gpt-4"}
        model = hp.select(options, default="fast", name="model")
        return {"selected": model}

    # Without return_params - just check basic behavior
    result = instantiate(config)
    print(f"Default result: {result}")

    result = instantiate(config, values={"model": "smart"})
    print(f"Override result: {result}")


def test_dict_complex():
    """Test dictionary with complex values."""

    class ModelConfig:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"ModelConfig({self.name})"

    def config(hp: HP):
        options = {"small": ModelConfig("gpt-3.5"), "large": ModelConfig("gpt-4")}
        choice = hp.select(options, default="small", name="model")
        return {"model": choice}

    result = instantiate(config)
    print(f"Complex dict result: {result}")


if __name__ == "__main__":
    test_dict_primitive()
    print()
    test_dict_complex()
