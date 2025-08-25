import pytest

from hypster import HP, config


# Helper configs that will be used across tests
@config
def nested_config(hp: HP):
    nested_param = hp.select(["a", "b"], default="a", name="nested_param")
    nested_number = hp.number(default=1.0, name="nested_number")


@config
def deep_nested_config(hp: HP):
    deep_param = hp.select(["deep1", "deep2"], default="deep1", name="deep_param")
    deep_number = hp.number(default=2.0, name="deep_number")


@config
def middle_config(hp: HP):
    deep = hp.nest("tests/helper_configs/deep_nested_config.py", name="deep")
    middle_param = hp.select(["mid1", "mid2"], default="mid1", name="middle_param")


nested_config.save("tests/helper_configs/nested_config.py")
deep_nested_config.save("tests/helper_configs/deep_nested_config.py")
middle_config.save("tests/helper_configs/middle_config.py")


@pytest.fixture(scope="session", autouse=True)
def setup_config_files():
    nested_config.save("tests/helper_configs/nested_config.py")
    deep_nested_config.save("tests/helper_configs/deep_nested_config.py")
    middle_config.save("tests/helper_configs/middle_config.py")


# Basic Two-Layer Nesting Tests
def test_basic_two_layer_defaults():
    @config
    def main_config(hp: HP):
        nested = hp.nest("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    result = main_config()
    assert result["main_param"] == "x"
    assert result["nested"]["nested_param"] == "a"
    assert result["nested"]["nested_number"] == 1.0


def test_basic_dot_notation_values():
    @config
    def main_config(hp: HP):
        nested = hp.nest("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test dot notation selections
    result = main_config(values={"nested.nested_param": "b", "nested.nested_number": 1.5})
    assert result["nested"]["nested_param"] == "b"
    assert result["nested"]["nested_number"] == 1.5
    assert result["main_param"] == "x"


def test_basic_dot_notation_values():
    @config
    def main_config(hp: HP):
        nested = hp.nest("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test dot notation overrides
    result = main_config(values={"main_param": "y", "nested.nested_param": "b", "nested.nested_number": 1.5})
    assert result["main_param"] == "y"
    assert result["nested"]["nested_param"] == "b"
    assert result["nested"]["nested_number"] == 1.5


def test_basic_dict_style_values():
    @config
    def main_config(hp: HP):
        nested = hp.nest("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test dictionary style selections
    result = main_config(values={"nested": {"nested_param": "b", "nested_number": 1.5}})
    assert result["nested"]["nested_param"] == "b"
    assert result["nested"]["nested_number"] == 1.5
    assert result["main_param"] == "x"


def test_basic_dict_style_values():
    @config
    def main_config(hp: HP):
        nested = hp.nest("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test dictionary style overrides
    result = main_config(values={"main_param": "y", "nested": {"nested_param": "b", "nested_number": 1.5}})
    assert result["main_param"] == "y"
    assert result["nested"]["nested_param"] == "b"
    assert result["nested"]["nested_number"] == 1.5


def test_basic_final_vars():
    @config
    def main_config(hp: HP):
        nested = hp.nest("tests/helper_configs/nested_config.py", name="nested")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test dot notation final vars
    result = main_config(final_vars=["nested.nested_param"])
    assert "nested_param" in result["nested"]
    assert "nested_number" not in result["nested"]

    # Test list style final vars
    result = main_config(final_vars=["nested.nested_number"])
    assert "nested_number" in result["nested"]
    assert "nested_param" not in result["nested"]


# Three-Layer Nesting Tests
def test_three_layer_defaults():
    @config
    def main_config(hp: HP):
        middle = hp.nest("tests/helper_configs/middle_config.py", name="middle")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    result = main_config()
    assert result["main_param"] == "x"
    assert result["middle"]["middle_param"] == "mid1"
    assert result["middle"]["deep"]["deep_param"] == "deep1"
    assert result["middle"]["deep"]["deep_number"] == 2.0


def test_three_layer_dot_notation():
    @config
    def main_config(hp: HP):
        middle = hp.nest("tests/helper_configs/middle_config.py", name="middle")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test selections through all layers
    result = main_config(
        values={"middle.middle_param": "mid2", "middle.deep.deep_param": "deep2", "middle.deep.deep_number": 2.5},
    )
    assert result["middle"]["middle_param"] == "mid2"
    assert result["middle"]["deep"]["deep_param"] == "deep2"
    assert result["middle"]["deep"]["deep_number"] == 2.5


def test_three_layer_dict_style():
    @config
    def main_config(hp: HP):
        middle = hp.nest("tests/helper_configs/middle_config.py", name="middle")
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test dictionary style through all layers
    result = main_config(
        values={"middle": {"middle_param": "mid2", "deep": {"deep_param": "deep2", "deep_number": 2.5}}},
    )
    assert result["middle"]["middle_param"] == "mid2"
    assert result["middle"]["deep"]["deep_param"] == "deep2"
    assert result["middle"]["deep"]["deep_number"] == 2.5


def test_inner_configurations():
    @config
    def main_config(hp: HP):
        nested = hp.nest(
            "tests/helper_configs/nested_config.py",
            name="nested",
            final_vars=["nested_param", "nested_number"],
            values={"nested_param": "b", "nested_number": 1.5},
        )
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test that inner configurations work
    result = main_config()
    assert result["nested"]["nested_param"] == "b"
    assert result["nested"]["nested_number"] == 1.5

    # Test that outer selections override inner selections
    result = main_config(values={"nested.nested_param": "a"})
    assert result["nested"]["nested_param"] == "a"

    # Test that outer overrides override inner overrides
    result = main_config(values={"nested.nested_number": 2.0})
    assert result["nested"]["nested_number"] == 2.0

    # Test that outer final_vars override inner final_vars
    # result = main_config(final_vars=["nested.nested_number"])
    # assert "nested_number" in result["nested"]
    # assert "nested_param" not in result["nested"]


def test_inner_configurations_dict_style():
    @config
    def main_config(hp: HP):
        nested = hp.nest(
            "tests/helper_configs/nested_config.py",
            name="nested",
            final_vars=["nested_param", "nested_number"],
            values={"nested_param": "b", "nested_number": 1.5},
        )
        main_param = hp.select(["x", "y"], default="x", name="main_param")

    # Test that inner configurations work with dict style
    result = main_config()
    assert result["nested"]["nested_param"] == "b"
    assert result["nested"]["nested_number"] == 1.5

    # Test that outer dict selections override inner selections
    result = main_config(values={"nested": {"nested_param": "a"}})
    assert result["nested"]["nested_param"] == "a"

    # Test that outer dict overrides override inner overrides
    result = main_config(values={"nested": {"nested_number": 2.0}})
    assert result["nested"]["nested_number"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__])
