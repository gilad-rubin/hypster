import pytest

from hypster import HP, instantiate
from hypster.hp_calls import ParameterValidator


def test_nested_float_accepts_int_when_not_strict():
    def cfg(hp: HP):
        def child(hp: HP):
            return {
                "lr": hp.float(1.0, name="lr"),
                "lrs": hp.multi_float([1.0], name="lrs"),
            }

        return {"x": hp.nest(child, name="model")}

    assert instantiate(cfg, values={"model.lr": 1, "model.lrs": [2]}) == {
        "x": {
            "lr": 1.0,
            "lrs": [2.0],
        }
    }


def test_select_multi_non_list_error():
    def cfg(hp: HP):
        return hp.multi_select(["a", "b"], name="choices", default=["a"])  # default list used

    with pytest.raises(ValueError, match="expected list but got"):
        instantiate(cfg, values={"choices": "a"})


def test_select_options_only_long_message():
    def cfg(hp: HP):
        return hp.select(["o1", "o2", "o3", "o4", "o5", "o6"], name="pick", options_only=True)

    # Message includes a truncated options list like: ['o1', 'o2', 'o3', 'o4', 'o5', ... (1 more)]
    with pytest.raises(ValueError, match=r"not in allowed options.*o1.*o2.*o3.*o4.*o5.*more"):
        instantiate(cfg, values={"pick": "x"})


def test_parameter_validator_validate_value_not_implemented():
    class Dummy(ParameterValidator):
        pass

    with pytest.raises(NotImplementedError):
        Dummy().validate_value(1, "p")
