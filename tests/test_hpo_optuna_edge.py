import pytest

from hypster import HP
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoCategorical, HpoFloat, HpoInt


class TrialSpy:
    def __init__(self):
        self.calls = []

    def suggest_int(self, name, low, high, step=None, log=False):
        self.calls.append((name, low, high, step, log))
        return low

    def suggest_float(self, *args, **kwargs):
        raise AssertionError("not used")

    def suggest_categorical(self, *args, **kwargs):
        raise AssertionError("not used")


def cfg(hp: HP):
    # include_max=False should reduce high by step
    return hp.int(10, name="n", min=0, max=10, hpo_spec=HpoInt(step=2, include_max=False))


def test_include_max_reduces_high():
    t = TrialSpy()
    values = suggest_values(t, config=cfg)
    assert values["n"] == 0
    name, low, high, step, log = t.calls[0]
    assert (name, low, step, log) == ("n", 0, 2, False)
    assert high == 8


def test_include_max_false_keeps_largest_valid_stepped_value_below_max():
    def config(hp: HP):
        return hp.int(0, name="n", min=0, max=10, hpo_spec=HpoInt(step=3, include_max=False))

    t = TrialSpy()
    values = suggest_values(t, config=config)

    assert values["n"] == 0
    name, low, high, step, log = t.calls[0]
    assert (name, low, high, step, log) == ("n", 0, 9, 3, False)


def test_hpo_proxy_rejects_invalid_names_and_nullable_numeric_params():
    def bad_name(hp: HP):
        return hp.int(1, name="bad-name")

    with pytest.raises(ValueError, match="valid Python identifiers"):
        suggest_values(TrialSpy(), config=bad_name)

    def nullable_int(hp: HP):
        return hp.int(None, name="depth", allow_none=True)

    with pytest.raises(ValueError, match="not supported"):
        suggest_values(TrialSpy(), config=nullable_int)


def test_hpo_proxy_rejects_complex_select_choices_with_dict_guidance():
    class CategoricalTrial(TrialSpy):
        def suggest_categorical(self, name, options):
            return options[0]

    def config(hp: HP):
        return hp.select([{"layers": 2}], name="model")

    with pytest.raises(ValueError, match="Use dict-backed select"):
        suggest_values(CategoricalTrial(), config=config)


def test_hpo_proxy_rejects_unsupported_optuna_spec_fields_instead_of_ignoring_them():
    def int_with_custom_log_base(hp: HP):
        return hp.int(1, name="depth", min=1, max=16, hpo_spec=HpoInt(scale="log", base=2.0))

    with pytest.raises(ValueError, match="base"):
        suggest_values(TrialSpy(), config=int_with_custom_log_base)

    def normal_float(hp: HP):
        return hp.float(
            1.0,
            name="lr",
            min=0.001,
            max=1.0,
            hpo_spec=HpoFloat(distribution="normal", center=0.1, spread=0.01),
        )

    with pytest.raises(ValueError, match="normal"):
        suggest_values(TrialSpy(), config=normal_float)

    def weighted_categorical(hp: HP):
        return hp.select(["a", "b"], name="choice", hpo_spec=HpoCategorical(weights=[0.9, 0.1]))

    with pytest.raises(ValueError, match="weights"):
        suggest_values(TrialSpy(), config=weighted_categorical)


def test_hpo_nested_overrides_are_validated_like_instantiation_values():
    def config(hp: HP):
        return hp.nest(
            lambda hp: hp.int(0, name="depth", min=0, max=10, hpo_spec=HpoInt()),
            name="tree",
            values={"depth": 999},
        )

    with pytest.raises(ValueError, match="exceeds maximum bound 10"):
        suggest_values(TrialSpy(), config=config)


def test_hpo_nested_overrides_raise_for_unknown_child_parameters():
    def config(hp: HP):
        return hp.nest(
            lambda hp: hp.int(0, name="depth", min=0, max=10, hpo_spec=HpoInt()),
            name="tree",
            values={"typo": 3},
        )

    with pytest.raises(ValueError, match="Unknown or unreachable parameters"):
        suggest_values(TrialSpy(), config=config)
