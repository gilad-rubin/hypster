from hypster import HP
from hypster.hpo.optuna import suggest_values
from hypster.hpo.types import HpoInt


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
