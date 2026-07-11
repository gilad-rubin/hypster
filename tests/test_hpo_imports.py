def test_hpo_public_surface():
    """The HPO surface is importable from hypster.hpo directly."""
    from hypster.hpo import (  # noqa: F401
        HpoCategorical,
        HpoFloat,
        HpoInt,
        TrialValueProvider,
        suggest_values,
    )


def test_integrations_namespace_is_gone():
    """hypster.integrations was a wildcard re-export that leaked internals; removed."""
    import importlib.util

    assert importlib.util.find_spec("hypster.integrations") is None
