def test_integrations_optuna_import_proxy():
    # Ensure public alias path exists and re-exports
    # Import the real module too to ensure presence
    from hypster.hpo import optuna as real  # noqa: F401
    from hypster.integrations import optuna as alias  # noqa: F401
