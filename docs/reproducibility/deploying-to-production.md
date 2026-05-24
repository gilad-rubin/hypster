# Deploying To Production

When you deploy Hypster, treat selected params as part of the release artifact.

## Recommended Pattern

1. Define config functions in version-controlled Python modules.
2. Validate production values in CI with `instantiate(config, values=prod_params)`.
3. Store the exact `params` produced by `instantiate_with_params()`.
4. Log `hypster.__version__`, your app version, and the git commit.
5. Re-run `explore(config, values=prod_params)` during rollout to confirm the active branch.

## Example Smoke Test

{% code overflow="wrap" %}
```python
from hypster import HP, instantiate_with_params
from my_app.deploy import ServiceDeployment

def service_config(hp: HP) -> ServiceDeployment:
    replicas = hp.int(2, name="replicas", min=1, max=20)
    provider = hp.select(["local", "remote"], name="provider", default="remote", options_only=True)
    timeout = hp.float(10.0, name="timeout", min=0.1, max=120.0)
    return ServiceDeployment(replicas=replicas, provider=provider, timeout=timeout)

def test_production_config():
    run = instantiate_with_params(
        service_config,
        values={"replicas": 4, "provider": "remote", "timeout": 30.0},
    )
    assert run.params["replicas"] == 4
```
{% endcode %}

## Avoid Stale Overrides

Keep `on_unknown="raise"` in production. Unknown and unreachable values often indicate a typo, dead branch, or payload from an older config version.

## Keep Configs Portable

Avoid doing network calls, training, or file writes while defining the config. Return values or lightweight objects, then execute effects in application code after instantiation.
