# Upgrade From 0.8 To 0.9

Hypster 0.9 adds a version check to interactive messages and replaces the repository's React configuration model with an experimental snapshot renderer. Python-only applications that do not dispatch interactive actions can upgrade without changing their config functions.

Install the target release and check the version:

```bash
uv add "hypster==0.9.*"
uv run python -c "import hypster; print(hypster.__version__)"
```

## Add `protocol_version` To Interactive Actions

Custom clients that call `InteractiveResult.dispatch()` must send Protocol V1 on every action. A missing or mismatched version raises `ValueError` before session state changes.

Before:

```python
snapshot = result.dispatch({"type": "set_value", "path": "count", "value": 4})
```

After:

<!-- test: exec -->
```python
from hypster import HP, interact


def config(hp: HP):
    return {"count": hp.int(1, name="count", min=1, max=5)}


result = interact(config)
snapshot = result.dispatch(
    {"protocol_version": 1, "type": "set_value", "path": "count", "value": 4}
)

assert result.value == {"count": 4}
assert snapshot["protocol_version"] == 1
assert snapshot["selected_params"] == {"count": 4}
```

`protocol_version` is mismatch protection: it stops an incompatible renderer and backend from silently corrupting state. Protocol V1 may still change across 0.x releases. It does not freeze this wire shape for a future 1.x release.

## Replace React Configuration Logic With Snapshot Rendering

`@hypster/react` no longer fetches schemas, reconciles values, or keeps branch memory in the browser.

Before:

```tsx
const config = useConfigSchema({ fetchSchema, kind, values });
```

After:

```tsx
import { HypsterRenderer } from "@hypster/react";

<HypsterRenderer
  snapshot={snapshotFromPython}
  onAction={async (action) => {
    const nextSnapshot = await sendActionToPython(action);
    setSnapshot(nextSnapshot);
  }}
/>
```

The host sends an emitted action to the Python `InteractiveSession`, then replaces the previous snapshot with Python's response. Read `snapshot.selected_params` as the authoritative applied configuration. Python owns validation, reachable branches, draft and applied values, Branch Choice Memory, selected parameters, and apply/reset semantics. React owns presentation state.

The React package is experimental and private to this repository. It is not published to npm, and 0.9 does not promise package or wire compatibility with 1.x. Repository consumers should install it from the workspace and pin the exact commit they test.

## Treat Environment Claims As Current Test Evidence

The 0.9 host matrix records real installed-wheel round trips in JupyterLab, Notebook 7, and VS Code. Those results describe environments tested for this release line; they are not a permanent support or compatibility guarantee.

Check [Currently Tested Environments](../reference/currently-tested-environments.md) for the exact CI pins and proof runs. When you deploy on a different frontend, browser, extension, operating system, or version, run a smoke test that crosses DOM → widget transport → Python → replacement DOM in your environment.

## Verify The Upgrade

Search custom transports for unversioned actions and old React configuration helpers:

```bash
rg -n 'dispatch\(\{[^}]*"type"|fetchSchema|useConfigSchema|useSchemaField|useRulesField' .
```

Then run the Python and React suites used by your application. For all user-visible 0.9 changes, see the [changelog](../../CHANGELOG.md).
