# Follow-up issue payload: supported VS Code kernel selection seam

> Superseded historical observation verified by the pinned Ubuntu witness in
> [workflow run 29192874852](https://github.com/gilad-rubin/hypster/actions/runs/29192874852).
> Do not file this as an unavoidable gate: the supported exported
> `openNotebook` seam was physically proven in workflow run 29193331802.

The uploaded `spike-result.json` records VS Code Desktop `1.128.0`, Microsoft
Jupyter `2025.9.1`, and `selector.commandRegistered: false` after explicit
Jupyter activation. It classifies the exact observation as
`kernel_selection_gate_failure`; no notebook cell or substitute renderer path
was attempted. The cleanup artifact confirms that no VS Code, Electron,
Chromium, or kernel process remained.

Pinned Jupyter `2025.9.1` also publicly exports
`openNotebook(uri, pythonEnvironment)`, which selects its internal controller
without exposing or copying the private ID. The harness now exercises the same
Python-environment resolution and `openNotebook` sequence as Jupyter's own
smoke test. Workflow run 29193331802 proved that export against the exact clean
Python environment and reached the widget. Its later widget-CDN prompt failure
is a red harness-configuration failure, not this accepted gate.

Workflow run 29193666883 subsequently proved the exact global widget setting
before and after activation, plus successful startup of that exact clean
kernel. The remaining failure was a concrete renderer source error:
`Failed to access CDN https://unpkg.com/ ... TypeError: Failed to fetch`.
Pinned Jupyter orders configured network/custom providers before its installed
local provider. The harness now uses the public custom-source setting to serve
the exact selected kernel prefix's installed `anywidget/index.js` on loopback,
and requires path/hash/request evidence before green. This does not change the
historical kernel-selection gate recorded by this payload.

Workflow run 29194056537 then proved that both Jupyter's extension host and the
Electron webview fetched the exact installed anywidget bundle. The remaining
renderer timeout is downstream of kernel selection and widget-source delivery.
Pinned Jupyter's own third-party tests warn that offscreen notebook outputs are
virtualized and not rendered, so the harness now explicitly prepares and
reveals the creation-cell viewport and returns renderer diagnostics inside the
unchanged outer deadline. This remains unrelated to the superseded selection
gate described here.

## Title

Expose or adopt a supported VS Code kernel-controller discovery seam for the
real-host harness

## Parent

#86; discovered by #98; spec #93; decision #82.

## Reproduction

Run the dedicated `VS Code host spike` workflow, or on Ubuntu with its setup
environment:

```bash
cd tests/vscode_host
timeout --signal=TERM --kill-after=30s 8m xvfb-run -a npm test
```

The reproduction workflow is designed to launch VS Code Desktop 1.128.0
through `@vscode/test-electron`, use isolated user-data/extensions directories,
install stable
`ms-python.python@2026.4.0`, `ms-toolsai.jupyter@2025.9.1`, and
`ms-toolsai.jupyter-renderers@1.3.0`, and register the clean `hypster-host`
kernelspec from the installed repository wheel.

`notebook.selectKernel` accepts `{ notebookEditor, id, extension }`, but `id`
is the owning extension's opaque controller ID. The public VS Code API exposes
neither foreign-controller enumeration nor selected-controller identity. The
kernelspec name `hypster-host` is not that ID. The exact pinned Jupyter
extension derives controller IDs with private kernel metadata/path logic; the
host harness is forbidden from copying or importing that algorithm.

A reproduction run must preserve `host-evidence/vscode/spike-result.json` with
the public `vscode.notebooks` runtime keys, command result/timeout, exact
process/extension versions, and cell/renderer progress. Its uploaded VS Code
logs must preserve the controller-discovery evidence.

## Acceptance criteria

- [ ] A supported, versioned API lets the external test extension select the
      controller for the exact clean `hypster-host` kernelspec without copying
      Microsoft Jupyter internals or driving private workbench UI.
- [ ] The harness can prove which controller was selected before executing the
      shared notebook.
- [ ] #98 is rerun unchanged through the real renderer-messaging bridge and the
      Python oracle proves `{'mode': 'remote', 'remote.temperature': 1.25}`.
- [ ] Exact versions, logs, timeouts, and process cleanup remain release
      artifacts.

## Forbidden workaround

Do not derive Jupyter's controller ID from its private `getKernelId` algorithm,
reach into extension internals, use private commands/UI selectors, or replace
the real kernel/widget/renderer.
