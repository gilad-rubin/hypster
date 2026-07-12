# Follow-up issue payload: supported VS Code kernel selection seam

> Draft only. Do not file this payload or claim the failure was observed until
> a corrected Ubuntu run records `kernel_selection_gate_failure` in
> `host-evidence/vscode/spike-result.json`.

The qualifying artifact must show fulfilled selector and creation commands,
`creationExecutionSummary: null`, `creationOutputCount: 0`, and an empty
`creationRawOutput`. Command rejection/timeout, any execution summary, or any
output is a runtime failure and does not qualify this payload for filing.

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

A qualifying run must preserve `host-evidence/vscode/spike-result.json` with
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
