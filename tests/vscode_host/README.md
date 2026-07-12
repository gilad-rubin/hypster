# VS Code Desktop host spike

This harness exercises issue #98's two explicit spike gates against a real,
exactly pinned VS Code Desktop Electron process on Ubuntu/Xvfb.

## Runtime outcomes

The renderer gate has a supported implementation. The probe contributes a
renderer whose entrypoint extends `jupyter-ipywidget-renderer`, so it runs in
the shared notebook output webview. Its extension-host half communicates only
through the probe's own `NotebookRendererMessaging` channel. If execution
reaches the widget, it emits real branch and numeric DOM events and requires a
replacement numeric node containing `1.25` before the notebook's Python oracle
runs.

Static analysis identifies a possible kernel-selection gate at the public API
boundary:

- `notebook.selectKernel` is a documented built-in command and accepts
  `{ notebookEditor, id, extension }`.
- `id` is an opaque notebook-controller ID, not a kernelspec name.
- the public `vscode.notebooks` namespace can create controllers owned by the
  calling extension, but cannot enumerate another extension's controllers or
  observe the selected controller identity;
- Microsoft Jupyter 2025.9.1 computes its controller IDs with private kernel
  metadata/path logic. This harness deliberately does not copy that algorithm.

The Ubuntu witness determines the outcome rather than assuming it in advance:

- if the pinned Jupyter extension activates but VS Code does not register the
  documented `notebook.selectKernel` command, the artifact reports
  `kernel_selection_gate_failure`;
- if both supported commands fulfill but the creation marker is absent while
  the cell has no execution summary and zero outputs, the cell never executed;
  the artifact reports `kernel_selection_gate_failure` and the job may complete
  successfully;
- a rejected or timed-out selector/execution command, any execution summary,
  or any non-marker output is a red `runtime_failure` rather than a gate;
- if the creation marker appears, every later renderer, messaging, or Python
  oracle failure is a red `runtime_failure`;
- only the complete renderer-to-Python round trip reports
  `basic_scenario_green`.

[`SPIKE_FAILURE.md`](SPIKE_FAILURE.md) is a draft root-owned follow-up payload.
It must not be filed or described as observed until corrected Ubuntu CI
produces a `kernel_selection_gate_failure` artifact.

## Before / after

Before:

```text
The repo had no VS Code Desktop process, no isolated extension installation,
and no supported renderer bridge.
```

After:

```text
real VS Code 1.128.0 Electron on Ubuntu/Xvfb
  -> exact stable Python/Jupyter/Jupyter-renderers extensions
  -> clean installed-wheel kernelspec
  -> documented notebook.selectKernel attempt
  -> raw gate evidence (and full renderer/Python oracle if execution proceeds)
```

## Exact pins

- Node.js `24.18.0`
- npm `11.16.0`
- `@vscode/test-electron==3.0.0`
- VS Code Desktop `1.128.0`
- Microsoft Python `2026.4.0` (stable)
- Microsoft Jupyter `2025.9.1` (stable)
- Microsoft Jupyter Notebook Renderers `1.3.0` (stable)
- Python `3.13.13`
- `uv==0.11.8`
- kernel packages: the checked-in
  `tests/jupyterlab_host/kernel-requirements.lock`

The job records VS Code, Electron, Chromium, Node, npm, OS, kernel Python,
every extension/package version, the wheel SHA-256, the explicit Electron
environment-key allowlist, runtime public notebook API keys, command
result/timeout, cell outputs, and VS Code/Jupyter logs.

The local structural check includes pure classifier falsifiers for rejected and
timed-out commands, missing kernelspec errors, completed/failed execution
summaries, text errors, and non-text outputs.

## Supported upstream seams read before implementation

- [Testing Extensions](https://code.visualstudio.com/api/working-with-extensions/testing-extension)
  documents real Desktop tests through `@vscode/test-electron` and exact-version
  downloads.
- [Continuous Integration](https://code.visualstudio.com/api/working-with-extensions/continuous-integration)
  documents Xvfb for Linux Electron tests.
- [Built-in Commands](https://code.visualstudio.com/api/references/commands)
  documents `notebook.selectKernel`, `notebook.cell.execute`, and the selector's
  void return.
- [Notebook API](https://code.visualstudio.com/api/extension-guides/notebook)
  documents renderer extension points and `NotebookRendererMessaging`.
- [VS Code API](https://code.visualstudio.com/api/references/vscode-api)
  documents that an extension may create messaging only for a renderer it
  contributes; it exposes no foreign-controller discovery API.

## Local structural proof

macOS cannot satisfy the required Ubuntu/Xvfb witness. It can validate the
locked JavaScript installation and harness structure:

```bash
cd tests/vscode_host
npx --yes --package=node@24.18.0 --package=npm@11.16.0 --call \
  'node scripts/assert-runtime.cjs && npm ci --ignore-scripts && npm run check'
```

## Bounded Ubuntu reproduction

The workflow performs setup first, then runs this exact physical command:

```bash
cd tests/vscode_host
timeout --signal=TERM --kill-after=30s 8m xvfb-run -a npm test
```

Required absolute environment variables are `HYPSTER_NOTEBOOK`,
`HYPSTER_VSCODE_ARTIFACT_DIR`, and `HYPSTER_VSCODE_RUNTIME_DIR`. The workflow
also supplies `JUPYTER_PATH` pointing at the isolated clean-kernel prefix.
