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

The pinned Ubuntu witness identifies a kernel-selection gate at the public API
boundary:

- `notebook.selectKernel` is a documented built-in command and accepts
  `{ notebookEditor, id, extension }`.
- `id` is an opaque notebook-controller ID, not a kernelspec name.
- the public `vscode.notebooks` namespace can create controllers owned by the
  calling extension, but cannot enumerate another extension's controllers or
  observe the selected controller identity;
- Microsoft Jupyter 2025.9.1 computes its controller IDs with private kernel
  metadata/path logic. This harness deliberately does not copy that algorithm.

The harness determines the outcome rather than assuming it in advance:

- if the pinned Jupyter extension exposes neither its public `openNotebook`
  API nor the documented `notebook.selectKernel` command after activation, the
  artifact reports `kernel_selection_gate_failure`;
- once either supported selection route is available, a rejected or timed-out
  selection/execution request, any creation-marker timeout, any execution
  summary, or any non-marker output is a red `runtime_failure`. An empty cell at
  the marker deadline is inconclusive because uncancelled execution may still
  start;
- if the creation marker appears, every later renderer, messaging, or Python
  oracle failure—including verification-command rejection or timeout—is a red
  `runtime_failure`;
- only the complete renderer-to-Python round trip reports
  `basic_scenario_green`.

The first exact pinned witness completed in
[workflow run 29192874852](https://github.com/gilad-rubin/hypster/actions/runs/29192874852).
After explicit activation, Microsoft Jupyter `2025.9.1` did not register the
documented `notebook.selectKernel` command in VS Code Desktop `1.128.0`. The
uploaded evidence therefore records `kernel_selection_gate_failure`, with no
round-trip attempt and complete process cleanup. [`SPIKE_FAILURE.md`](SPIKE_FAILURE.md)
captures that verified historical observation.

## Supported exported seam

Follow-up source review found a supported route that does not require the
missing built-in command or copying Jupyter's private controller IDs. In pinned
Jupyter `2025.9.1`, the extension's public, non-breaking API exports
`openNotebook(uri, pythonEnvironment)`. Jupyter's own smoke test resolves an
exact executable with `PythonExtension.api().environments.resolveEnvironment()`
and passes the resolved environment to that export.

The harness follows that same path first. It pins
`@vscode/python-extension==1.0.6`, records both facade/export keys, requires the
resolved executable to equal the isolated installed-wheel Python, and calls
`jupyterApi.openNotebook()` before cell execution. The command-based route is
only a fallback when the public export is absent.

The second exact witness in
[workflow run 29193331802](https://github.com/gilad-rubin/hypster/actions/runs/29193331802)
proved that path through the clean kernel: the Python facade resolved the exact
isolated executable, `openNotebook` fulfilled, and the creation cell imported
Hypster from `site-packages`. The run then stayed correctly red because Jupyter
refused an interactive widget-CDN prompt in its test host.

Jupyter's own standard and widget tests avoid that prompt by globally setting
`jupyter.widgetScriptSources` to exactly `['jsdelivr.com', 'unpkg.com']`. The
harness now writes that supported setting before activating Jupyter or creating
the widget, records its global and effective values, and fails if either value
differs. A new Ubuntu witness is still required to prove the full
renderer-to-Python round trip.

[Workflow run 29193526092](https://github.com/gilad-rubin/hypster/actions/runs/29193526092)
confirmed that both inspected global values persisted exactly, but exposed a
harness-verifier bug: `configuration.get()` was called on the configuration
object created before the update and returned its cached default. The harness
now reacquires configuration after writing the setting and verifies the
effective value again after Jupyter activation, at the point the widget
consumer reads it.

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
- `@vscode/python-extension==1.0.6`
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

The local structural check includes pure classifier falsifiers for transient
empty creation state, rejected and timed-out commands, missing kernelspec
errors, completed/failed execution summaries, text errors, and non-text
outputs.

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
- [Jupyter 2025.9.1 public API](https://github.com/microsoft/vscode-jupyter/blob/v2025.9.1/src/standalone/api/index.ts)
  exports `openNotebook` and forbids breaking changes to the extension API.
- [Jupyter 2025.9.1 smoke test](https://github.com/microsoft/vscode-jupyter/blob/v2025.9.1/src/test/smoke/datascience.smoke.test.ts)
  resolves an exact Python environment through the official Python-extension
  facade before calling `jupyterExt.exports.openNotebook`.
- [Jupyter 2025.9.1 standard widget test](https://github.com/microsoft/vscode-jupyter/blob/v2025.9.1/src/test/datascience/widgets/standardWidgets.vscode.common.test.ts)
  sets `widgetScriptSources` globally to `jsdelivr.com` and `unpkg.com` before
  initializing widgets in CI.

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
`HYPSTER_VSCODE_ARTIFACT_DIR`, `HYPSTER_VSCODE_RUNTIME_DIR`, and
`HYPSTER_VSCODE_PYTHON`. The workflow also supplies `JUPYTER_PATH` pointing at
the isolated clean-kernel prefix.
