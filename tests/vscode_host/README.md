# VS Code Desktop host spike

This harness exercises issue #98's two explicit spike gates against a real,
exactly pinned VS Code Desktop Electron process on Ubuntu/Xvfb.

## Runtime outcomes

The renderer gate has a supported implementation. The probe contributes a
renderer whose entrypoint extends `jupyter-ipywidget-renderer`, so it runs in
the shared notebook output webview. Its extension-host half communicates only
through the probe's own `NotebookRendererMessaging` channel. If execution
reaches the widget, it emits real branch and numeric DOM events and requires a
connected widget root that either replaced and detached the prior root or
changed in place. It then requires a replacement numeric node containing
`1.25` before the notebook's Python oracle runs.

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
harness first wrote that supported setting before activating Jupyter or
creating the widget, recorded its global and effective values, and failed if
either value differed.

[Workflow run 29193526092](https://github.com/gilad-rubin/hypster/actions/runs/29193526092)
confirmed that both inspected global values persisted exactly, but exposed a
harness-verifier bug: `configuration.get()` was called on the configuration
object created before the update and returned its cached default. The harness
now reacquires configuration after writing the setting and verifies the
effective value again after Jupyter activation, at the point the widget
consumer reads it.

[Workflow run 29193666883](https://github.com/gilad-rubin/hypster/actions/runs/29193666883)
then proved the exact setting before and after activation, the exact clean
interpreter, and successful kernel startup. It also exposed the next concrete
boundary: pinned Jupyter tries configured CDN providers before its installed
local nbextension provider, and the Electron webview could not fetch
`https://unpkg.com/`. The unchanged 30-second creation timeout therefore stayed
correctly red.

The harness now uses Jupyter's public custom-source setting without depending
on an external network. Before Jupyter activation it derives the selected
kernel prefix from `HYPSTER_VSCODE_PYTHON`, requires parser-compatible
`share/jupyter/nbextensions/anywidget/extension.js`, and serves that same
installation's `index.js` from a loopback-only HTTP endpoint. It configures the
exact dynamic template
`http://127.0.0.1:<port>/${packageName}/${fileNameWithExt}` globally and verifies
the effective value again after activation. The artifact records both asset
paths, byte counts, SHA-256 hashes, and every request. Even if the renderer
probe otherwise succeeds, the run remains red unless VS Code consumed exact
bytes: either the Electron webview fetched `/anywidget/index.js` directly, or
RequireJS defined `anywidget` from VS Code's supported copied file-resource and
the extension host hashes that actual copied file byte-for-byte against the
selected kernel bundle. An extension-host availability GET alone cannot
satisfy the gate.

[Workflow run 29194056537](https://github.com/gilad-rubin/hypster/actions/runs/29194056537)
proved both sides of that boundary: one GET came from Jupyter's extension-host
source check and a second came from the exact Electron/Chromium webview. The
creation cell completed successfully, but `.hypster-widget` never appeared and
the outer 30-second renderer deadline won the race against an identical
renderer-side wait. The older generic `unpkg.com` reachability log remained,
but the request evidence proves it was not the source used for `anywidget`.

Pinned Jupyter's own third-party widget tests document another necessary host
condition: VS Code does not render a virtualized notebook output that is
outside the visible viewport. Before exercising the renderer, this harness now
uses the same supported workbench commands to close the panel, maximize the
editor, collapse cell inputs, and explicitly reveal the creation cell in the
center. A 25-second renderer-side envelope returns before the unchanged
30-second extension-host deadline. Failure evidence includes body/output DOM,
RequireJS `anywidget` registration state, a bounded blob-module import probe,
the inherited Jupyter renderer/kernel globals, CSP, and browser errors or
unhandled rejections captured from renderer module load.

[Workflow run 29194503005](https://github.com/gilad-rubin/hypster/actions/runs/29194503005)
proved the creation cell was inside VS Code's reported visible range and
executed the clean installed wheel, but only Jupyter's Node-side source check
fetched `anywidget`. The renderer probe again hit the host's generic 30-second
timeout without returning its 20-second diagnostic. Pinned VS Code source
explains why: `NotebookRendererMessaging.postMessage()` returns `true` when the
notebook webview accepts the envelope, even when the target extending renderer
has not loaded. Its message event then has no listener and the message is lost.

The probe now follows the documented renderer-originated messaging pattern.
Its extending renderer first resolves `jupyter-ipywidget-renderer` through
`RendererContext.getRenderer()`, then posts a ready handshake containing base
API and activation diagnostics. The extension host waits at most five seconds
for that ready message before sending the exercise request. The renderer has a
20-second exercise deadline and the host has a 22-second response deadline, so
both stages fail before the unchanged 30-second outer budget. A missing base
renderer now reports an activation-handshake failure instead of a misleading
response timeout.

[Workflow run 29194823654](https://github.com/gilad-rubin/hypster/actions/runs/29194823654)
crossed that boundary: the extending renderer activated, RequireJS defined
`anywidget`, the widget rendered, and the real branch action produced the
remote numeric control. It exposed two false-negative witnesses. First, the
probe retained the detached pre-action `.hypster-widget` and compared its stale
HTML while the global document already contained the replacement root. Second,
the source gate required an Electron loopback GET even though pinned Jupyter
had copied the exact bundle to its supported
`temp/scripts/.../jupyter/nbextensions/anywidget/index.js` file-resource. The
probe now observes the current connected root and the source gate hashes that
copied file. The numeric node replacement and Python verification oracle remain
unchanged.

## Before / after

Before:

```text
exact local anywidget source reaches the webview
  -> host sends before extending renderer installs its listener
  -> VS Code accepts then drops the message
  -> generic outer timeout hides the activation boundary
```

After:

```text
close panel + maximize editor + collapse inputs + reveal creation cell
  -> extending renderer resolves the real Jupyter base and posts ready
  -> host sends only after ready: 5s activation + 22s response < 30s outer
  -> renderer returns its own diagnostic by 20s
  -> success proves the real branch/numeric flow
  -> failure returns DOM + AMD + blob import + inheritance + early errors

exact selected kernel prefix
  -> validate anywidget extension.js mapping and hash installed index.js
  -> loopback-only custom Jupyter source serves that exact index.js
  -> artifact proves either direct Electron GET or AMD-defined copied URL
  -> hash actual copied file against selected-kernel bytes before green

branch action
  -> capture connected root immediately before click
  -> accept distinct connected root only after prior root detaches
  -> otherwise require that same connected root's HTML to change
  -> require numeric node replacement, value 1.25, then Python oracle
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
- `anywidget==0.11.0`
- kernel packages: the checked-in
  `tests/jupyterlab_host/kernel-requirements.lock`

The job records VS Code, Electron, Chromium, Node, npm, OS, kernel Python,
every extension/package version, the wheel SHA-256, the explicit Electron
environment-key allowlist, runtime public notebook API keys, command
result/timeout, cell outputs, and VS Code/Jupyter logs. Marketplace CLI calls
also have a two-minute per-process timeout while retaining status, stdout,
stderr, and spawn errors.

The local structural check includes pure classifier falsifiers for transient
empty creation state, rejected and timed-out commands, missing kernelspec
errors, completed/failed execution summaries, text errors, and non-text
outputs. It also builds a temporary exact-prefix nbextension, proves the custom
template's HEAD/GET behavior and SHA-256 evidence, rejects other routes, and
proves an extension-host GET alone cannot satisfy the gate. It accepts a VS
Code copied resource reported by AMD only when the actual file matches the
selected kernel bytes, rejects a mismatched copy, rejects an incompatible
`extension.js` before server startup, and verifies root-transition and CLI
timeout behavior independently.

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
  documents renderer extension points, renderer-originated messaging, and the
  `onRenderer:<id>` activation event needed before messages are delivered.
- [VS Code 1.128 renderer implementation](https://github.com/microsoft/vscode/blob/1.128.0/src/vs/workbench/contrib/notebook/browser/view/renderers/webviewPreloads.ts)
  loads dependent renderers after their base and exposes that base through
  `RendererContext.getRenderer()`.
- [VS Code 1.128 notebook webview implementation](https://github.com/microsoft/vscode/blob/1.128.0/src/vs/workbench/contrib/notebook/browser/view/renderers/backLayerWebView.ts)
  reports successful host delivery when the webview accepts a message; that
  result does not prove the target renderer installed a listener.
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
- [Jupyter 2025.9.1 script-source factory](https://github.com/microsoft/vscode-jupyter/blob/v2025.9.1/src/notebooks/controllers/ipywidgets/scriptSourceProvider/scriptSourceProviderFactory.node.ts)
  orders configured CDN/custom sources before the installed local provider.
- [Jupyter 2025.9.1 local widget manager](https://github.com/microsoft/vscode-jupyter/blob/v2025.9.1/src/notebooks/controllers/ipywidgets/scriptSourceProvider/localIPyWidgetScriptManager.node.ts)
  discovers `extension.js` below the interpreter's Jupyter data directories and
  parses its RequireJS mapping.
- [Jupyter 2025.9.1 third-party widget tests](https://github.com/microsoft/vscode-jupyter/blob/v2025.9.1/src/test/datascience/widgets/thirdpartyWidgets.vscode.common.test.ts)
  state that outputs outside the viewport are not rendered and prepare the
  workbench by hiding the panel and maximizing the editor.

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
