# Currently Tested Environments

This page records Hypster's current CI evidence. It helps you choose a known test combination; it does not promise permanent compatibility with these environments or exclude other combinations that may work.

## Python Package Matrix

The core suite currently runs on:

| Operating system | Python versions |
| --- | --- |
| Ubuntu (`ubuntu-latest`) | 3.10, 3.11, 3.12, 3.13 |
| macOS (`macos-latest`) | 3.13 |
| Windows (`windows-latest`) | 3.13 |

The package metadata requires Python 3.10 or newer. The table is the narrower statement: these are the combinations exercised by [the current CI workflow](https://github.com/gilad-rubin/hypster/blob/master/.github/workflows/ci.yml).

## Notebook Widget Matrix

The 0.9 real-host harness builds a wheel, installs it into a clean Python 3.13 kernel, performs real browser actions, and verifies the resulting Python state.

| Host proof | Current pinned environment | Browser |
| --- | --- | --- |
| JupyterLab | JupyterLab 4.5.9, anywidget 0.11.0, ipywidgets 8.1.8, jupyterlab-widgets 3.0.16, Playwright 1.61.0 | Playwright Chromium |
| Notebook 7 | Notebook 7.5.7, JupyterLab 4.5.9, anywidget 0.11.0, ipywidgets 8.1.8, jupyterlab-widgets 3.0.16, Playwright 1.61.0 | Playwright Firefox |
| VS Code Desktop | Ubuntu 24.04, VS Code 1.128.0, Electron 42.5.0, Chromium 148, Python 3.13.13, Node.js 24.18.0, npm 11.16.0, Python extension 2026.4.0, Jupyter extension 2025.9.1, Notebook Renderers 1.3.0, anywidget 0.11.0 | VS Code's Electron webview |

The web-host matrix passed on 12 July 2026 in [Notebook hosts run 29195385284](https://github.com/gilad-rubin/hypster/actions/runs/29195385284). The VS Code matrix passed on the same date in [VS Code host run 29195330706](https://github.com/gilad-rubin/hypster/actions/runs/29195330706). Each run retains setup versions and host evidence as workflow artifacts.

Notebook 7.6.0 with JupyterLab 4.6.1 produced a frontend exception before any Hypster cell executed. The 0.9 harness therefore pins the newest prior coherent pair it proved green: Notebook 7.5.7 with JupyterLab 4.5.9. This is a recorded test boundary, not a claim that every other patch is incompatible.

## Experimental React Renderer Matrix

`@hypster/react` currently runs its type, test, build, and dry-pack checks on Node.js 22 with Python 3.13. Its physical witness uses React DOM 19.2.0 and Playwright Chromium to complete a Python → React → Python round trip. [React package run 29195064948](https://github.com/gilad-rubin/hypster/actions/runs/29195064948) records the 0.9 acceptance proof.

The package declares React 18 or newer as a peer dependency, but it remains experimental and unpublished. The acceptance run is current evidence, not a stability promise for npm or 1.x.

## Testing Another Environment

For a notebook frontend outside this matrix, install the built wheel with the `viz` extra and verify a branch-sensitive interaction:

1. Render an `interact()` result.
2. Change a branch control through the real DOM.
3. Change a dependent control.
4. Verify `result.params` in a fresh Python cell.
5. Check the browser console, kernel output, and widget transport for errors.

The repository's [notebook host harness](https://github.com/gilad-rubin/hypster/tree/master/tests/jupyterlab_host) and [VS Code host harness](https://github.com/gilad-rubin/hypster/tree/master/tests/vscode_host) are the executable references.
