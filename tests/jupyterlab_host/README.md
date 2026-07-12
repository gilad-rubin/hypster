# JupyterLab real-host harness

This harness proves the current Hypster anywidget through a real JupyterLab 4
server, kernel, and Chromium page. Its sole notebook fixture is
`branch_round_trip.ipynb`.

The test creates a temporary kernel environment, installs the repository's
built wheel with `[viz]` from the checked-in kernel lock, and copies the fixture
outside the source checkout. It then crosses the real boundary:

```text
Chromium DOM event -> anywidget comm -> Python controller -> replacement DOM
```

Notebook verification cells are independent Python oracles for auto-applied,
staged, invalid, applied, and reset `result.params` values. The canonical
scenario also switches the real JupyterLab theme, verifies Branch Choice
Memory, drives Apply and Reset in manual mode, and sends a mismatched Protocol
V1 snapshot through the live widget comm before recovering with a valid
snapshot. Reused Python oracle cells must publish a new execution count and
unique output identity. Browser errors, cell errors, unexpected visible widget
errors, missing comm-driven replacement DOM, and timeouts fail the run. The
browser, Jupyter server, kernel, and their process group are terminated on
every exit.

## Local run

From the repository root:

```bash
uv sync --project tests/jupyterlab_host --frozen
uv build --wheel --out-dir dist
uv run --project tests/jupyterlab_host --frozen playwright install chromium
WHEEL=$(find "$PWD/dist" -maxdepth 1 -name '*.whl' -print -quit)
HYPSTER_HOST_WHEEL="$WHEEL" \
HYPSTER_HOST_ARTIFACT_DIR="$PWD/host-evidence" \
uv run --project tests/jupyterlab_host --frozen \
  pytest tests/jupyterlab_host/test_jupyterlab_host.py -o addopts= -q -s
```

The wheel path and artifact directory are mandatory; the harness has no source
checkout or evidence-path fallback.
