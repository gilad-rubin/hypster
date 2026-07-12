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

The notebook's second cell is the independent oracle for the final exact
`result.params` value. Browser errors, cell errors, visible widget errors,
missing comm-driven replacement DOM, and timeouts fail the run. The browser,
Jupyter server, kernel, and their process group are terminated on every exit.

## Local run

From the repository root:

```bash
uv sync --project tests/jupyterlab_host --frozen
uv build --wheel --out-dir dist
uv run --project tests/jupyterlab_host --frozen playwright install chromium
HYPSTER_HOST_WHEEL="$PWD/dist/hypster-0.8.0-py3-none-any.whl" \
HYPSTER_HOST_ARTIFACT_DIR="$PWD/host-evidence" \
uv run --project tests/jupyterlab_host --frozen \
  pytest tests/jupyterlab_host/test_jupyterlab_host.py -o addopts= -q -s
```

The wheel path and artifact directory are mandatory; the harness has no source
checkout or evidence-path fallback.
