# Notebook real-host harness

This harness proves the current Hypster anywidget through real JupyterLab 4 +
Chromium and Notebook 7 + Firefox hosts. Both use the sole notebook fixture,
`branch_round_trip.ipynb`, and the same basic branch/numeric round trip.

The test creates a temporary kernel environment, installs the repository's
built wheel with `[viz]` from the checked-in kernel lock, and copies the fixture
outside the source checkout. It then crosses the real boundary:

```text
Browser DOM event -> anywidget comm -> Python controller -> replacement DOM
```

The Notebook 7 job runs the shared basic scenario and its fresh Python oracle.
The canonical JupyterLab scenario additionally switches the real host theme,
verifies Branch Choice Memory, drives Apply and Reset in manual mode, and sends
a mismatched Protocol V1 snapshot through the live widget comm before recovery.
Reused Python oracle cells must publish a new execution count and unique output
identity. Browser errors, cell errors, unexpected visible widget errors,
missing comm-driven replacement DOM, and timeouts fail the run. The browser,
Jupyter server, kernel, and their process group terminate on every exit.

## Local run

From the repository root:

```bash
uv sync --project tests/jupyterlab_host --frozen
uv build --wheel --out-dir dist
uv run --project tests/jupyterlab_host --frozen playwright install chromium firefox
WHEEL=$(find "$PWD/dist" -maxdepth 1 -name '*.whl' -print -quit)
HYPSTER_HOST_WHEEL="$WHEEL" \
HYPSTER_HOST_ARTIFACT_DIR="$PWD/host-evidence-jupyterlab" \
uv run --project tests/jupyterlab_host --frozen \
  pytest tests/jupyterlab_host/test_jupyterlab_host.py::test_real_jupyterlab_round_trip -o addopts= -q -s

HYPSTER_HOST_WHEEL="$WHEEL" \
HYPSTER_HOST_ARTIFACT_DIR="$PWD/host-evidence-notebook7" \
uv run --project tests/jupyterlab_host --frozen \
  pytest tests/jupyterlab_host/test_jupyterlab_host.py::test_real_notebook7_round_trip -o addopts= -q -s
```

The wheel path and artifact directory are mandatory; the harness has no source
checkout or evidence-path fallback.

## Notebook 7.6.0 regression

Issue #108 records a Notebook `7.6.0` / JupyterLab `4.6.1` page exception
observed in Firefox before any Hypster cell executes. The currently-tested host
pair is pinned to Notebook `7.5.7` / JupyterLab `4.5.9`, the newest prior
coherent pair proven to load with zero page errors and complete the physical
round trip. The harness still records errors before widget execution, after the
basic round trip, and at shutdown, then fails without filtering host errors.

The required fault injections were also exercised against the physical host:

- Sending `1.50` while the shared oracle requires `1.25` failed at the
  replacement-DOM value assertion (`1.5 != 1.25`).
- Leaving the previous verification output in place while making the next cell
  execution a no-op failed because the prompt stayed at `[2]:`; stale marker
  text did not satisfy the fresh-execution check.
