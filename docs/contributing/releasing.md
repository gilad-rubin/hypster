# Release Hypster

This guide is for maintainers publishing a stable 0.x release through `.github/workflows/release.yml`. The workflow is source-read-only: version and changelog changes must reach `master` through a release-prep pull request before dispatch.

Published filenames, bytes, and tags are immutable. If a published release is bad, yank it on PyPI and release a new patch; never replace its artifacts or move its tag.

## 1. Land The Release-Prep Pull Request

Update these sources in one reviewed pull request:

- `[project].version` in `pyproject.toml`;
- `__version__` in `src/hypster/_version.py`;
- an exact `## [X.Y.Z]` section in `CHANGELOG.md`;
- `uv.lock` if the version change updates the root package entry.

Before:

```toml
[project]
version = "0.8.0"
```

```python
__version__ = "0.8.0"
```

After, for a 0.9.0 release:

```toml
[project]
version = "0.9.0"
```

```python
__version__ = "0.9.0"
```

Run the version test and core local checks. The dry run in the next step remains the authoritative artifact gate.

```bash
uv sync --frozen --extra optuna --dev
uv run pytest tests/test_version.py
uv run ruff format --check .
uv run ruff check .
uv run mypy src/hypster
uv run pytest --no-cov -q
rm -rf dist/ build/
uv build
uvx twine check --strict dist/*
```

Merge the pull request and confirm `master` contains the intended version and changelog section. Do not dispatch a release from a feature branch.

## 2. Run A Dry Run

Dispatch `Release` from `master` with the committed version and `publish_to_pypi=false`. Leave `create_draft=false`; dry runs skip the publication job, so they cannot create a tag or GitHub release regardless of the draft input.

```bash
gh workflow run release.yml \
  --ref master \
  -f version=0.9.0 \
  -f publish_to_pypi=false \
  -f create_draft=false
```

The dry run validates the branch, stable `X.Y.Z` format, both committed version sources, absence of `vX.Y.Z`, and the changelog section. It then runs lint, type checks, tests, build, manifest checks, wheel and sdist smoke tests, a built-wheel Optuna test, strict Twine validation, and checksum generation. The `dist-X.Y.Z` workflow artifact is retained. No bytes are sent to PyPI.

## 3. Publish The Same Commit

After the dry run passes, dispatch the workflow again from the same `master` commit with `publish_to_pypi=true`:

```bash
gh workflow run release.yml \
  --ref master \
  -f version=0.9.0 \
  -f publish_to_pypi=true \
  -f create_draft=false
```

The publication order is fixed:

1. Validate, test, and build.
2. Publish the wheel and sdist to PyPI through Trusted Publishing with attestations.
3. Fetch PyPI metadata and verify its SHA-256 hashes match the bytes built by this run.
4. Create `vX.Y.Z` and the GitHub release, attaching the wheel, sdist, and `SHA256SUMS`.

The tag and GitHub release do not exist until PyPI has accepted and served byte-identical artifacts. Set `create_draft=true` only when you intentionally want the post-PyPI GitHub release hidden for a final editorial check; it does not make PyPI publication a draft.

## 4. Verify The Public Release

Check the workflow, package index, tag target, and attached files:

```bash
gh run list --workflow release.yml --limit 1
gh release view v0.9.0
python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("https://pypi.org/pypi/hypster/0.9.0/json"))["info"]["version"])'
```

Confirm the GitHub release contains the wheel, sdist, and `SHA256SUMS`, and that the tag points to the released `master` commit.

## Recover Without Replacing Published Bytes

If PyPI accepts the artifacts but a later job fails, rerun the workflow for the same commit and version. `skip-existing` leaves the PyPI files untouched, and the GitHub release job independently compares their hashes with the run's `SHA256SUMS` before creating the tag.

If the hashes differ, the run fails. Do not overwrite, delete and re-upload, or retag. Investigate, yank the bad release if necessary, increment the patch version in a new release-prep pull request, and publish the fix forward.
