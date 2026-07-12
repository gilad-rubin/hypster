# Hypster 1.0 release supply-chain audit

Date: 2026-07-12
Wayfinder ticket: [#83](https://github.com/gilad-rubin/hypster/issues/83)
Source snapshot: `1c06c52745061ae637021bb0bf1aba5abd81eb02` on `refactor/production-readiness`

## Verdict

**No-go with the current release workflow.** The committed package source can
produce deterministic, valid, installable distributions. The release workflow
does not preserve that property: it introduces an extra backup module into the
artifacts, can create an unreviewed version commit that receives no CI run,
publishes a GitHub release before PyPI succeeds, and produces no verifiable PyPI
provenance.

The 1.0 release should be a promotion of reviewed bytes, not a workflow that
rewrites source while publishing it.

```text
Before
dispatch any ref -> rewrite/push version -> replace tag -> build -> import-only smoke
                 -> public GitHub release -> download artifact -> PyPI

Required after
reviewed release-prep PR -> protected master + green gates -> validate exact version
                         -> reproducible build + strict artifact tests
                         -> approval -> publish the same bytes with provenance
                         -> public-index verification -> immutable GitHub release
```

## What is already proven

- Three builds of the same clean source—two from the worktree and one from a
  `git archive` without `.git`—were byte-identical under `uv 0.11.8`:

  ```text
  wheel  a444bad2f8aea6df40c9b36582a73e242b47398cd0e99266d7da49ad5125e790
  sdist  24caef60aaa645a588f8413490070c100d60fc020c22b630491cea733d87b99d
  ```

  This is good evidence for the current tool resolution, but not a durable
  reproducibility guarantee: `pyproject.toml:34-36` leaves Hatchling
  unconstrained and `.github/workflows/release.yml:58-62` does not select a uv
  version. uv documents hashed build constraints as its reproducible-build
  control ([uv build constraints](https://docs.astral.sh/uv/concepts/projects/build/#build-constraints)).

- The clean 0.8.0 wheel contains the Python modules, `py.typed`, and the widget
  `interact.js`/`interact.css`; the sdist contains all source required to build
  that wheel plus `pyproject.toml`, README, license, and changelog. The wheel has
  27 entries and the sdist 29. `twine check --strict` passed both artifacts and
  `check-wheel-contents` reported `OK`.

- Real isolated installs passed:

  ```text
  wheel core / Python 3.10: import + instantiate + exact version -> ok
  sdist core / Python 3.14: build + import + instantiate + exact version -> ok
  wheel [viz,optuna] / Python 3.14: widget construction + real Optuna Study -> ok
  ```

  `--no-project` matters because it prevents the source checkout from masking a
  broken distribution ([uv packaging guide](https://docs.astral.sh/uv/guides/package/#installing-your-package)).

- The last successful release run handed the same bytes from its build job to
  PyPI. GitHub artifact `dist-0.7.0` had bundle digest
  `d3f7302032a01ca040088f489aab74d4a0d5b2dafcf663584d6280a643f8e10a`;
  the downloaded wheel and sdist exactly matched PyPI SHA-256 values
  `5627e60f...8dc93b` and `4558ae34...191a8e`. The named artifact flow in
  `.github/workflows/release.yml:174-179,290-300` is therefore a sound seam.
  Evidence: [release run 27917314942](https://github.com/gilad-rubin/hypster/actions/runs/27917314942)
  and [PyPI 0.7.0 files](https://pypi.org/project/hypster/0.7.0/#files).

- PyPI publishing already uses an OIDC-capable `pypi` environment and
  `id-token: write` (`.github/workflows/release.yml:271-300`), avoiding a
  repository-stored upload token. PyPI Trusted Publishing exchanges the CI
  identity for a short-lived token, but explicitly does not prove that the
  artifact was unmodified; attestations supply that missing evidence
  ([PyPI security model](https://docs.pypi.org/trusted-publishers/security-model/)).

## Release blockers and required acceptance evidence

| Gap | Required control | Acceptance evidence for 1.0 |
|---|---|---|
| **The workflow contaminates its own artifacts.** It copies `src/hypster/_version.py.backup` at `.github/workflows/release.yml:71-77`, while the wheel includes the entire package (`pyproject.toml:67-68`). The real 0.7.0 wheel and sdist both shipped that backup file. | Never create backups or edit source in a release job. Build from a clean export of the reviewed commit and enforce an explicit archive manifest. | Wheel/sdist manifests contain only the allowlisted package/data/metadata files; searches for `.backup`, caches, tests, `dev/`, and credentials return empty. The artifacts installed in smoke tests are the exact files later published. |
| **A release can create unreviewed source.** Lines 64-116 rewrite, commit, and push both version files. GitHub says ordinary events caused by a `GITHUB_TOKEN` push do not launch another workflow, so that commit does not receive push CI ([GitHub `GITHUB_TOKEN`](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs)). | Version, changelog, migration notes, and metadata land through a reviewed release-prep PR. The release workflow is source-read-only and rejects non-`master`, a dirty tree, a non-HEAD default-branch SHA, or disagreement among input, `pyproject.toml`, `_version.py`, wheel metadata, and filenames. | Release logs show one reviewed commit SHA; `git diff --exit-code` stays clean; no `git push` or source mutation exists in the workflow; required CI checks belong to that SHA. |
| **Release does not run the release gate.** Lines 137-170 only build, list part of the wheel, import, and print a version; they do not assert the version or run pytest, Ruff, mypy, artifact metadata checks, extras, or supported-Python smoke tests. CI itself does not build distributions (`.github/workflows/ci.yml:8-46`). | Add a blocking packaging job and make release depend on the complete #90 gate. Assert rather than print. Test the real wheel and sdist, not the source checkout. | Green pytest/Ruff/mypy 3.10+3.13 evidence plus `twine check --strict`, wheel-content allowlist, core wheel/sdist smoke at Python floor/latest, extras smoke, and the real-host/real-Optuna evidence defined by #82. |
| **Rebuilds are only accidentally reproducible.** `hatchling` is unbounded (`pyproject.toml:34-36`), setup-uv installs latest when no version is configured, and all actions use mutable tags. GitHub identifies a full commit SHA as the only immutable way to consume an action ([secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use#using-third-party-actions)). | Pin uv, constrain build requirements with exact hashes, build with `uv build --no-sources --build-constraint ... --require-hashes`, and pin every action to a reviewed full SHA (retain version comments for Dependabot). | Two independent clean builds produce identical per-file SHA-256 values; the log records runner image, Python, uv, backend constraints, source SHA, and artifact hashes; repository Actions policy requires SHA pinning. |
| **The changelog extractor is broken and permissive.** The 0.7.0 run logged “No changelog section found” even though `CHANGELOG.md` at that SHA began with `## [0.7.0]`; lines 188-208 then fabricated generic notes. | Replace the shell-quoting parser with a tested extractor and fail when the exact version section, rolled-up migration notes, or release date is absent. Remove the fallback. | A fixture test extracts 1.0.0 exactly; the release body contains the committed 1.0 changelog and migration section; a missing section fails before any tag, release, or upload. |
| **Metadata can contradict the support contract.** Current metadata is `0.8.0`, Beta, and lists Python only through 3.13 (`pyproject.toml:3,12-20`), while #82 chose Python 3.10-3.14. | Make distribution metadata an explicit assertion surface, including 1.0.0, production classifier, Python 3.10-3.14, optional-dependency bounds, project URLs, license, README, and `py.typed`. | Inspect wheel `METADATA` and sdist `PKG-INFO`; compare both with the #82 matrix and release input; install every extra from the built wheel. |
| **No package provenance is published.** Both PyPI 0.7.0 Integrity API endpoints return `404: No provenance available`. `uv publish` can upload attestations but does not generate them ([uv guide](https://docs.astral.sh/uv/guides/package/#uploading-attestations-with-your-package)). | Publish through the official PyPA action, pinned by SHA, or generate PEP 740 attestations before `uv publish`. Keep OIDC and give the publish job only `id-token: write` and `contents: read`. | PyPI Integrity API returns 200 for both files and `pypi-attestations verify pypi --repository https://github.com/gilad-rubin/hypster <file-url>` passes. PyPI explains that an attestation binds a distribution digest to its publisher identity ([PyPI attestations](https://docs.pypi.org/attestations/)). |
| **Artifact identity is not public or durable.** The v0.7.0 GitHub release has no attached wheel/sdist, is not immutable, and the workflow publishes no checksum manifest. | Attach the exact PyPI-bound wheel, sdist, and `SHA256SUMS` to a draft release; publish it only after PyPI verification; enable GitHub immutable releases. | GitHub release assets, workflow artifact, and PyPI files have identical SHA-256 values; the release reports immutable. Immutable releases lock their tag/assets and add a release attestation ([GitHub immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)). |
| **Ordering and rollback are unsafe.** Lines 118-135 delete/recreate an existing tag; lines 217-253 publish the GitHub release before the PyPI job. There is no concurrency guard or rollback runbook. PyPI does not allow a distribution filename to be reused ([PyPI file-reuse policy](https://pypi.org/help/#file-name-reuse)). | Add release concurrency, fail if version/tag/release already exists, stage the GitHub release as draft, and publish it only after PyPI and public-install checks. Rollback means yank the bad PyPI version, deprecate/note it, fix forward with a new patch, and publish a security advisory when relevant—never replace bytes or move a release tag. | A rehearsal proves duplicate dispatch cannot race or replace a tag; a written drill identifies the yank/advisory/patch steps and owners; a forced publish failure leaves no public GitHub release claiming success. |
| **Repository/account protections do not enforce the human gate.** Live API audit: `master` is unprotected; `pypi` has no reviewer or branch policy; Actions permits all actions, does not require SHA pins, and defaults tokens to write. | Protect `master` with PR + required checks; restrict the `pypi` environment to `master` and a maintainer approval; make default workflow permissions read-only; require SHA-pinned actions. GitHub environments can require reviewers and branch restrictions ([GitHub environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)). | API evidence shows the branch/ruleset, required checks, environment reviewer/branch policy, default read permission, and SHA-pinning policy before the 1.0 workflow is dispatched. |
| **Vulnerability intake is only partially real.** `SECURITY.md:1-7` gives an email and seven-day target, but says “GitHub Security Advisories” while private vulnerability reporting is disabled. It has no supported-version or remediation/yank policy. Secret scanning and CodeQL are disabled; Dependabot alerts are enabled with zero open alerts. | Enable private vulnerability reporting; expand `SECURITY.md` with supported versions, private channel, acknowledgement/triage expectations, disclosure and patch/yank process; enable secret scanning/push protection and CodeQL; add npm Dependabot coverage. GitHub documents private reporting as the structured confidential intake for public repositories ([GitHub vulnerability reporting](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository)). | “Report a vulnerability” is visible to an external reporter; a test draft reaches the maintainer; security settings are enabled; no unresolved high/critical dependency alert exists; the policy names the supported 1.x line and rollback owner. |
| **`@hypster/react` has no releasable npm path.** `packages/react/package.json:1-25` has only a build script, no tests/repository/publish policy; CI and Dependabot ignore it. `npm pack --dry-run` produced 29 entries containing only `dist/` and `package.json`, with no README or license file. The registry currently returns 404. | Keep npm publication separate from PyPI and behind its own explicit maintainer approval, as decided in #81. Add clean `npm ci`, tests, build, pack-manifest check, consumer install smoke, README/license/repository metadata, npm Dependabot, and a dedicated OIDC trusted-publisher workflow. | Protocol V1 contract tests and package gates pass; tarball manifest is reviewed; a fresh consumer imports types/runtime; maintainer verifies control of the `@hypster` scope. If publication is approved, npm provenance is present. npm Trusted Publishing uses OIDC and automatically generates provenance for a public package from a public GitHub repo ([npm trusted publishing](https://docs.npmjs.com/trusted-publishers/)). |

## Required release shape

1. **Release-prep PR:** set both version sources to `1.0.0`, update classifiers
   and dependency bounds, roll the changelog into an exact 1.0 section, include
   migration notes, and pass all #90 checks on protected `master`.
2. **Read-only validation/build job:** verify ref/SHA/version/changelog, build
   twice from a clean export with pinned inputs, compare hashes, inspect
   manifests/metadata, and run real wheel/sdist/extras smoke tests.
3. **Artifact promotion:** upload exactly the verified files plus checksums; the
   publish job downloads that artifact and verifies hashes before obtaining an
   OIDC credential.
4. **Explicit approval:** the maintainer approves the production environment;
   no package or release is public before this point.
5. **Publish and prove:** publish wheel/sdist with PEP 740 attestations, download
   them from PyPI without cache, compare hashes, install on the Python
   floor/latest, and verify both Integrity API provenance objects.
6. **Finalize:** attach the same bytes/checksums to the GitHub draft release,
   publish it as immutable, and record source SHA, artifact hashes, workflow run,
   PyPI URLs, and smoke evidence in #90.

## Maintainer-only account checklist

No account settings were changed during this audit.

- GitHub: protect `master`; configure `pypi` with a required reviewer and
  `master`-only deployment; enable immutable releases, private vulnerability
  reporting, secret scanning/push protection, and CodeQL; change default Actions
  permission to read and require full-SHA action references. Delete the stale
  repository secret named `PYPI_API_TOKEN` after confirming OIDC is the only
  production path.
- PyPI: verify the Trusted Publisher is exactly repository
  `gilad-rubin/hypster`, workflow `release.yml`, environment `pypi`; remove any
  obsolete long-lived automation token; verify account 2FA and offline recovery
  access; after a rehearsal, verify both artifact provenance endpoints.
- npm, only if separately approved: verify ownership of the `@hypster` scope;
  create/configure the public package and an exact GitHub-workflow trusted
  publisher; remove obsolete write tokens. Scoped packages need an explicit
  public publication decision ([npm scoped public packages](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/)).

## Concise recommendation

Keep Hatchling, uv, GitHub artifacts, and PyPI Trusted Publishing—the clean
build and byte handoff are sound. Replace in-workflow version mutation with a
reviewed release-prep PR; make packaging a strict, reproducible gate; generate
PEP 740 provenance; publish the exact verified bytes; harden GitHub/PyPI account
boundaries; and treat rollback as yank plus a new patch. Do not couple npm
publication to the Python release.

## Scope and routing decisions

- `publish_to_pypi=false` becomes a true dry run: it may build, test, attest,
  and retain an Actions artifact, but must not push source or tags, create a
  GitHub release, or contact an upload endpoint.
- A partial PyPI upload may be resumed only with a missing file from the exact
  retained and already-vetted artifact set, after comparing every accepted
  file hash. Any wrong byte or uncertain provenance requires a yank and a new
  patch version. A complete broken release is always yanked and fixed forward;
  tags and published filenames are never reused.
- `SECURITY.md` must replace the inaccurate claim that Hypster performs "no
  deserialization." The truthful boundary is that it accepts validated
  JSON-compatible data but does not use code-executing deserializers such as
  pickle, or `exec`/`eval`, on untrusted input.
- A separate TestPyPI rehearsal is useful but not required for the 1.0 gate:
  TestPyPI has separate accounts and publisher configuration, so it does not
  prove the production trust path. Exact local artifact tests, the protected
  production OIDC identity, post-publish PyPI hashes/provenance, and clean
  installs from real PyPI are mandatory.
- Traditional GPG signatures, a release-specific SBOM, and OpenSSF Scorecard
  are not 1.0 blockers. PEP 740 plus GitHub attestations provide the required
  provenance; a GitHub dependency-graph SBOM already exists.
- #88 owns the pipeline and account-control implementation, #89 owns the
  release/security/migration documentation, and #90 records the final evidence.
  #87 owns the React package gates. npm publication remains a distinct explicit
  maintainer approval and never rides along with the PyPI workflow.
- The existing tickets cover every concrete finding, so this audit graduates no
  new fog item and invalidates no existing ticket.

The production command remains a hard human gate and must not run until #90 is
green and the maintainer explicitly approves it:

```bash
gh workflow run release.yml -f version=1.0.0 -f publish_to_pypi=true
```
