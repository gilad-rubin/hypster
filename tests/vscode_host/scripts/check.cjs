"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const pins = require("../pins.cjs");

const root = path.resolve(__dirname, "..");
const repository = path.resolve(root, "..", "..");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const renderer = manifest.contributes.notebookRenderer[0];
if (renderer.entrypoint.extends !== "jupyter-ipywidget-renderer") {
  throw new Error("probe must extend the real Jupyter IPyWidget renderer");
}
if (renderer.requiresMessaging !== "always") {
  throw new Error("probe must require the supported renderer messaging boundary");
}
if (manifest.devDependencies["@vscode/test-electron"] !== "3.0.0") {
  throw new Error("@vscode/test-electron must remain exactly pinned");
}
if (manifest.devDependencies["@vscode/python-extension"] !== "1.0.6") {
  throw new Error("@vscode/python-extension must remain exactly pinned");
}
if (
  JSON.stringify(manifest.extensionDependencies) !==
  JSON.stringify(["ms-python.python", "ms-toolsai.jupyter"])
) {
  throw new Error("the host probe must activate the exact Python and Jupyter extension APIs");
}
if (
  JSON.stringify(manifest.activationEvents) !==
  JSON.stringify(["onRenderer:hypster-vscode-host-probe"])
) {
  throw new Error("the extension host must activate from its renderer-originated ready message");
}
if (manifest.packageManager !== `npm@${pins.npm}` || manifest.engines.node !== pins.node) {
  throw new Error("Node.js and npm must remain exactly pinned");
}
if (pins.vscode !== "1.128.0") {
  throw new Error("VS Code Desktop must remain exactly pinned");
}
if (pins.widgetScriptSourcePathTemplate !== "/${packageName}/${fileNameWithExt}") {
  throw new Error("Jupyter's local widget source path template must remain exactly pinned");
}

const hostTestSource = fs.readFileSync(path.join(root, "test", "index.cjs"), "utf8");
for (const fragment of [
  'getConfiguration("jupyter")',
  '"widgetScriptSources"',
  "vscode.ConfigurationTarget.Global",
  'inspect("widgetScriptSources")',
  "effectiveBeforeActivation",
  "effectiveAfterActivation",
  "startLocalWidgetSource(pythonExecutable)",
  "localWidgetSource.assertUsed(",
  "prepareWidgetViewport(evidence, editor, creationCell)",
  "NotebookEditorRevealType.InCenter",
  "creationCellVisible",
]) {
  if (!hostTestSource.includes(fragment)) {
    throw new Error(`host test must configure and verify the global widget allowlist: ${fragment}`);
  }
}
const configurationReads = hostTestSource.match(/getConfiguration\("jupyter"\)/g) ?? [];
if (configurationReads.length < 3) {
  throw new Error("widget settings must be reacquired after update and after Jupyter activation");
}
const activationIndex = hostTestSource.indexOf("const jupyterApi = await jupyter.activate();");
const consumerVerificationIndex = hostTestSource.indexOf(
  "verifyWidgetScriptSourcesAfterActivation(evidence, localWidgetSource.template);",
);
if (activationIndex < 0 || consumerVerificationIndex < activationIndex) {
  throw new Error("effective widget settings must be verified after Jupyter activation");
}

const rendererSource = fs.readFileSync(path.join(root, "renderer.mjs"), "utf8");
const extensionSource = fs.readFileSync(path.join(root, "extension.cjs"), "utf8");
for (const fragment of [
  "const EXERCISE_TIMEOUT_MS = 20_000;",
  "context.getRenderer(BASE_RENDERER_ID)",
  'kind: "hypster-probe-ready"',
  'window.addEventListener(\n  "error"',
  'window.addEventListener("unhandledrejection"',
  "RENDERER_DIAGNOSTICS",
  "baseRendererContextPresent",
  "blobModuleImport",
  "bodyHtml",
]) {
  if (
    !rendererSource.includes(fragment) &&
    !extensionSource.includes(fragment) &&
    !hostTestSource.includes(fragment)
  ) {
    throw new Error(`renderer diagnostics must remain fail-loud: ${fragment}`);
  }
}

const notebookPath = path.join(repository, "tests", "jupyterlab_host", "branch_round_trip.ipynb");
const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
if (notebook.metadata.kernelspec.name !== "hypster-host") {
  throw new Error("the shared host fixture's clean kernelspec changed");
}
for (const marker of ["HYPSTER_IMPORT_PATH=", "HYPSTER_PARAMS_VERIFIED="]) {
  const matches = notebook.cells.filter((cell) => cell.source.join("").includes(marker));
  if (matches.length !== 1) {
    throw new Error(`expected one shared-fixture cell containing ${marker}, got ${matches.length}`);
  }
  const [cell] = matches;
  if (cell.execution_count !== null || cell.outputs.length !== 0) {
    throw new Error(`shared-fixture cell containing ${marker} must not carry stale execution state`);
  }
}
const copiedNotebooks = fs.readdirSync(root).filter((name) => name.endsWith(".ipynb"));
if (copiedNotebooks.length) {
  throw new Error(`VS Code-only notebook copies are forbidden: ${copiedNotebooks}`);
}

for (const file of [
  "extension.cjs",
  "cli-runner.cjs",
  "creation-gate.cjs",
  "renderer.mjs",
  "renderer-witness.mjs",
  "renderer-gate.cjs",
  "run.cjs",
  "widget-source.cjs",
  "scripts/assert-runtime.cjs",
  "scripts/test-creation-gate.cjs",
  "scripts/test-cli-runner.cjs",
  "scripts/test-renderer-gate.cjs",
  "scripts/test-renderer-witness.mjs",
  "scripts/test-widget-source.cjs",
  "test/index.cjs",
]) {
  const completed = spawnSync(process.execPath, ["--check", path.join(root, file)], {
    encoding: "utf8",
  });
  if (completed.status !== 0) {
    throw new Error(`syntax check failed for ${file}:\n${completed.stderr}`);
  }
}

console.log("VS Code host structure and fail-loud local widget source are valid");
