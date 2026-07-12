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
if (manifest.packageManager !== `npm@${pins.npm}` || manifest.engines.node !== pins.node) {
  throw new Error("Node.js and npm must remain exactly pinned");
}
if (pins.vscode !== "1.128.0") {
  throw new Error("VS Code Desktop must remain exactly pinned");
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
  "creation-gate.cjs",
  "renderer.mjs",
  "run.cjs",
  "scripts/assert-runtime.cjs",
  "scripts/test-creation-gate.cjs",
  "test/index.cjs",
]) {
  const completed = spawnSync(process.execPath, ["--check", path.join(root, file)], {
    encoding: "utf8",
  });
  if (completed.status !== 0) {
    throw new Error(`syntax check failed for ${file}:\n${completed.stderr}`);
  }
}

console.log("VS Code host spike structure is valid");
