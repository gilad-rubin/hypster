"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {
  ANYWIDGET_ROUTE,
  anywidgetAssetPaths,
  startLocalWidgetSource,
  widgetSourceTemplate,
} = require("../widget-source.cjs");

const extensionSource = `define(function () {
  "use strict";
  window.requirejs?.config({
    map: { "*": { anywidget: "nbextensions/anywidget/index" } },
  });
});\n`;
const indexSource = "define('anywidget', [], function () { return {}; });\n";
const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "hypster-widget-source-"));
const pythonExecutable = path.join(temporaryRoot, "kernel", "bin", "python");
const assets = anywidgetAssetPaths(pythonExecutable);
fs.mkdirSync(path.dirname(pythonExecutable), { recursive: true });
fs.writeFileSync(pythonExecutable, "#!/bin/sh\n");
fs.mkdirSync(assets.nbextensionRoot, { recursive: true });
fs.writeFileSync(assets.extensionPath, extensionSource);
fs.writeFileSync(assets.indexPath, indexSource);

async function main() {
  const source = await startLocalWidgetSource(pythonExecutable);
  try {
    assert.equal(
      source.template,
      `http://127.0.0.1:${source.evidence.port}/\${packageName}/\${fileNameWithExt}`,
    );
    assert.equal(widgetSourceTemplate(source.evidence.port), source.template);
    assert.equal(source.evidence.asset.kernelPrefix, path.join(temporaryRoot, "kernel"));
    assert.equal(
      source.evidence.asset.indexSha256,
      crypto.createHash("sha256").update(indexSource).digest("hex"),
    );

    const headResponse = await fetch(`http://127.0.0.1:${source.evidence.port}${ANYWIDGET_ROUTE}`, {
      method: "HEAD",
    });
    assert.equal(headResponse.status, 200);
    const extensionHostResponse = await fetch(
      `http://127.0.0.1:${source.evidence.port}${ANYWIDGET_ROUTE}`,
    );
    assert.equal(extensionHostResponse.status, 200);
    assert.throws(
      () => source.assertUsed(),
      /Electron webview did not fetch the exact local anywidget bundle/,
    );
    const getResponse = await fetch(
      `http://127.0.0.1:${source.evidence.port}${ANYWIDGET_ROUTE}`,
      { headers: { "User-Agent": "Code/1.128.0 Electron/42.5.0" } },
    );
    assert.equal(getResponse.status, 200);
    assert.equal(await getResponse.text(), indexSource);
    const rejectedResponse = await fetch(
      `http://127.0.0.1:${source.evidence.port}/unexpected.js`,
    );
    assert.equal(rejectedResponse.status, 404);
    source.assertUsed();
    assert.equal(source.evidence.successfulIndexGetVerified, true);
  } finally {
    await source.close();
  }

  fs.writeFileSync(assets.extensionPath, "define(function () {});\n");
  await assert.rejects(
    startLocalWidgetSource(pythonExecutable),
    /not compatible with Jupyter 2025\.9\.1's local widget-source parser/,
  );
  console.log("local anywidget source path, hash, route, and fail-loud parser gate passed");
}

main()
  .finally(() => fs.rmSync(temporaryRoot, { recursive: true, force: true }))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
