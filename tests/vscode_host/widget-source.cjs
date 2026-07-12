"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const pins = require("./pins.cjs");

const ANYWIDGET_ROUTE = "/anywidget/index.js";

function sha256(contents) {
  return crypto.createHash("sha256").update(contents).digest("hex");
}

function anywidgetAssetPaths(pythonExecutable) {
  const kernelPrefix = path.dirname(path.dirname(pythonExecutable));
  const nbextensionRoot = path.join(
    kernelPrefix,
    "share",
    "jupyter",
    "nbextensions",
    "anywidget",
  );
  return {
    kernelPrefix,
    nbextensionRoot,
    extensionPath: path.join(nbextensionRoot, "extension.js"),
    indexPath: path.join(nbextensionRoot, "index.js"),
  };
}

function readRegularFile(filePath, label) {
  let stat;
  try {
    stat = fs.statSync(filePath);
  } catch (error) {
    throw new Error(`${label} is missing at ${filePath}: ${error.message}`);
  }
  if (!stat.isFile()) {
    throw new Error(`${label} is not a regular file: ${filePath}`);
  }
  return fs.readFileSync(filePath);
}

function inspectAnywidgetAssets(pythonExecutable) {
  const pythonStat = fs.statSync(pythonExecutable);
  if (!pythonStat.isFile()) {
    throw new Error(`selected kernel Python is not a regular file: ${pythonExecutable}`);
  }

  const paths = anywidgetAssetPaths(pythonExecutable);
  const extension = readRegularFile(paths.extensionPath, "anywidget extension.js");
  const index = readRegularFile(paths.indexPath, "anywidget index.js");
  const extensionSource = extension.toString("utf8");
  const requireJsConfigPresent = extensionSource.includes("window.requirejs?.config({");
  const anywidgetMappingPresent =
    /anywidget\s*:\s*["']nbextensions\/anywidget\/index["']/.test(extensionSource);
  if (!requireJsConfigPresent || !anywidgetMappingPresent) {
    throw new Error(
      `anywidget extension.js at ${paths.extensionPath} is not compatible with ` +
        "Jupyter 2025.9.1's local widget-source parser",
    );
  }
  if (index.length === 0) {
    throw new Error(`anywidget index.js is empty: ${paths.indexPath}`);
  }

  return {
    paths,
    extension,
    index,
    evidence: {
      ...paths,
      extensionBytes: extension.length,
      extensionSha256: sha256(extension),
      indexBytes: index.length,
      indexSha256: sha256(index),
      requireJsConfigPresent,
      anywidgetMapping: "anywidget -> nbextensions/anywidget/index",
    },
  };
}

function widgetSourceTemplate(port) {
  if (!Number.isInteger(port) || port <= 0 || port > 65_535) {
    throw new Error(`invalid local widget-source port: ${port}`);
  }
  return `http://127.0.0.1:${port}${pins.widgetScriptSourcePathTemplate}`;
}

async function startLocalWidgetSource(pythonExecutable) {
  const assets = inspectAnywidgetAssets(pythonExecutable);
  const requests = [];
  const server = http.createServer((request, response) => {
    const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
    const methodAllowed = request.method === "GET" || request.method === "HEAD";
    const routeAllowed = requestUrl.pathname === ANYWIDGET_ROUTE && requestUrl.search === "";
    const status = methodAllowed && routeAllowed ? 200 : 404;
    requests.push({
      method: request.method ?? null,
      path: `${requestUrl.pathname}${requestUrl.search}`,
      status,
      origin: request.headers.origin ?? null,
      userAgent: request.headers["user-agent"] ?? null,
    });

    response.writeHead(status, {
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "no-store",
      "Content-Length": status === 200 ? assets.index.length : 0,
      "Content-Type": "text/javascript; charset=utf-8",
    });
    if (status === 200 && request.method === "GET") {
      response.end(assets.index);
    } else {
      response.end();
    }
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error(`local widget source returned an invalid address: ${address}`);
  }
  server.unref();

  const evidence = {
    transport: "loopback-http",
    host: "127.0.0.1",
    port: address.port,
    route: ANYWIDGET_ROUTE,
    template: widgetSourceTemplate(address.port),
    asset: assets.evidence,
    requests,
    successfulIndexGetVerified: false,
  };

  return {
    template: evidence.template,
    evidence,
    assertUsed() {
      const used = requests.some(
        (request) =>
          request.method === "GET" &&
          request.path === ANYWIDGET_ROUTE &&
          request.status === 200 &&
          request.userAgent?.includes("Code/") &&
          request.userAgent?.includes("Electron/"),
      );
      evidence.successfulIndexGetVerified = used;
      if (!used) {
        throw new Error(
          `Electron webview did not fetch the exact local anywidget bundle at ${ANYWIDGET_ROUTE}; ` +
            `requests=${JSON.stringify(requests)}`,
        );
      }
    },
    async close() {
      await new Promise((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
        server.closeAllConnections?.();
      });
    },
  };
}

module.exports = {
  ANYWIDGET_ROUTE,
  anywidgetAssetPaths,
  inspectAnywidgetAssets,
  startLocalWidgetSource,
  widgetSourceTemplate,
};
