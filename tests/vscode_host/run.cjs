"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const {
  downloadAndUnzipVSCode,
  resolveCliArgsFromVSCodeExecutablePath,
  runTests,
} = require("@vscode/test-electron");
const pins = require("./pins.cjs");

const HOST_ENVIRONMENT_KEYS = [
  "CI",
  "DBUS_SESSION_BUS_ADDRESS",
  "DISPLAY",
  "HOME",
  "LANG",
  "LC_ALL",
  "LD_LIBRARY_PATH",
  "LOGNAME",
  "NODE_EXTRA_CA_CERTS",
  "NO_PROXY",
  "PATH",
  "PWD",
  "SHELL",
  "SSL_CERT_FILE",
  "TEMP",
  "TERM",
  "TMP",
  "TMPDIR",
  "TZ",
  "USER",
  "XAUTHORITY",
  "XDG_CACHE_HOME",
  "XDG_CONFIG_HOME",
  "XDG_DATA_HOME",
  "XDG_RUNTIME_DIR",
  "http_proxy",
  "https_proxy",
  "no_proxy",
  "HTTP_PROXY",
  "HTTPS_PROXY",
];

function requiredPath(name) {
  const value = process.env[name];
  if (!value || !path.isAbsolute(value)) {
    throw new Error(`${name} must be an explicit absolute path`);
  }
  return value;
}

function minimalEnvironment(extra = {}) {
  const environment = {};
  for (const name of HOST_ENVIRONMENT_KEYS) {
    if (process.env[name] !== undefined) {
      environment[name] = process.env[name];
    }
  }
  return { ...environment, ...extra };
}

async function runWithEnvironment(environment, operation) {
  // @vscode/test-electron merges process.env into the Electron child. Snapshot
  // only so it can be restored; the child sees the explicit allowlist below.
  const original = { ...process.env };
  for (const name of Object.keys(process.env)) {
    delete process.env[name];
  }
  Object.assign(process.env, environment);
  try {
    return await operation();
  } finally {
    for (const name of Object.keys(process.env)) {
      delete process.env[name];
    }
    Object.assign(process.env, original);
  }
}

function runCLI(cliPath, baseArgs, args, environment) {
  const completed = spawnSync(cliPath, [...baseArgs, ...args], {
    encoding: "utf8",
    env: environment,
    shell: process.platform === "win32",
  });
  if (completed.error || completed.status !== 0) {
    throw new Error(
      `VS Code CLI failed (${completed.status}): ${JSON.stringify(args)}\n` +
        `stdout:\n${completed.stdout}\nstderr:\n${completed.stderr}\n${completed.error ?? ""}`,
    );
  }
  return `${completed.stdout}${completed.stderr}`.trim();
}

async function main() {
  const harnessRoot = __dirname;
  const artifactDir = requiredPath("HYPSTER_VSCODE_ARTIFACT_DIR");
  const runtimeDir = requiredPath("HYPSTER_VSCODE_RUNTIME_DIR");
  const notebook = requiredPath("HYPSTER_NOTEBOOK");
  const jupyterPath = requiredPath("JUPYTER_PATH");
  const userDataDir = path.join(runtimeDir, "user-data");
  const extensionsDir = path.join(runtimeDir, "extensions");
  fs.mkdirSync(artifactDir, { recursive: true });
  fs.mkdirSync(userDataDir, { recursive: true });
  fs.mkdirSync(extensionsDir, { recursive: true });

  const executable = await downloadAndUnzipVSCode({
    version: pins.vscode,
    cachePath: path.join(runtimeDir, "download"),
  });
  const [cliPath, ...cliArgs] = resolveCliArgsFromVSCodeExecutablePath(executable, {
    reuseMachineInstall: true,
  });
  const environment = minimalEnvironment({
    HYPSTER_NOTEBOOK: notebook,
    HYPSTER_VSCODE_ARTIFACT_DIR: artifactDir,
    HYPSTER_VSCODE_RUNTIME_DIR: runtimeDir,
    JUPYTER_PATH: jupyterPath,
  });
  fs.writeFileSync(
    path.join(artifactDir, "electron-environment-keys.json"),
    `${JSON.stringify(Object.keys(environment).sort(), null, 2)}\n`,
  );
  const profileArgs = ["--user-data-dir", userDataDir, "--extensions-dir", extensionsDir];

  const installLog = [];
  for (const [extension, version] of Object.entries(pins.extensions)) {
    installLog.push(
      runCLI(
        cliPath,
        cliArgs,
        [
          ...profileArgs,
          "--install-extension",
          `${extension}@${version}`,
          "--force",
          "--do-not-include-pack-dependencies",
        ],
        environment,
      ),
    );
  }
  fs.writeFileSync(path.join(artifactDir, "extension-install.log"), `${installLog.join("\n")}\n`);
  const installed = runCLI(
    cliPath,
    cliArgs,
    [...profileArgs, "--list-extensions", "--show-versions"],
    environment,
  );
  fs.writeFileSync(path.join(artifactDir, "installed-extensions.txt"), `${installed}\n`);
  for (const [extension, version] of Object.entries(pins.extensions)) {
    if (!installed.split(/\r?\n/).includes(`${extension}@${version}`)) {
      throw new Error(`exact extension pin missing after installation: ${extension}@${version}`);
    }
  }

  const exitCode = await runWithEnvironment(environment, () =>
    runTests({
      vscodeExecutablePath: executable,
      reuseMachineInstall: true,
      extensionDevelopmentPath: harnessRoot,
      extensionTestsPath: path.join(harnessRoot, "test", "index.cjs"),
      extensionTestsEnv: environment,
      launchArgs: [
        path.dirname(notebook),
        ...profileArgs,
        "--disable-workspace-trust",
        "--disable-updates",
        "--skip-release-notes",
        "--skip-welcome",
      ],
    }),
  );
  if (exitCode !== 0) {
    throw new Error(`VS Code extension test exited with ${exitCode}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
