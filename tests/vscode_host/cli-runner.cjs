"use strict";

const { spawnSync } = require("node:child_process");

const CLI_TIMEOUT_MS = 120_000;

function runCLI(cliPath, baseArgs, args, environment, spawn = spawnSync) {
  const completed = spawn(cliPath, [...baseArgs, ...args], {
    encoding: "utf8",
    env: environment,
    shell: process.platform === "win32",
    timeout: CLI_TIMEOUT_MS,
  });
  if (completed.error || completed.status !== 0) {
    throw new Error(
      `VS Code CLI failed (${completed.status}): ${JSON.stringify(args)}\n` +
        `stdout:\n${completed.stdout}\nstderr:\n${completed.stderr}\n${completed.error ?? ""}`,
    );
  }
  return `${completed.stdout}${completed.stderr}`.trim();
}

module.exports = { CLI_TIMEOUT_MS, runCLI };
