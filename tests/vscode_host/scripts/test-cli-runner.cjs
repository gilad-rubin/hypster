"use strict";

const assert = require("node:assert/strict");
const { CLI_TIMEOUT_MS, runCLI } = require("../cli-runner.cjs");

let invocation;
const output = runCLI(
  "/vscode/bin/code",
  ["--base"],
  ["--install-extension", "publisher.extension@1.2.3"],
  { PATH: "/bin" },
  (...args) => {
    invocation = args;
    return { error: undefined, status: 0, stdout: "installed\n", stderr: "warning\n" };
  },
);
assert.equal(CLI_TIMEOUT_MS, 120_000);
assert.deepEqual(invocation.slice(0, 2), [
  "/vscode/bin/code",
  ["--base", "--install-extension", "publisher.extension@1.2.3"],
]);
assert.equal(invocation[2].timeout, CLI_TIMEOUT_MS);
assert.equal(output, "installed\nwarning");

assert.throws(
  () =>
    runCLI("code", [], ["--list-extensions"], {}, () => ({
      error: Object.assign(new Error("spawnSync code ETIMEDOUT"), { code: "ETIMEDOUT" }),
      status: null,
      stdout: "partial stdout",
      stderr: "partial stderr",
    })),
  (error) =>
    error.message.includes("VS Code CLI failed (null)") &&
    error.message.includes("partial stdout") &&
    error.message.includes("partial stderr") &&
    error.message.includes("ETIMEDOUT"),
);

console.log("bounded VS Code CLI runner and error evidence passed");
