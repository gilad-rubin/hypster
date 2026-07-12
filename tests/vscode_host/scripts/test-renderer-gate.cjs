"use strict";

const assert = require("node:assert/strict");
const {
  ACTIVATION_TIMEOUT_MS,
  OUTER_TIMEOUT_MS,
  RENDERER_EXERCISE_TIMEOUT_MS,
  RESPONSE_TIMEOUT_MS,
  RendererActivationGate,
} = require("../renderer-gate.cjs");

async function main() {
  assert.ok(ACTIVATION_TIMEOUT_MS + RESPONSE_TIMEOUT_MS < OUTER_TIMEOUT_MS);
  assert.ok(RENDERER_EXERCISE_TIMEOUT_MS < RESPONSE_TIMEOUT_MS);

  const earlyReady = new RendererActivationGate();
  earlyReady.markReady("notebook-1", { phase: "activate" });
  assert.deepEqual(await earlyReady.wait("notebook-1"), { phase: "activate" });
  earlyReady.dispose();

  const lateReady = new RendererActivationGate();
  const waiting = lateReady.wait("notebook-2", 100);
  lateReady.markReady("notebook-2", { phase: "activate-after-wait" });
  assert.deepEqual(await waiting, { phase: "activate-after-wait" });
  lateReady.dispose();

  const missingReady = new RendererActivationGate();
  await assert.rejects(
    missingReady.wait("notebook-3", 10),
    /renderer activation handshake timed out after 10ms for notebook-3/,
  );
  missingReady.dispose();

  console.log("renderer ready handshake rejects the pre-activation message race before outer timeout");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
