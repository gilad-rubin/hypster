"use strict";

function classifyMissingCreationMarker({
  selectorCommand,
  creationCommand,
  executionSummary,
  outputCount,
  rawOutput,
}) {
  const runtimeEvidence = [];
  if (selectorCommand.status !== "fulfilled") {
    runtimeEvidence.push(`selector command ${selectorCommand.status}`);
  }
  if (creationCommand.status !== "fulfilled") {
    runtimeEvidence.push(`creation command ${creationCommand.status}`);
  }
  if (executionSummary !== null) {
    runtimeEvidence.push("creation cell has an execution summary");
  }
  if (outputCount !== 0 || rawOutput.length !== 0) {
    runtimeEvidence.push(`creation cell has ${outputCount} output(s)`);
  }
  if (runtimeEvidence.length === 0) {
    runtimeEvidence.push(
      "creation state was still empty at the marker deadline; uncancelled execution may start later",
    );
  }
  return {
    accepted: false,
    outcome: "runtime_failure",
    reason: `Creation marker missing after runtime activity: ${runtimeEvidence.join("; ")}.`,
  };
}

function classifyCommandResult(label, commandResult) {
  if (commandResult.status === "fulfilled") {
    return { ok: true };
  }
  return {
    ok: false,
    outcome: "runtime_failure",
    reason: `${label} command ${commandResult.status}`,
  };
}

module.exports = { classifyCommandResult, classifyMissingCreationMarker };
