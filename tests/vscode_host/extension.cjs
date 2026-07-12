"use strict";

const crypto = require("node:crypto");
const vscode = require("vscode");
const {
  ACTIVATION_TIMEOUT_MS,
  RESPONSE_TIMEOUT_MS,
  RendererActivationGate,
} = require("./renderer-gate.cjs");

const RENDERER_ID = "hypster-vscode-host-probe";
const pending = new Map();
const activationGate = new RendererActivationGate();

function editorKey(editor) {
  return editor.notebook.uri.toString();
}

function activate(context) {
  const channel = vscode.notebooks.createRendererMessaging(RENDERER_ID);
  context.subscriptions.push(
    channel.onDidReceiveMessage(({ editor, message }) => {
      if (!message) {
        return;
      }
      if (message.kind === "hypster-probe-ready") {
        activationGate.markReady(editorKey(editor), message.evidence ?? null);
        return;
      }
      if (message.kind !== "hypster-probe-result") {
        return;
      }
      const request = pending.get(message.requestId);
      if (!request) {
        return;
      }
      if (message.ok) {
        request.resolve(message.evidence);
      } else {
        const error = new Error(
          `${message.error}\nRENDERER_DIAGNOSTICS=${JSON.stringify(message.diagnostics ?? null)}`,
        );
        error.rendererDiagnostics = message.diagnostics ?? null;
        request.reject(error);
      }
    }),
    { dispose: () => activationGate.dispose() },
  );

  return Object.freeze({
    async exercise(editor, options = {}) {
      const activationEvidence = await activationGate.wait(
        editorKey(editor),
        ACTIVATION_TIMEOUT_MS,
      );
      const requestId = crypto.randomUUID();
      let timer;
      const response = new Promise((resolve, reject) => {
        timer = setTimeout(() => {
          reject(
            new Error(
              `renderer response timed out after ${RESPONSE_TIMEOUT_MS}ms following ready handshake`,
            ),
          );
        }, RESPONSE_TIMEOUT_MS);
        pending.set(requestId, { resolve, reject });
      });
      response.catch(() => {});

      try {
        const delivered = await channel.postMessage(
          {
            kind: "hypster-probe-exercise",
            requestId,
            creationCellIndex: options.creationCellIndex ?? null,
            creationCellVisible: options.creationCellVisible ?? null,
          },
          editor,
        );
        if (!delivered) {
          throw new Error("renderer exercise was not delivered after ready handshake");
        }
        const evidence = await response;
        return { ...evidence, activation: activationEvidence };
      } finally {
        pending.delete(requestId);
        clearTimeout(timer);
      }
    },
  });
}

function deactivate() {
  for (const [requestId, request] of pending) {
    request.reject(new Error("Hypster host probe deactivated"));
    pending.delete(requestId);
  }
}

module.exports = { activate, deactivate };
