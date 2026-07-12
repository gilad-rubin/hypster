"use strict";

const crypto = require("node:crypto");
const vscode = require("vscode");

const RENDERER_ID = "hypster-vscode-host-probe";
const pending = new Map();

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function activate(context) {
  const channel = vscode.notebooks.createRendererMessaging(RENDERER_ID);
  context.subscriptions.push(
    channel.onDidReceiveMessage(({ message }) => {
      if (!message || message.kind !== "hypster-probe-result") {
        return;
      }
      const request = pending.get(message.requestId);
      if (!request) {
        return;
      }
      if (message.ok) {
        request.resolve(message.evidence);
      } else {
        request.reject(new Error(message.error));
      }
    }),
  );

  return Object.freeze({
    async exercise(editor, timeoutMilliseconds = 30_000) {
      const requestId = crypto.randomUUID();
      const deadline = Date.now() + timeoutMilliseconds;
      let phase = "delivery";
      let timer;
      const response = new Promise((resolve, reject) => {
        timer = setTimeout(() => {
          reject(new Error(`renderer ${phase} timed out after ${timeoutMilliseconds}ms`));
        }, timeoutMilliseconds);
        pending.set(requestId, { resolve, reject });
      });
      response.catch(() => {});

      try {
        let delivered = false;
        while (!delivered && Date.now() < deadline) {
          const attempt = await Promise.race([
            channel
              .postMessage({ kind: "hypster-probe-exercise", requestId }, editor)
              .then((value) => ({ kind: "delivery", delivered: value })),
            response.then((evidence) => ({ kind: "response", evidence })),
          ]);
          if (attempt.kind === "response") {
            return attempt.evidence;
          }
          delivered = attempt.delivered;
          if (!delivered) {
            await delay(Math.min(250, Math.max(0, deadline - Date.now())));
          }
        }
        if (!delivered) {
          return await response;
        }
        phase = "response";
        return await response;
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
