"use strict";

const OUTER_TIMEOUT_MS = 30_000;
const ACTIVATION_TIMEOUT_MS = 5_000;
const RENDERER_EXERCISE_TIMEOUT_MS = 20_000;
const RESPONSE_TIMEOUT_MS = RENDERER_EXERCISE_TIMEOUT_MS + 2_000;

if (ACTIVATION_TIMEOUT_MS + RESPONSE_TIMEOUT_MS >= OUTER_TIMEOUT_MS) {
  throw new Error("renderer gate stages must finish before the outer timeout");
}

class RendererActivationGate {
  #ready = new Map();
  #waiters = new Map();
  #disposed = false;

  markReady(key, evidence) {
    if (this.#disposed) {
      return;
    }
    this.#ready.set(key, evidence);
    const waiters = this.#waiters.get(key) ?? [];
    this.#waiters.delete(key);
    for (const waiter of waiters) {
      clearTimeout(waiter.timer);
      waiter.resolve(evidence);
    }
  }

  wait(key, timeoutMilliseconds = ACTIVATION_TIMEOUT_MS) {
    if (this.#disposed) {
      return Promise.reject(new Error("renderer activation gate is disposed"));
    }
    if (this.#ready.has(key)) {
      return Promise.resolve(this.#ready.get(key));
    }
    return new Promise((resolve, reject) => {
      const waiter = {
        resolve,
        reject,
        timer: setTimeout(() => {
          const waiters = this.#waiters.get(key) ?? [];
          const remaining = waiters.filter((candidate) => candidate !== waiter);
          if (remaining.length) {
            this.#waiters.set(key, remaining);
          } else {
            this.#waiters.delete(key);
          }
          reject(
            new Error(
              `renderer activation handshake timed out after ${timeoutMilliseconds}ms for ${key}`,
            ),
          );
        }, timeoutMilliseconds),
      };
      const waiters = this.#waiters.get(key) ?? [];
      waiters.push(waiter);
      this.#waiters.set(key, waiters);
    });
  }

  dispose() {
    this.#disposed = true;
    for (const waiters of this.#waiters.values()) {
      for (const waiter of waiters) {
        clearTimeout(waiter.timer);
        waiter.reject(new Error("renderer activation gate disposed before ready"));
      }
    }
    this.#waiters.clear();
    this.#ready.clear();
  }
}

module.exports = {
  ACTIVATION_TIMEOUT_MS,
  OUTER_TIMEOUT_MS,
  RENDERER_EXERCISE_TIMEOUT_MS,
  RESPONSE_TIMEOUT_MS,
  RendererActivationGate,
};
