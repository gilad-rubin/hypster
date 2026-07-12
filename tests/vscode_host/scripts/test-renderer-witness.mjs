import assert from "node:assert/strict";
import {
  detectWidgetRootTransition,
  findPublishedRemoteState,
} from "../renderer-witness.mjs";

const numericSelector = "input[data-path='remote.temperature'][data-kind='float']";
const modeSelector = ".hypster-choice[data-path='mode'] .hypster-choice-value";

const menuClosingRoot = {
  innerHTML: "menu-closed-before-backend-publish",
  isConnected: true,
  querySelector(selector) {
    return selector === modeSelector ? { textContent: "local" } : undefined;
  },
};
assert.deepEqual(
  detectWidgetRootTransition(menuClosingRoot, "menu-open", menuClosingRoot),
  { kind: "updated-in-place", currentRoot: menuClosingRoot },
  "menu close reproduces the premature HTML-only transition",
);
assert.equal(
  findPublishedRemoteState({ querySelector: () => undefined }),
  undefined,
  "menu close alone is not published remote state",
);

const publishedRoot = {
  innerHTML: "remote-published",
  isConnected: true,
  querySelector(selector) {
    return selector === modeSelector ? { textContent: "remote" } : undefined;
  },
};
const publishedNumeric = {
  isConnected: true,
  closest(selector) {
    return selector === ".hypster-widget" ? publishedRoot : undefined;
  },
};
const publishedState = findPublishedRemoteState({
  querySelector(selector) {
    return selector === numericSelector ? publishedNumeric : undefined;
  },
});
assert.deepEqual(publishedState, { currentRoot: publishedRoot, numeric: publishedNumeric });
menuClosingRoot.isConnected = false;
assert.deepEqual(
  detectWidgetRootTransition(menuClosingRoot, "menu-open", publishedState.currentRoot),
  { kind: "replaced", currentRoot: publishedRoot },
);

const previousRoot = { innerHTML: "before", isConnected: false };
const replacementRoot = { innerHTML: "after", isConnected: true };
assert.deepEqual(
  detectWidgetRootTransition(previousRoot, "before", replacementRoot),
  { kind: "replaced", currentRoot: replacementRoot },
);

previousRoot.isConnected = true;
assert.equal(detectWidgetRootTransition(previousRoot, "before", replacementRoot), undefined);

previousRoot.innerHTML = "after";
assert.deepEqual(
  detectWidgetRootTransition(previousRoot, "before", previousRoot),
  { kind: "updated-in-place", currentRoot: previousRoot },
);

previousRoot.innerHTML = "before";
assert.equal(detectWidgetRootTransition(previousRoot, "before", previousRoot), undefined);
assert.equal(
  detectWidgetRootTransition(previousRoot, "before", { innerHTML: "after", isConnected: false }),
  undefined,
);

console.log("semantic publication precedes connected renderer root transition");
