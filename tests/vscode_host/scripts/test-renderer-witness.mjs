import assert from "node:assert/strict";
import { detectWidgetRootTransition } from "../renderer-witness.mjs";

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

console.log("connected renderer root transition witness passed");
