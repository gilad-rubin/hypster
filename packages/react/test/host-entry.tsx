import React, { useLayoutEffect } from "react";
import { createRoot } from "react-dom/client";

import {
  HypsterRenderer,
  type InteractiveAction,
  type InteractiveSnapshot,
} from "../dist/index.js";

declare global {
  interface Window {
    __hypsterHost: { executionId: string; sequence: number; renderer: string };
  }
}

type BridgeSnapshot = InteractiveSnapshot & {
  bridge_execution_id: string;
  bridge_sequence: number;
};

const rootElement = document.querySelector<HTMLElement>("#root");
if (!rootElement) throw new Error("Missing #root host element");
const root = createRoot(rootElement);
let current: BridgeSnapshot;

function HostRenderer({ snapshot }: { snapshot: BridgeSnapshot }) {
  useLayoutEffect(() => {
    rootElement.dataset.renderedSequence = String(snapshot.bridge_sequence);
  }, [snapshot.bridge_sequence]);
  return <HypsterRenderer snapshot={snapshot} onAction={onAction} />;
}

function render(snapshot: BridgeSnapshot) {
  current = snapshot;
  window.__hypsterHost = {
    executionId: snapshot.bridge_execution_id,
    sequence: snapshot.bridge_sequence,
    renderer: "@hypster/react",
  };
  rootElement.dataset.renderer = "@hypster/react";
  rootElement.dataset.executionId = snapshot.bridge_execution_id;
  rootElement.dataset.sequence = String(snapshot.bridge_sequence);
  root.render(<HostRenderer snapshot={snapshot} />);
}

async function onAction(action: InteractiveAction) {
  const response = await fetch("/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ execution_id: current.bridge_execution_id, action }),
  });
  if (!response.ok) throw new Error(`Action failed: ${response.status}`);
  render((await response.json()) as BridgeSnapshot);
}

const v2 = new URLSearchParams(location.search).has("v2");
const response = await fetch(v2 ? "/snapshot?v2=1" : "/snapshot");
if (!response.ok) throw new Error(`Snapshot failed: ${response.status}`);
render((await response.json()) as BridgeSnapshot);
