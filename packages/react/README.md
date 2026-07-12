# `@hypster/react`

`@hypster/react` is an experimental React renderer for Hypster interactive
configuration snapshots. It is a presentation adapter: Python owns exploration,
validation, branch memory, application, and the selected parameters.

```tsx
import { HypsterRenderer } from "@hypster/react";

<HypsterRenderer
  snapshot={snapshotFromPython}
  onAction={(action) => sendActionToPython(action)}
/>
```

The host sends each emitted action to the Python `InteractiveSession`, replaces
the old snapshot with the response, and uses `snapshot.selected_params` as the
authoritative applied configuration. React must not reconcile values or rebuild
branch semantics locally.

The renderer currently recognizes Interactive Protocol V1. A different version
produces a visible mismatch and no controls, which detects an incompatible host;
V1 is not a stability promise for a future 1.x release.

This package is experimental and private to the repository. Publishing it to npm
is explicitly out of scope.
