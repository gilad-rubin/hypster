# Interactive Snapshot and Modern Notebook Renderer

Hypster's restored interactive UI will be built around a headless **Interactive Snapshot** controller contract rather than an ipywidgets-specific state model. The public API is `interact(config)`, which returns a live **Interactive Result** handle with `.value` and `.params` rather than returning the raw instantiation value. This avoids the misleading `interactive_explore` name and avoids a separate `interact_params` API, since the handle can expose selected params directly. The first-class UI target remains Jupyter notebooks, but the renderer should use a modern custom-widget approach such as anywidget while staying small, theme-aware, and easy to replace with future Streamlit or React renderers that consume the same snapshot and action contract. Renderers send **Interactive Actions** and render snapshots; Python remains the source of truth for branch reachability, validation, selected-param collection, and replay semantics.

## Notebook renderer direction

The first Jupyter renderer should use the smallest implementation that still feels clear and current:

- Use one anywidget custom widget with file-backed JavaScript and CSS, such as `_esm = Path(...)` and `_css = Path(...)`.
- Use vanilla JavaScript with simple themed controls; do not introduce React, a bundler, generated HTML blobs, a default value preview, or a default params panel.
- Sync only compact `snapshot` and `action` traits between Python and the frontend.
- Keep the raw **Instantiation Value**, full branch history, and **Branch Choice Memory** Python-side in the **Interactive Result** controller. Branch memory is keyed by reachable branch context and parameter metadata, not dotted path alone, because different branches may reuse the same parameter path for different parameter meanings.
- Scope CSS under a Hypster-specific root class because anywidget CSS is loaded globally.
- Treat saved Jupyter widget state as normal notebook behavior; keep synced state tiny so notebooks remain lightweight even when widget state is saved.

The renderer is deliberately shallow. It renders the current **Interactive Snapshot** and emits **Interactive Actions**; the Python controller owns all semantics.
