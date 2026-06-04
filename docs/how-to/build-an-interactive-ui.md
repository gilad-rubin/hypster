# Build An Interactive UI

Use [`interact()`](../getting-started/interactive-instantiation-ui.md) when you want Hypster's built-in notebook widget. This guide is for custom Streamlit, Gradio, Panel, web app, or internal dashboard UIs.

Use `explore(config, return_schema=True)` to discover fields, render controls in your UI framework, then pass the collected values to `instantiate()`.

`explore()` executes the config function to discover the active branch. A custom UI may call it on every branch-changing edit, so keep config functions cheap and side-effect-free for interactive use; avoid paid API calls, database calls, file writes, training loops, and costly resource initialization in paths that the UI will explore.

## 1. Get A Schema

{% code overflow="wrap" %}
```python
from hypster import HP, explore
from my_app.search import KeywordRetriever, SearchRuntime, VectorRetriever


def keyword_retrieval(hp: HP) -> KeywordRetriever:
    index = hp.text("documents-v1", name="index", description="Keyword index name.")
    top_k = hp.int(20, name="top_k", min=1, max=100)
    return KeywordRetriever(index=index, top_k=top_k)


def vector_retrieval(hp: HP) -> VectorRetriever:
    index = hp.text("embeddings-v1", name="index", description="Vector index name.")
    top_k = hp.int(10, name="top_k", min=1, max=100)
    score_threshold = hp.float(0.2, name="score_threshold", min=0.0, max=1.0)
    return VectorRetriever(index=index, top_k=top_k, score_threshold=score_threshold)


retrieval_options = {
    "keyword": keyword_retrieval,
    "vector": vector_retrieval,
}


def search_config(hp: HP) -> SearchRuntime:
    selected_config = hp.select(
        retrieval_options,
        name="backend",
        default="keyword",
        options_only=True,
        description="Chooses the retrieval branch.",
    )
    retrieval = hp.nest(selected_config, name="retrieval")

    features = hp.multi_select(
        [None, "cache", "trace"],
        name="features",
        default=["cache"],
        allow_none=True,
    )

    return SearchRuntime(retrieval=retrieval, features=features)


schema = explore(search_config, return_schema=True)
metadata = schema.to_dict()
```
{% endcode %}

## 2. Flatten Field Metadata

{% code overflow="wrap" %}
```python
def flatten_fields(parameters):
    for parameter in parameters:
        if parameter["kind"] == "group":
            yield from flatten_fields(parameter["children"])
        else:
            yield parameter

fields = list(flatten_fields(metadata["parameters"]))
```
{% endcode %}

Each field has `path`, `kind`, `default_value`, `selected_value`, optional `options`, optional `minimum`, and optional `maximum`.

Schema metadata is JSON-serializable. After exploring the vector branch with values such as `{"backend": "vector", "features": ["cache", None], "retrieval.index": "embeddings-v3", "retrieval.top_k": 12, "retrieval.score_threshold": 0.35}`, the payload looks like this shape:

{% code overflow="wrap" %}
```python
{
    "name": "search_config",
    "display_label": "Search Config",
    "parameters": [
        {
            "name": "backend",
            "path": "backend",
            "kind": "select",
            "default_value": "keyword",
            "selected_value": "vector",
            "options": ["keyword", "vector"],
            "minimum": None,
            "maximum": None,
            "description": "Chooses the retrieval branch.",
            "display_label": "Backend",
            "children": [],
        },
        {
            "name": "retrieval",
            "path": "retrieval",
            "kind": "group",
            "default_value": None,
            "selected_value": None,
            "options": None,
            "minimum": None,
            "maximum": None,
            "description": None,
            "display_label": "Retrieval",
            "children": [
                {
                    "name": "index",
                    "path": "retrieval.index",
                    "kind": "text",
                    "default_value": "embeddings-v1",
                    "selected_value": "embeddings-v3",
                    "options": None,
                    "minimum": None,
                    "maximum": None,
                    "description": "Vector index name.",
                    "display_label": "Index",
                    "children": [],
                },
                {
                    "name": "top_k",
                    "path": "retrieval.top_k",
                    "kind": "int",
                    "default_value": 10,
                    "selected_value": 12,
                    "options": None,
                    "minimum": 1,
                    "maximum": 100,
                    "description": None,
                    "display_label": "Top K",
                    "children": [],
                },
                {
                    "name": "score_threshold",
                    "path": "retrieval.score_threshold",
                    "kind": "float",
                    "default_value": 0.2,
                    "selected_value": 0.35,
                    "options": None,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": None,
                    "display_label": "Score Threshold",
                    "children": [],
                },
            ],
        },
        {
            "name": "features",
            "path": "features",
            "kind": "multi_select",
            "default_value": ["cache"],
            "selected_value": ["cache", None],
            "options": [None, "cache", "trace"],
            "minimum": None,
            "maximum": None,
            "description": None,
            "display_label": "Features",
            "children": [],
        },
    ],
}
```
{% endcode %}

For dict-backed selects, `options` contains the replayable keys, not the mapped runtime objects.

This is another reason to prefer named option dictionaries for swappable components: UIs can render simple stable keys such as `"keyword"` and `"vector"`, while the config still receives the mapped class or child config function.

## 3. Render Controls

Map field kinds to controls in your UI framework:

{% code overflow="wrap" %}
```python
def control_spec(field):
    if field["kind"] == "select":
        return {"widget": "dropdown", "options": field["options"], "value": field["selected_value"]}
    if field["kind"] == "bool":
        return {"widget": "checkbox", "value": field["selected_value"]}
    if field["kind"] in {"int", "float"}:
        return {
            "widget": "number",
            "value": field["selected_value"],
            "min": field["minimum"],
            "max": field["maximum"],
        }
    if field["kind"] == "text":
        return {"widget": "text", "value": field["selected_value"]}
    if field["kind"] in {"multi_int", "multi_float", "multi_text", "multi_bool", "multi_select"}:
        return {"widget": "list", "value": field["selected_value"], "options": field["options"]}
    raise ValueError(f"Unsupported kind: {field['kind']}")

controls = {field["path"]: control_spec(field) for field in fields}
```
{% endcode %}

## 4. Recompute Conditional Branches

When a branch-selecting field changes, call `explore()` again with current UI values:

{% code overflow="wrap" %}
```python
ui_values = {"backend": "vector"}
schema = explore(search_config, values=ui_values, return_schema=True)
fields = list(flatten_fields(schema.to_dict()["parameters"]))

assert [field["path"] for field in fields] == [
    "backend",
    "retrieval.index",
    "retrieval.top_k",
    "retrieval.score_threshold",
    "features",
]
```
{% endcode %}

Custom UIs should submit only paths present in the latest schema. A robust branch-change loop is:

{% code overflow="wrap" %}
```python
def reachable_paths(schema):
    return {field["path"] for field in flatten_fields(schema.to_dict()["parameters"])}

def refresh_schema(config, current_values):
    schema = explore(config, values=current_values, on_unknown="ignore", return_schema=True)
    reachable = reachable_paths(schema)
    pruned_values = {path: value for path, value in current_values.items() if path in reachable}
    schema = explore(config, values=pruned_values, return_schema=True)
    return schema, pruned_values
```
{% endcode %}

If your UI remembers draft values per branch, keep that memory outside the submitted `values=` dictionary. Before calling `instantiate()`, remove stale paths from inactive branches.

Use `on_unknown="ignore"` only for this schema-refresh pruning pass. For the final submit, use the default `on_unknown="raise"` so typos and stale inactive paths are surfaced.

## 5. Instantiate From UI State

{% code overflow="wrap" %}
```python
from hypster import instantiate

ui_values = {
    "backend": "vector",
    "features": ["cache", "trace"],
    "retrieval.index": "embeddings-v3",
    "retrieval.top_k": 12,
    "retrieval.score_threshold": 0.35,
}

cfg = instantiate(search_config, values=ui_values)
assert isinstance(cfg.retrieval, VectorRetriever)
assert cfg.retrieval.top_k == 12
```
{% endcode %}

## 6. Submit And Show Errors

{% code overflow="wrap" %}
```python
from hypster import instantiate

try:
    cfg = instantiate(search_config, values=ui_values)
except ValueError as exc:
    show_form_error(str(exc))
else:
    run_search(cfg)
```
{% endcode %}

If you intentionally allow old UI payloads while users are editing, use `on_unknown="warn"` and capture warnings near the form. Keep final run submissions strict unless you have a migration path for ignored fields.

{% hint style="warning" %}
Do not send stale fields from inactive branches. If the user switches from `vector` back to `keyword`, remove `retrieval.score_threshold` from the submitted values or call `instantiate(..., on_unknown="ignore")` only when you intentionally want softer handling.
{% endhint %}
