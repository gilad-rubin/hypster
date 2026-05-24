# Data Processing

Data pipelines often mix environment choices, schema decisions, cleaning rules, and export settings. Hypster keeps those choices explicit and replayable.

## Configure The Pipeline

{% code overflow="wrap" %}
```python
from dataclasses import dataclass
from hypster import HP, explore, instantiate

@dataclass
class CsvInput:
    path: str
    delimiter: str
    encoding: str

@dataclass
class CleaningRules:
    drop_empty_rows: bool
    normalize_columns: bool
    fill_missing_numeric: float | None
    date_columns: list[str]

@dataclass
class ExportConfig:
    format: str
    path: str
    include_header: bool

def input_config(hp: HP) -> CsvInput:
    return CsvInput(
        path=hp.text("data/raw/events.csv", name="path"),
        delimiter=hp.select([",", "\t", "|"], name="delimiter", default=",", options_only=True),
        encoding=hp.select(["utf-8", "latin-1"], name="encoding", default="utf-8", options_only=True),
    )

def cleaning_config(hp: HP) -> CleaningRules:
    return CleaningRules(
        drop_empty_rows=hp.bool(True, name="drop_empty_rows"),
        normalize_columns=hp.bool(True, name="normalize_columns"),
        fill_missing_numeric=hp.float(None, name="fill_missing_numeric", allow_none=True),
        date_columns=hp.multi_text(["created_at"], name="date_columns"),
    )

def export_config(hp: HP) -> ExportConfig:
    return ExportConfig(
        format=hp.select(["parquet", "csv", "jsonl"], name="format", default="parquet", options_only=True),
        path=hp.text("data/processed/events.parquet", name="path"),
        include_header=hp.bool(True, name="include_header"),
    )

def data_pipeline_config(hp: HP):
    mode = hp.select(["sample", "full"], name="mode", default="sample", options_only=True)

    if mode == "full":
        row_limit = hp.int(10_000_000, name="row_limit", min=1)
    else:
        row_limit = hp.int(10_000, name="row_limit", min=1)

    return {
        "mode": mode,
        "row_limit": row_limit,
        "input": hp.nest(input_config, name="input"),
        "cleaning": hp.nest(cleaning_config, name="cleaning"),
        "export": hp.nest(export_config, name="export"),
    }
```
{% endcode %}

## Explore A Production Branch

{% code overflow="wrap" %}
```python
explore(
    data_pipeline_config,
    values={
        "mode": "full",
        "input.path": "s3://warehouse/events/2026-05-24.csv",
        "export.format": "jsonl",
    },
)
```
{% endcode %}

## Instantiate A Run

{% code overflow="wrap" %}
```python
cfg = instantiate(
    data_pipeline_config,
    values={
        "mode": "full",
        "input": {
            "path": "s3://warehouse/events/2026-05-24.csv",
            "delimiter": ",",
        },
        "cleaning.fill_missing_numeric": 0.0,
        "export.format": "jsonl",
        "export.path": "s3://warehouse/processed/events.jsonl",
    },
)

assert cfg["mode"] == "full"
assert cfg["input"].path.startswith("s3://")
assert cfg["cleaning"].fill_missing_numeric == 0.0
```
{% endcode %}

## Why This Shape Works

* The `mode` branch changes the default row limit while keeping the same parameter path.
* Dotted keys and nested dictionaries can both express nested values.
* `options_only=True` prevents typos in finite choices such as export formats.
* Nullable numeric values use `allow_none=True`, which makes `None` an explicit, replayable value.
