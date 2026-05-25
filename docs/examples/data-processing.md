# Data Processing

Data pipelines often mix environment choices, schema decisions, cleaning rules, and export settings. Hypster keeps those choices explicit and replayable.

## Configure The Pipeline

{% code overflow="wrap" %}
```python
from hypster import HP, explore, instantiate
from my_app.data import Cleaner, CsvReader, DataPipeline
from my_app.exporters import CsvExporter, JsonLinesExporter, ParquetExporter

def input_config(hp: HP) -> CsvReader:
    path = hp.text("data/raw/events.csv", name="path")
    delimiter = hp.select([",", "\t", "|"], name="delimiter", default=",", options_only=True)
    encoding = hp.select(["utf-8", "latin-1"], name="encoding", default="utf-8", options_only=True)

    return CsvReader(
        path=path,
        delimiter=delimiter,
        encoding=encoding,
    )

def cleaning_config(hp: HP) -> Cleaner:
    drop_empty_rows = hp.bool(True, name="drop_empty_rows")
    normalize_columns = hp.bool(True, name="normalize_columns")
    fill_missing_numeric = hp.float(None, name="fill_missing_numeric", allow_none=True)
    date_columns = hp.multi_text(["created_at"], name="date_columns")

    return Cleaner(
        drop_empty_rows=drop_empty_rows,
        normalize_columns=normalize_columns,
        fill_missing_numeric=fill_missing_numeric,
        date_columns=date_columns,
    )

def parquet_export_config(hp: HP) -> ParquetExporter:
    path = hp.text("data/processed/events.parquet", name="path")
    return ParquetExporter(path=path)

def csv_export_config(hp: HP) -> CsvExporter:
    path = hp.text("data/processed/events.csv", name="path")
    include_header = hp.bool(True, name="include_header")

    return CsvExporter(
        path=path,
        include_header=include_header,
    )

def jsonl_export_config(hp: HP) -> JsonLinesExporter:
    path = hp.text("data/processed/events.jsonl", name="path")
    return JsonLinesExporter(path=path)

export_options = {
    "parquet": parquet_export_config,
    "csv": csv_export_config,
    "jsonl": jsonl_export_config,
}

def export_config(hp: HP):
    selected_config = hp.select(export_options, name="format", default="parquet", options_only=True)
    return hp.nest(selected_config, name="settings")

def data_pipeline_config(hp: HP) -> DataPipeline:
    mode = hp.select(["sample", "full"], name="mode", default="sample", options_only=True)

    if mode == "full":
        row_limit = hp.int(10_000_000, name="row_limit", min=1)
    else:
        row_limit = hp.int(10_000, name="row_limit", min=1)

    reader = hp.nest(input_config, name="input")
    cleaner = hp.nest(cleaning_config, name="cleaning")
    exporter = hp.nest(export_config, name="export")

    return DataPipeline(
        mode=mode,
        row_limit=row_limit,
        reader=reader,
        cleaner=cleaner,
        exporter=exporter,
    )
```
{% endcode %}

This shape assumes the reader, cleaner, and exporter constructors are lightweight and do not read or write data. Run the actual pipeline after `instantiate()` returns.

## Explore A Production Branch

{% code overflow="wrap" %}
```python
explore(
    data_pipeline_config,
    values={
        "mode": "full",
        "input.path": "s3://warehouse/events/2026-05-24.csv",
        "export.format": "jsonl",
        "export.settings.path": "s3://warehouse/processed/events.jsonl",
    },
)
```
{% endcode %}

## Instantiate A Run

{% code overflow="wrap" %}
```python
pipeline = instantiate(
    data_pipeline_config,
    values={
        "mode": "full",
        "input.path": "s3://warehouse/events/2026-05-24.csv",
        "input.delimiter": ",",
        "cleaning.fill_missing_numeric": 0.0,
        "export.format": "jsonl",
        "export.settings.path": "s3://warehouse/processed/events.jsonl",
    },
)

assert pipeline.mode == "full"
assert pipeline.reader.path.startswith("s3://")
assert pipeline.cleaner.fill_missing_numeric == 0.0
```
{% endcode %}

## Why This Shape Works

* The `mode` branch changes the default row limit while keeping the same parameter path.
* Dotted keys keep nested runtime objects replayable without returning a settings dictionary.
* `options_only=True` prevents typos in finite choices such as export formats.
* Nullable numeric values use `allow_none=True`, which makes `None` an explicit, replayable value.
