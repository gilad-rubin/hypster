from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ParameterSource(Enum):
    UI = "ui"
    USER = "user"  # for programmatic calls


@dataclass
class NumericOptions:
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    allow_float: bool = True


@dataclass
class ParameterRecord:
    name: str
    parameter_type: str
    default: Any
    value: Any
    run_id: str
    options: Optional[Any] = None
    numeric_options: Optional[NumericOptions] = None
    source: ParameterSource = ParameterSource.USER


class DatabaseInterface:
    def add_record(self, record: ParameterRecord, run_id: str) -> None:
        pass

    def get_records(self, run_id: str) -> Dict[str, ParameterRecord]:
        pass

    def get_latest_record(self, param_name: str, source: Optional[ParameterSource] = None) -> Optional[ParameterRecord]:
        """Get the most recent record for a parameter name, optionally filtered by source."""
        pass


@dataclass
class NestedDBRecord:
    """Record type for nested configurations from propagate calls.
    Stores the nested database and the name prefix for nested parameters."""

    name: str
    parameter_type: str
    db: DatabaseInterface
    run_id: str
    source: ParameterSource


class InMemoryDatabase(DatabaseInterface):
    def __init__(self):
        self.records: Dict[str, Dict[str, ParameterRecord]] = {}  # run_id -> {param_name -> record}
        self.run_history: List[str] = []  # Ordered list of run_ids

    def add_record(self, record: Union[ParameterRecord, NestedDBRecord], run_id: str) -> None:
        if run_id not in self.records:
            self.records[run_id] = {}
            self.run_history.append(run_id)

        if isinstance(record, NestedDBRecord):
            # For nested records, merge the nested db records into the current db
            nested_records = record.db.get_records(run_id)
            for nested_name, nested_record in nested_records.items():
                prefixed_name = f"{record.name}.{nested_name}"
                self.records[run_id][prefixed_name] = nested_record
        else:
            self.records[run_id][record.name] = record

    def get_latest_run_records(self) -> Dict[str, Dict[str, ParameterRecord]]:
        return self.records[self.run_history[-1]]

    def get_records(self, run_id: str) -> Dict[str, ParameterRecord]:
        return self.records.get(run_id, {})

    def get_latest_record_for_param(
        self, param_name: str, source: Optional[ParameterSource] = None
    ) -> Optional[ParameterRecord]:
        """Get the most recent record for a parameter name, optionally filtered by source."""
        for run_id in reversed(self.run_history):
            records = self.records.get(run_id, {})
            if param_name in records:
                record = records[param_name]
                if source is None or record.source == source:
                    return record
        return None
