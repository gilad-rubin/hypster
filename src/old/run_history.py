from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from .hp_calls import BasicType, NumericBounds


class ParameterSource(str, Enum):
    USER = "user"
    UI = "ui"


class HistoryRecord(BaseModel):
    """Base class for all history records"""

    name: str
    parameter_type: str
    run_id: UUID
    source: ParameterSource


class ParameterRecord(HistoryRecord):
    """Record of a parameter's value and metadata"""

    single_value: bool
    default: Optional[Union[BasicType, List[BasicType]]] = None
    value: Union[BasicType, List[BasicType]]
    is_reproducible: Union[bool, List[bool]]
    options: Optional[List[BasicType]] = None
    numeric_bounds: Optional[NumericBounds] = None


class NestedHistoryRecord(HistoryRecord):
    """Record type for nested configurations from nest calls"""

    run_history: "HistoryDatabase"  # Forward reference

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HistoryDatabase(ABC):
    """Interface for parameter history storage"""

    @abstractmethod
    def add_record(self, record: Union[ParameterRecord, NestedHistoryRecord]) -> None:
        pass

    @abstractmethod
    def get_run_records(
        self, run_id: str, flattened: bool = False
    ) -> Dict[str, Union[ParameterRecord, NestedHistoryRecord]]:
        pass

    @abstractmethod
    def get_latest_run_records(self, flattened: bool = False) -> Dict[str, Union[ParameterRecord, NestedHistoryRecord]]:
        pass

    @abstractmethod
    def get_param_records(
        self, param_name: str, run_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[ParameterRecord, NestedHistoryRecord]]:
        pass

    @abstractmethod
    def get_latest_param_record(self, param_name: str) -> Optional[Union[ParameterRecord, NestedHistoryRecord]]:
        pass


class InMemoryHistory(HistoryDatabase):
    """In-memory implementation of parameter history storage"""

    def __init__(self):
        self._records: Dict[str, OrderedDict[str, Union[ParameterRecord, NestedHistoryRecord]]] = defaultdict(
            OrderedDict
        )
        self._run_ids: List[str] = []

    def add_record(self, record: Union[ParameterRecord, NestedHistoryRecord]) -> None:
        if record.run_id not in self._run_ids:
            self._run_ids.append(record.run_id)
        if record.run_id not in self._records:
            self._records[record.run_id] = OrderedDict()
        self._records[record.run_id][record.name] = record

    def get_run_records(self, run_id: Optional[str] = None, flattened: bool = False) -> List[Dict[str, Any]]:
        if run_id is None:  # get all records
            records = self._records
        else:
            records = self._records.get(run_id, {})

        if not flattened:
            return records
        else:
            flattened_records = []
            for run_id, run_records in self._records.items():
                flattened_records.append(self._flatten_records(run_records))
            return flattened_records

    def get_latest_run_records(self, flattened: bool = False) -> Dict[str, Any]:
        if not self._run_ids:
            return {}
        run_id = self._run_ids[-1]
        records = self._records.get(run_id, {})
        if not flattened:
            return records
        else:
            return self._flatten_records(records)

    def get_param_records(
        self, param_name: str, run_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[ParameterRecord, NestedHistoryRecord]]:
        if run_ids is None:
            run_ids = self._run_ids
        return {run_id: self._records[run_id][param_name] for run_id in run_ids if param_name in self._records[run_id]}

    def get_latest_param_record(self, param_name: str) -> Optional[Union[ParameterRecord, NestedHistoryRecord]]:
        for run_id in reversed(self._run_ids):
            if param_name in self._records[run_id]:
                return self._records[run_id][param_name]
        return None

    def check_reproducibility(self, record: ParameterRecord) -> None:
        if isinstance(record.is_reproducible, bool):
            if not record.is_reproducible:
                print(
                    f"Value {record.value} of parameter {record.name} was not originally of type: "
                    "boolean, string, int or float type."
                    "This means that putting this value in the config as-is will likely lead to an error."
                )
        elif isinstance(record.is_reproducible, List):
            not_reproducible_values = []
            for value, is_reproducible in zip(record.value, record.is_reproducible):
                if not is_reproducible:
                    not_reproducible_values.append(value)
            if not_reproducible_values:
                print(
                    f"Values {not_reproducible_values} of parameter {record.name} were not originally of type: "
                    "boolean, string, int or float type."
                    "This means that putting these values in the config as-is will likely lead to an error."
                )

    def _flatten_records(self, records: Dict[str, Union[ParameterRecord, NestedHistoryRecord]]) -> Dict[str, Any]:
        """Convert nested records to flat dictionary of values"""
        flattened = {}
        for name, record in records.items():
            if isinstance(record, ParameterRecord):
                flattened[name] = record.value
                self.check_reproducibility(record)
            elif isinstance(record, NestedHistoryRecord):
                nested_records = record.run_history.get_latest_run_records(flattened=True)
                flattened.update({f"{name}.{k}": v for k, v in nested_records.items()})
        return flattened
