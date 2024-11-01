from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from .hp_calls import NumericBounds, ValidKeyType


class ParameterSource(str, Enum):
    USER = "user"
    UI = "ui"


class ParameterRecord(BaseModel):
    """Record of a parameter's value and metadata"""

    name: str
    parameter_type: str
    default: Optional[Union[ValidKeyType, List[ValidKeyType]]] = None
    value: Union[ValidKeyType, List[ValidKeyType]]
    is_reproducible: Union[bool, List[bool]]
    run_id: UUID
    options: Optional[List[ValidKeyType]] = None
    numeric_bounds: Optional[NumericBounds] = None
    source: ParameterSource


class NestedDBRecord(BaseModel):
    """Record type for nested configurations from propagate calls"""

    name: str
    parameter_type: str = "propagate"
    run_id: UUID
    db: "HistoryDatabase"  # Forward reference
    source: ParameterSource

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HistoryDatabase(ABC):
    """Interface for parameter history storage"""

    @abstractmethod
    def add_record(self, record: Union[ParameterRecord, NestedDBRecord]) -> None:
        pass

    @abstractmethod
    def get_run_records(
        self, run_id: str, flattened: bool = False
    ) -> Dict[str, Union[ParameterRecord, NestedDBRecord]]:
        pass

    @abstractmethod
    def get_latest_run_records(self, flattened: bool = False) -> Dict[str, Union[ParameterRecord, NestedDBRecord]]:
        pass

    @abstractmethod
    def get_param_records(
        self, param_name: str, run_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[ParameterRecord, NestedDBRecord]]:
        pass

    @abstractmethod
    def get_latest_param_record(self, param_name: str) -> Optional[Union[ParameterRecord, NestedDBRecord]]:
        pass


class InMemoryHistory(HistoryDatabase):
    """In-memory implementation of parameter history storage"""

    def __init__(self):
        self._records: Dict[str, Dict[str, Union[ParameterRecord, NestedDBRecord]]] = defaultdict(dict)
        self._run_ids: List[str] = []

    def add_record(self, record: Union[ParameterRecord, NestedDBRecord]) -> None:
        if record.run_id not in self._run_ids:
            self._run_ids.append(record.run_id)
        self._records[record.run_id][record.name] = record

    def get_run_records(
        self, run_id: str, flattened: bool = False
    ) -> Dict[str, Union[ParameterRecord, NestedDBRecord]]:
        records = self._records.get(run_id, {})
        if not flattened:
            return records
        return self._flatten_records(records)

    def get_latest_run_records(self, flattened: bool = False) -> Dict[str, Union[ParameterRecord, NestedDBRecord]]:
        if not self._run_ids:
            return {}
        return self.get_run_records(self._run_ids[-1], flattened)

    def get_param_records(
        self, param_name: str, run_ids: Optional[List[str]] = None
    ) -> Dict[str, Union[ParameterRecord, NestedDBRecord]]:
        if run_ids is None:
            run_ids = self._run_ids
        return {run_id: self._records[run_id][param_name] for run_id in run_ids if param_name in self._records[run_id]}

    def get_latest_param_record(self, param_name: str) -> Optional[Union[ParameterRecord, NestedDBRecord]]:
        for run_id in reversed(self._run_ids):
            if param_name in self._records[run_id]:
                return self._records[run_id][param_name]
        return None

    def _flatten_records(self, records: Dict[str, Union[ParameterRecord, NestedDBRecord]]) -> Dict[str, Any]:
        """Convert nested records to flat dictionary of values"""
        flattened = {}
        for name, record in records.items():
            if isinstance(record, ParameterRecord):
                flattened[name] = record.value
            elif isinstance(record, NestedDBRecord):
                nested_records = record.db.get_latest_run_records(flattened=True)
                flattened.update({f"{name}.{k}": v for k, v in nested_records.items()})
        return flattened
