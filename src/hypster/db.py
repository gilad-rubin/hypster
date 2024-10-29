from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict

# Define data classes for logging


@dataclass
class NumericOptions:
    min_val: float | int | None
    max_val: float | int | None
    allow_int: bool = True
    allow_float: bool = True


@dataclass
class ParameterRecord:
    name: str
    parameter_type: str
    default: any
    value: any
    options: list | NumericOptions | None = None


@dataclass
class NestedDBRecord:
    db: "DatabaseInterface"


class DatabaseInterface:
    """Abstract database interface."""

    def add_record(self, record: ParameterRecord) -> None:
        """Add a parameter record to the database."""
        raise NotImplementedError("DatabaseInterface subclasses must implement the add_record method.")

    def get_records(self) -> Dict[str, ParameterRecord]:
        """Retrieve all records from the database."""
        raise NotImplementedError("DatabaseInterface subclasses must implement the get_records method.")


class InMemoryDatabase(DatabaseInterface):
    """In-memory database implementation using a dictionary."""

    def __init__(self):
        self.data_store: OrderedDict[str, ParameterRecord] = OrderedDict()

    def add_record(self, record: ParameterRecord) -> None:
        """Add a parameter record to the database."""
        # TODO: add serializable check
        # if not record.serializable:
        #    record.value = str(record.value)
        self.data_store[record.name] = record

    def get_records(self) -> Dict[str, ParameterRecord]:
        """Retrieve all records from the database."""
        return self.data_store
