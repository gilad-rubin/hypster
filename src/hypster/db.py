from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

# Define data classes for logging


@dataclass
class NumericOptions:
    min: float | int | None
    max: float | int | None
    allow_int: bool = True
    allow_float: bool = True


@dataclass
class ParameterRecord:
    name: str
    parameter_type: str
    default: any
    value: any
    run_id: str
    options: list | NumericOptions | None = None


@dataclass
class NestedDBRecord:
    name: str
    parameter_type: str
    db: "DatabaseInterface"
    run_id: str


class DatabaseInterface:
    """Abstract database interface."""

    def add_record(self, record: ParameterRecord, run_id: str) -> None:
        """Add a parameter record to the database."""
        raise NotImplementedError("DatabaseInterface subclasses must implement the add_record method.")

    def get_records(self, run_id: str) -> Dict[str, ParameterRecord]:
        """Retrieve all records from the database."""
        raise NotImplementedError("DatabaseInterface subclasses must implement the get_records method.")

    def get_latest_record(self, param_name: str, run_id: str) -> Optional[ParameterRecord]:
        """Get the most recent record for a parameter."""
        raise NotImplementedError("DatabaseInterface subclasses must implement the get_latest_record method.")

    def get_run_ids(self) -> List[str]:
        """Get all run IDs in chronological order."""
        raise NotImplementedError("DatabaseInterface subclasses must implement the get_run_ids method.")


class InMemoryDatabase(DatabaseInterface):
    """In-memory database implementation using ordered dictionaries."""

    def __init__(self):
        # Store runs in order, each run containing ordered parameters
        self.data_store: OrderedDict[str, OrderedDict[str, ParameterRecord]] = OrderedDict()
        self.run_ids: List[str] = []

    def add_record(self, record: ParameterRecord, run_id: str) -> None:
        """
        Add a parameter record to the database.
        Creates a new run if run_id doesn't exist, or adds to existing run.
        """
        # Create new run if it doesn't exist
        if run_id not in self.data_store:
            self.data_store[run_id] = OrderedDict()
            self.run_ids.append(run_id)

        # Add/Update parameter record for this run
        record.run_id = run_id
        self.data_store[run_id][record.name] = record

    def get_records(self, run_id: str) -> OrderedDict[str, ParameterRecord]:
        """Retrieve all records from a specific run."""
        return self.data_store.get(run_id, OrderedDict())

    def get_latest_record(self, param_name: str, run_id: str) -> Optional[ParameterRecord]:
        """
        Get the most recent record for a parameter.
        If run_id is specified, gets from that run; otherwise searches through all runs
        in reverse order.
        """
        # Get from specific run
        return self.data_store.get(run_id, OrderedDict()).get(param_name)

        return None

    def get_run_ids(self) -> List[str]:
        """Get all run IDs in chronological order."""
        return self.run_ids.copy()

    def get_latest_records(self) -> OrderedDict[str, ParameterRecord]:
        """Get all records from the most recent run."""
        if not self.run_ids:
            return OrderedDict()
        return self.get_records(self.run_ids[-1])

    def start_new_run(self, run_id: str) -> None:
        """
        Start a new run with the given ID.
        """
        if run_id not in self.data_store:
            self.data_store[run_id] = OrderedDict()
            self.run_ids.append(run_id)
