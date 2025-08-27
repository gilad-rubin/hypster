import logging
from collections import OrderedDict
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel

from ..core import Hypster
from ..hp_calls import BasicType, NumericBounds
from ..run_history import NestedHistoryRecord, ParameterRecord, ParameterSource

logger = logging.getLogger(__name__)

T = TypeVar("T", int, float, str, bool)


class ComponentBase(BaseModel):
    """Base component model with common attributes."""

    id: str
    label: str
    parameter_type: str
    value: Any
    source: ParameterSource = ParameterSource.UI


class SelectComponent(ComponentBase):
    """Selection component with options."""

    parameter_type: str = "select"
    options: List[BasicType]
    value: Union[BasicType, List[BasicType]]
    single_value: bool


class NumericComponentBase(ComponentBase, Generic[T]):
    """Base for numeric components with bounds."""

    value: Union[T, List[T]]
    bounds: Optional[NumericBounds] = None
    single_value: bool


class IntComponent(NumericComponentBase[int]):
    """Integer input component."""

    parameter_type: str = "int"
    value: Union[int, List[int]]
    single_value: bool


class FloatComponent(NumericComponentBase[float]):
    """Float input component."""

    parameter_type: str = "number"
    value: Union[float, List[float]]
    single_value: bool


class TextComponent(ComponentBase):
    """Text input component."""

    parameter_type: str = "text"
    value: Union[str, List[str]]
    single_value: bool


class BooleanComponent(ComponentBase):
    """Boolean toggle component."""

    parameter_type: str = "bool"
    value: Union[bool, List[bool]]
    single_value: bool


class NestedComponent(ComponentBase):
    """Component for handling nested configurations."""

    parameter_type: str = "nest"
    value: Dict[str, Any]
    children: Dict[str, Union["NestedComponent", ComponentBase]]


class UIHandler:
    """Generic UI handler that manages component state and updates."""

    # Component type mapping
    COMPONENT_MAPPING = {
        "select": SelectComponent,
        "multi_select": SelectComponent,
        "int": IntComponent,
        "multi_int": IntComponent,
        "number": FloatComponent,
        "multi_number": FloatComponent,
        "text": TextComponent,
        "multi_text": TextComponent,
        "bool": BooleanComponent,
        "multi_bool": BooleanComponent,
        "nest": NestedComponent,
    }

    def __init__(self, config_func: Hypster, initial_values: Optional[Dict[str, Any]] = None):
        self.config_func = config_func
        self.initial_values: Dict[str, Any] = initial_values or {}
        self.components: OrderedDict[str, ComponentBase] = OrderedDict()
        self.latest_results = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize components from initial config run."""
        try:
            self.latest_results = self.config_func(values=self.initial_values, explore_mode=True)
        except Exception as e:
            logger.error(f"Error running config: {e}")
            self.latest_results = None
        latest_records = self.config_func.run_history.get_latest_run_records()
        self.components.clear()

        for name, record in latest_records.items():
            self.components[name] = self._create_component(name, record)

    def _create_nest_component(self, name: str, record: ParameterRecord) -> NestedComponent:
        """Create a nest component."""

    def _create_component(self, name: str, record: ParameterRecord) -> ComponentBase:
        """Create a component based on parameter type."""
        if record.parameter_type == "nest":
            nested_components = {}
            nested_latest_records = record.run_history.get_latest_run_records()
            logger.debug(f"Nested latest records: {list(nested_latest_records.keys())}")
            value_dct = {}
            for child_name, child_record in nested_latest_records.items():
                child_component = self._create_component(child_name, child_record)
                if child_component:
                    nested_components[child_name] = child_component
                value_dct[child_name] = child_component.value
            return NestedComponent(id=name, label=name, value=value_dct, children=nested_components)

        component_class = self.COMPONENT_MAPPING.get(record.parameter_type)

        return component_class(
            id=name,
            label=name,
            value=record.value,
            single_value=record.single_value,
            bounds=getattr(record, "numeric_bounds", None),
            options=getattr(record, "options", None),
        )

    def _get_new_values_dict(
        self, components: OrderedDict[str, ComponentBase], component_id: str, new_value: Any
    ) -> Dict[str, Any]:
        logger.debug(f"Component ID: {component_id}")
        logger.debug(f"New value: {new_value}")
        logger.debug(f"Components: {list(components.keys())}")
        component_names = list(components.keys())
        names_up_to_id = component_names[: component_names.index(component_id)]
        values = {name: component.value for name, component in components.items() if name in names_up_to_id}
        component = components[component_id]
        if component.parameter_type == "nest":
            nested_component_id = list(new_value.keys())[0]
            nested_value = list(new_value.values())[0]
            nested_components = component.children
            logger.debug(f"Nested components: {list(nested_components.keys())}")
            logger.debug(f"Nested component ID: {nested_component_id}")
            logger.debug(f"Nested value: {nested_value}")
            values[component_id] = self._get_new_values_dict(nested_components, nested_component_id, nested_value)
        else:
            values[component_id] = new_value
        return values

    def _remove_components(
        self,
        components: OrderedDict[str, ComponentBase],
        component_id: str,
        latest_records: Union[NestedHistoryRecord, ParameterRecord],
    ) -> None:
        latest_record_names = set(latest_records.keys())
        component_names = set(components.keys())
        for name in component_names:
            if name not in latest_record_names:
                components.pop(name)

        component = components[component_id]
        if component.parameter_type == "nest":
            self._remove_components(component.children, latest_records[component_id])

    def update_components(self, component_id: str, new_value: Any) -> List[str]:
        """Update component and get affected components."""
        values = self._get_new_values_dict(self.components, component_id, new_value)

        logger.debug(f"Values: {values}")

        # Run config with new values and get latest records
        try:
            self.latest_results = self.config_func(values=values, explore_mode=True)
        except Exception as e:
            logger.warning(f"Error running config: {e}")
            self.latest_results = None
        latest_records = self.config_func.run_history.get_latest_run_records()

        # Update or create components after the changed component
        affected_components = []
        for name, record in latest_records.items():
            logger.debug(f"Processing component {name}, with record {record}")
            self.components[name] = self._create_component(name, record)
            affected_components.append(name)

        affected_values = {name: self.components[name].value for name in affected_components}
        logger.debug(f"Affected components: {affected_values}")
        return affected_components, affected_values

    def get_ordered_components(self) -> List[ComponentBase]:
        """Get components in their definition order."""
        return list(self.components.values())

    def get_component(self, component_id: str) -> Optional[ComponentBase]:
        """Get component by ID."""
        return self.components.get(component_id)

    def get_latest_results(self) -> Any:
        """Get the results from the most recent config function run."""
        return self.latest_results


def create_ui_handler(config_func: Hypster, initial_values: Optional[Dict[str, Any]] = None) -> UIHandler:
    """Create a UI handler for the given config function."""
    return UIHandler(config_func, initial_values)
