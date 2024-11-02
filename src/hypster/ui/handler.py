import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel

from ..hp_calls import NumericBounds, ValidKeyType
from ..run_history import ParameterRecord, ParameterSource

logger = logging.getLogger("hypster.ui.handler")

T = TypeVar("T", int, float, str, bool)


class ComponentBase(BaseModel):
    """Base component model with common attributes."""

    id: str
    label: str
    value: Any
    parameter_type: str
    source: ParameterSource = ParameterSource.UI
    single_value: bool = True


class SelectComponent(ComponentBase):
    """Selection component with options."""

    parameter_type: str = "select"
    options: List[ValidKeyType]
    value: Union[ValidKeyType, List[ValidKeyType]]
    single_value: bool = True


class NumericComponentBase(ComponentBase, Generic[T]):
    """Base for numeric components with bounds."""

    value: Union[T, List[T]]
    bounds: Optional[NumericBounds] = None
    single_value: bool = True


class IntComponent(NumericComponentBase[int]):
    """Integer input component."""

    parameter_type: str = "int"
    value: Union[int, List[int]]


class FloatComponent(NumericComponentBase[float]):
    """Float input component."""

    parameter_type: str = "number"
    value: Union[float, List[float]]


class TextComponent(ComponentBase):
    """Text input component."""

    parameter_type: str = "text"
    value: Union[str, List[str]]
    single_value: bool = True


class BooleanComponent(ComponentBase):
    """Boolean toggle component."""

    parameter_type: str = "bool"
    value: Union[bool, List[bool]]
    single_value: bool = True


class UIComponent(ABC):
    """Base class for UI components."""

    def __init__(self, component: ComponentBase):
        self.component = component
        self.widget = self._create_widget()

    @abstractmethod
    def _create_widget(self) -> Any:
        """Create the UI widget."""
        pass

    def render(self) -> Any:
        """Render the widget."""
        return self.widget

    def update(self, component: ComponentBase) -> None:
        """Update the component and widget."""
        self.component = component
        self._update_widget()

    @abstractmethod
    def _update_widget(self) -> None:
        """Update the widget with current component state."""
        pass


class UIHandler:
    """Generic UI handler that manages component state and updates."""

    def __init__(self, config_func: Callable):
        self.config_func = config_func
        self.components: OrderedDict[str, ComponentBase] = OrderedDict()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize components from initial config run."""
        self.config_func(explore_mode=True)
        latest_records = self.config_func.run_history.get_latest_run_records()
        self.components.clear()

        for name, record in latest_records.items():
            if record.__class__.__name__ == "ParameterRecord":
                component = self._create_component(name, record)
                if component:
                    self.components[name] = component

    def _create_component(self, name: str, record: ParameterRecord) -> Optional[ComponentBase]:
        """Create a component based on parameter type."""
        component_data = {"id": name, "label": name, "value": record.value, "single_value": record.single_value}

        if record.parameter_type in ["select", "multi_select"]:
            return SelectComponent(**component_data, options=record.options, parameter_type="select")
        elif record.parameter_type in ["int", "multi_int"]:
            return IntComponent(**component_data, bounds=record.numeric_bounds, parameter_type="int")
        elif record.parameter_type in ["number", "multi_number"]:
            return FloatComponent(**component_data, bounds=record.numeric_bounds, parameter_type="number")
        elif record.parameter_type in ["text", "multi_text"]:
            return TextComponent(**component_data, parameter_type="text")
        elif record.parameter_type in ["bool", "multi_bool"]:
            return BooleanComponent(**component_data, parameter_type="bool")
        return None

    def _get_components_up_to(self, component_id: str) -> Dict[str, ComponentBase]:
        components = {}
        for name, component in self.components.items():
            components[name] = component
            if name == component_id:
                break
        return components

    def update_component(self, component_id: str, new_value: Any) -> Dict[str, ComponentBase]:
        """Update component and get affected components."""

        components_for_values = self._get_components_up_to(component_id)
        # logger.debug(f"Components for values: {components_for_values.keys()}")
        values = {name: component.value for name, component in components_for_values.items()}
        values[component_id] = new_value

        logger.debug(f"Values: {values}")
        # Run config with new values and get latest records
        self.config_func(values=values, explore_mode=True)
        latest_records = self.config_func.run_history.get_latest_run_records()
        latest_record_names = set(latest_records.keys())

        # Remove components that aren't in latest records
        new_components = set(self.components.keys())
        for name in new_components:
            if name not in latest_record_names:
                self.components.pop(name)

        # Update or create components after the changed component
        affected_components = OrderedDict()
        for name, record in latest_records.items():
            logger.debug(f"Processing component {name}")
            if record.__class__.__name__ != "ParameterRecord":
                continue

            component = self._create_component(name, record)
            if component:
                self.components[name] = component
                affected_components[name] = component

        return affected_components

    def get_ordered_components(self) -> List[ComponentBase]:
        """Get components in their definition order."""
        return list(self.components.values())

    def get_component(self, component_id: str) -> Optional[ComponentBase]:
        """Get component by ID."""
        return self.components.get(component_id)


def create_ui_handler(config_func: Callable) -> UIHandler:
    """Create a UI handler for the given config function."""
    return UIHandler(config_func)
