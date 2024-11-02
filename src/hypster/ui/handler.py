import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, PrivateAttr, ValidationInfo, field_validator

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

    def equals(self, other: "ComponentBase") -> bool:
        return isinstance(other, self.__class__) and self.model_dump() == other.model_dump()


class SelectComponent(ComponentBase):
    """Selection component with options."""

    parameter_type: str = "select"
    options: List[ValidKeyType]
    value: Union[ValidKeyType, List[ValidKeyType]]
    single_value: bool = True

    @property
    def widget_config(self) -> Dict[str, Any]:
        """Get widget configuration."""
        return {
            "options": self.options,
            "value": self.value,
            "description": self.label,
        }


class NumericComponentBase(ComponentBase, Generic[T]):
    """Base for numeric components with bounds."""

    value: Union[T, List[T]]
    bounds: Optional[NumericBounds] = None
    min_step: T
    single_value: bool = True
    _effective_bounds: Optional[NumericBounds] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self.bounds:
            self._effective_bounds = self.bounds
        else:
            self._effective_bounds = NumericBounds(
                min_val=self.value - 100 if isinstance(self.value, (int, float)) else 0,
                max_val=self.value + 100 if isinstance(self.value, (int, float)) else 0,
            )
        logger.debug(f"Initialized {self.id} with effective_bounds: {self._effective_bounds}")

    @property
    def effective_bounds(self) -> NumericBounds:
        """Get the actual bounds being used by the widget."""
        return self._effective_bounds

    @field_validator("value")
    def validate_bounds(cls, v: Union[T, List[T]], info: ValidationInfo) -> Union[T, List[T]]:
        bounds = info.data.get("bounds")
        if not bounds:
            return v

        def validate_single(val: T) -> None:
            if bounds.min_val is not None and val < bounds.min_val:
                raise ValueError(f"Value must be >= {bounds.min_val}")
            if bounds.max_val is not None and val > bounds.max_val:
                raise ValueError(f"Value must be <= {bounds.max_val}")

        if isinstance(v, list):
            for val in v:
                validate_single(val)
        else:
            validate_single(v)
        return v

    @property
    def step(self) -> T:
        """Calculate step size based on bounds."""
        if self.bounds:
            return max((self.bounds.max_val - self.bounds.min_val) / 100, self.min_step)
        return self.min_step

    @property
    def widget_config(self) -> Dict[str, Any]:
        """Get widget configuration."""
        bounds = self.effective_bounds
        logger.debug(f"Widget config for {self.id} using bounds: {bounds}")
        return {
            "value": self.value,
            "min": bounds.min_val,
            "max": bounds.max_val,
            "step": self.min_step,
            "description": self.label,
        }


class IntComponent(NumericComponentBase[int]):
    """Integer input component."""

    parameter_type: str = "int"
    value: Union[int, List[int]]
    min_step: int = 1


class FloatComponent(NumericComponentBase[float]):
    """Float input component."""

    parameter_type: str = "number"
    value: Union[float, List[float]]
    min_step: float = 0.01


class TextComponent(ComponentBase):
    """Text input component."""

    parameter_type: str = "text"
    value: Union[str, List[str]]
    single_value: bool = True

    @property
    def widget_config(self) -> Dict[str, Any]:
        """Get widget configuration."""
        return {
            "value": self.value,
            "description": self.label,
        }


class BooleanComponent(ComponentBase):
    """Boolean toggle component."""

    parameter_type: str = "bool"
    value: Union[bool, List[bool]]
    single_value: bool = True

    @property
    def widget_config(self) -> Dict[str, Any]:
        """Get widget configuration."""
        return {
            "value": self.value,
            "description": self.label,
        }


# UI Components
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
        logger.debug("Initializing UIHandler")
        self.config_func = config_func
        self.components: OrderedDict[str, ComponentBase] = OrderedDict()
        self._initialize_components()
        logger.debug(f"Initialized with components: {list(self.components.keys())}")

    def _initialize_components(self) -> None:
        """Initialize components from initial config run."""
        logger.debug("Running initial config")
        self.config_func(explore_mode=True)
        latest_records = self.config_func.run_history.get_latest_run_records()

        logger.debug(f"Initial records: {latest_records}")
        self.components.clear()

        for name, record in latest_records.items():
            logger.debug(f"Processing record: {name}")
            logger.debug(f"Record type: {type(record)}")
            logger.debug(f"Record dir: {dir(record)}")
            logger.debug(f"Is ParameterRecord: {isinstance(record, ParameterRecord)}")
            logger.debug(f"ParameterRecord module: {ParameterRecord.__module__}")
            logger.debug(f"Record module: {record.__class__.__module__}")

            # Try direct class comparison
            if record.__class__.__name__ == "ParameterRecord":
                component_info = self._get_component_info(record.parameter_type)
                if component_info:
                    self._create_or_update_component(name, record, component_info)
                    logger.debug(f"Created component: {name}")

    def _create_or_update_component(
        self, name: str, record: ParameterRecord, component_info: Tuple[Type[ComponentBase], bool]
    ) -> None:
        """Create or update a component based on record data."""
        component_cls, is_single = component_info
        logger.debug(f"Creating/updating component {name} with type {component_cls.__name__}")

        # Check if we have an existing numeric component
        existing_component = self.components.get(name)
        use_existing_bounds = False

        if existing_component and issubclass(component_cls, NumericComponentBase):
            logger.debug(f"Found existing numeric component for {name}")

            # Check if types match
            same_type = (isinstance(existing_component, IntComponent) and component_cls == IntComponent) or (
                isinstance(existing_component, FloatComponent) and component_cls == FloatComponent
            )

            if same_type:
                new_value = record.value
                current_bounds = existing_component.effective_bounds
                logger.debug(f"Checking new value {new_value} against current bounds {current_bounds}")

                def check_bounds(val):
                    return (current_bounds.min_val is None or val >= current_bounds.min_val) and (
                        current_bounds.max_val is None or val <= current_bounds.max_val
                    )

                # Check if single value or list
                if isinstance(new_value, list):
                    within_bounds = all(check_bounds(val) for val in new_value)
                else:
                    within_bounds = check_bounds(new_value)

                logger.debug(f"Within bounds: {within_bounds}")

                if within_bounds:
                    use_existing_bounds = True
                    logger.debug(f"Will keep existing bounds for {name}")

        component_data = {
            "id": name,
            "label": name,
            "value": record.value,
            "parameter_type": record.parameter_type,
            "single_value": is_single,
        }

        if component_cls == SelectComponent:
            component_data["options"] = record.options or []
        elif issubclass(component_cls, NumericComponentBase):
            if use_existing_bounds and existing_component:
                logger.debug(f"Using existing bounds for {name}")
                # Set bounds to effective_bounds instead of bounds
                component_data["bounds"] = existing_component.effective_bounds
                component_data["min_step"] = existing_component.min_step
            else:
                logger.debug(f"Using new bounds for {name}")
                logger.debug(f"New record bounds: {record.numeric_bounds}")
                component_data["bounds"] = record.numeric_bounds
                component_data["min_step"] = 1 if component_cls == IntComponent else 0.01

        try:
            logger.debug(f"Creating new component with data: {component_data}")
            new_component = component_cls(**component_data)

            # Only log bounds for numeric components
            if issubclass(component_cls, NumericComponentBase):
                logger.debug(f"New component bounds: {new_component.bounds}")
                logger.debug(f"New component effective_bounds: {new_component.effective_bounds}")
                logger.debug(f"New component widget config: {new_component.widget_config}")

            self.components[name] = new_component
            logger.debug(f"Successfully created/updated component {name}")
        except ValueError as e:
            logger.warning(f"Failed to create component {name}: {e}")
            if issubclass(component_cls, NumericComponentBase):
                fallback_value = record.numeric_bounds.min_val if record.numeric_bounds else 0
                component_data["value"] = fallback_value
                self.components[name] = component_cls(**component_data)
                logger.debug(f"Created component {name} with fallback value {fallback_value}")

    def update_component(self, component_id: str, new_value: Any) -> Dict[str, ComponentBase]:
        """Update component and get affected components."""
        logger.debug(f"Updating component {component_id} with value {new_value}")

        if component_id not in self.components:
            return {}

        # Get values up to and including the changed component
        values = {}
        found_target = False
        for name, component in self.components.items():
            values[name] = new_value if name == component_id else component.value
            if name == component_id:
                found_target = True
                break

        if not found_target:
            return {}

        logger.debug(f"Running config with values: {values}")
        self.config_func(values=values, explore_mode=True)

        latest_records = self.config_func.run_history.get_latest_run_records()
        logger.debug(f"Got latest records: {latest_records}")

        # Create new OrderedDict to maintain latest_records order
        new_components = OrderedDict()
        affected_components = {}
        update_started = False

        # First, copy all components up to and including the changed one
        for name, component in self.components.items():
            new_components[name] = component
            if name == component_id:
                update_started = True
                break

        # Then process the rest based on latest_records order
        for name, record in latest_records.items():
            if update_started:
                if record.__class__.__name__ == "ParameterRecord":
                    component_info = self._get_component_info(record.parameter_type)
                    if component_info:
                        self._create_or_update_component(name, record, component_info)
                        new_components[name] = self.components[name]
                        affected_components[name] = self.components[name]

        # Update the components dict with the new ordered dict
        self.components = new_components

        logger.debug(f"Updated components: {list(affected_components.keys())}")
        return affected_components

    def _get_component_info(self, parameter_type: str) -> Optional[Tuple[Type[ComponentBase], bool]]:
        """Get component class and single flag for parameter type."""
        COMPONENT_MAPPING = {
            "select": (SelectComponent, True),
            "multi_select": (SelectComponent, False),
            "int": (IntComponent, True),
            "multi_int": (IntComponent, False),
            "number": (FloatComponent, True),
            "multi_number": (FloatComponent, False),
            "text": (TextComponent, True),
            "multi_text": (TextComponent, False),
            "bool": (BooleanComponent, True),
            "multi_bool": (BooleanComponent, False),
        }
        return COMPONENT_MAPPING.get(parameter_type)

    def get_ordered_components(self) -> List[ComponentBase]:
        """Get components in their definition order."""
        return list(self.components.values())

    def get_component(self, component_id: str) -> Optional[ComponentBase]:
        """Get component by ID."""
        return self.components.get(component_id)


def create_ui_handler(config_func: Callable) -> UIHandler:
    """Create a UI handler for the given config function."""
    return UIHandler(config_func)
