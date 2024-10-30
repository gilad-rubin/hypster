import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
from IPython.display import display

logger = logging.getLogger("hypster.interactive")


class Component(ABC):
    """Base component interface."""

    def __init__(self, id: str, label: str, value: Any):
        self.id = id
        self.label = label
        self.value = value

    def get_state(self) -> dict:
        """Get component's current state as a dictionary."""
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def update(self, **kwargs) -> None:
        """Update component attributes."""
        state_changed = False
        for key, value in kwargs.items():
            if hasattr(self, key) and getattr(self, key) != value:
                setattr(self, key, value)
                state_changed = True
        return state_changed

    def equals(self, other: "Component") -> bool:
        """Compare components' states."""
        if not isinstance(other, self.__class__):
            return False
        return self.get_state() == other.get_state()


class SelectComponent(Component):
    """Selection from a list of options."""

    def __init__(self, id: str, label: str, value: Any, options: List[Any]):
        super().__init__(id, label, value)
        self.options = options


class NumericComponent(Component):
    """Base for numeric inputs."""

    def __init__(
        self,
        id: str,
        label: str,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
    ):
        super().__init__(id, label, value)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self._validate_type()

    def _validate_type(self):
        expected_type = int if isinstance(self.step, int) else float
        if not isinstance(self.value, expected_type):
            raise TypeError(f"Value must be of type {expected_type}")

    def update(self, **kwargs) -> None:
        """Update component attributes and validate type."""
        state_changed = super().update(**kwargs)
        if state_changed:
            self._validate_type()
        return state_changed


class UIComponent(ABC):
    """Base UI component interface."""

    def __init__(self, component: Component):
        self.component = component
        self.widget = self._create_widget()

    @abstractmethod
    def _create_widget(self) -> Any:
        """Create the widget."""
        pass

    @abstractmethod
    def render(self) -> Any:
        """Render the component."""
        pass

    def update(self, new_component: Component) -> None:
        """Update component with new state."""
        # Store old state before updating
        old_state = self.component.get_state()
        new_state = new_component.get_state()

        logger.debug(f"Comparing states for {self.component.id}:")
        logger.debug(f"Old state: {old_state}")
        logger.debug(f"New state: {new_state}")
        if old_state != new_state:
            logger.debug(f"Component {new_component.id} state changed, updating widget")
            self.component = new_component
            self._update_widget()
        else:
            logger.debug(f"Component {new_component.id} state unchanged, skipping widget update")

    @abstractmethod
    def _update_widget(self) -> None:
        """Update widget with current component state."""
        pass


class IPySelectComponent(UIComponent):
    def __init__(self, component: SelectComponent, on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        logger.debug(f"Creating select widget for {self.component.id}")
        widget = widgets.Dropdown(
            options=self.component.options,
            value=self.component.value,
            description=self.component.label,
            style={"description_width": "initial"},
        )
        widget.observe(lambda change: self.on_change(self.component.id, change["new"]), names="value")
        return widget

    def render(self) -> widgets.Widget:
        return self.widget

    def _update_widget(self) -> None:
        logger.debug(f"Updating select widget {self.component.id}")
        # First update options, then value to ensure value is valid
        self.widget.options = self.component.options
        self.widget.value = self.component.value
        self.widget.description = self.component.label


class IPyNumericComponent(UIComponent):
    def __init__(self, component: NumericComponent, on_change: Callable):
        self.on_change = on_change
        super().__init__(component)

    def _create_widget(self) -> widgets.Widget:
        logger.debug(f"Creating numeric widget for {self.component.id}")
        widget_cls = widgets.IntSlider if isinstance(self.component.step, int) else widgets.FloatSlider
        widget = widget_cls(
            value=self.component.value,
            min=self.component.min_value or -100,
            max=self.component.max_value or 100,
            step=self.component.step,
            description=self.component.label,
            style={"description_width": "initial"},
            continuous_update=False,
        )
        widget.observe(lambda change: self.on_change(self.component.id, change["new"]), names="value")
        return widget

    def render(self) -> widgets.Widget:
        return self.widget

    def _update_widget(self) -> None:
        logger.debug(f"Updating numeric widget {self.component.id} with:")
        logger.debug(f"  value: {self.component.value}")
        logger.debug(f"  min: {self.component.min_value}")
        logger.debug(f"  max: {self.component.max_value}")

        min_value = self.component.value - 100 if self.component.min_value is None else self.component.min_value
        max_value = self.component.value + 100 if self.component.max_value is None else self.component.max_value

        self.widget.value = self.component.value

        self.widget.min = min_value
        self.widget.max = max_value
        self.widget.step = self.component.step

        self.widget.description = self.component.label


class SelectionHandler:
    """Manages component state and updates."""

    def __init__(self, config_func: Callable):
        self.config_func = config_func
        self.components: Dict[str, Component] = {}
        self._component_order: List[str] = []  # Track order of components
        logger.debug("Initializing SelectionHandler")
        self._initialize_components()

    def _initialize_components(self) -> None:
        logger.debug("Running initial config")
        self.config_func()
        latest_run_id = self.config_func.db.get_run_ids()[-1]
        latest_records = self.config_func.db.get_records(latest_run_id)

        logger.debug(f"Initial records: {latest_records}")
        self.components.clear()
        self._component_order = list(latest_records.keys())  # Preserve order from OrderedDict

        for name, record in latest_records.items():
            logger.debug(f"Creating component for {name}: {record}")
            if record.parameter_type == "select":
                self.components[name] = SelectComponent(id=name, label=name, value=record.value, options=record.options)
            elif record.parameter_type == "int":
                self.components[name] = NumericComponent(
                    id=name,
                    label=name,
                    value=record.value,
                    min_value=record.options.min if record.options else None,
                    max_value=record.options.max if record.options else None,
                    step=1,
                )

    def update_component(self, component_id: str, **kwargs) -> None:
        logger.debug(f"Updating component {component_id} with {kwargs}")
        if component_id in self.components:
            self.components[component_id].update(**kwargs)

            # Get values only up to the changed component
            values = {}
            for name in self._component_order:
                if name in self.components:
                    values[name] = self.components[name].value
                if name == component_id:
                    break

            logger.debug(f"Updating config with values up to {component_id}: {values}")
            self.config_func(values=values)

            self._refresh_components()

    def _refresh_components(self) -> None:
        logger.debug("Refreshing components from latest config")
        latest_run_id = self.config_func.db.get_run_ids()[-1]
        latest_records = self.config_func.db.get_records(latest_run_id)

        logger.debug(f"Latest records: {latest_records}")

        # Update component order
        self._component_order = list(latest_records.keys())

        # Keep track of valid component IDs
        current_component_ids = set(latest_records.keys())
        existing_component_ids = set(self.components.keys())

        # Remove components that are no longer in the config
        for component_id in existing_component_ids - current_component_ids:
            logger.debug(f"Removing component {component_id}")
            self.components.pop(component_id)

        # Update or create components
        for name, record in latest_records.items():
            logger.debug(f"Processing record for {name}: {record}")
            if name in self.components:
                # Instead of updating in place, create a new component
                component = self.components[name]

                if isinstance(component, SelectComponent):
                    # Instead of updating in place, create a new component
                    new_value = record.value if record.value in record.options else record.default
                    new_component = SelectComponent(id=name, label=name, value=new_value, options=record.options)
                    self.components[name] = new_component

                elif isinstance(component, NumericComponent):
                    new_min = record.options.min if record.options else None
                    new_max = record.options.max if record.options else None
                    current_value = component.value

                    if (new_min is not None and current_value < new_min) or (
                        new_max is not None and current_value > new_max
                    ):
                        logger.debug(
                            f"Current value {current_value} is outside new bounds {new_min} {new_max}"
                            f" using default {record.default}"
                        )
                        new_value = record.default
                    else:
                        new_value = current_value

                    new_component = NumericComponent(
                        id=name, label=name, value=new_value, min_value=new_min, max_value=new_max, step=1
                    )
                    self.components[name] = new_component
            else:
                # Create new component
                logger.debug(f"Creating new component for {name}")
                if record.parameter_type == "select":
                    self.components[name] = SelectComponent(
                        id=name, label=name, value=record.value, options=record.options
                    )
                elif record.parameter_type == "int":
                    self.components[name] = NumericComponent(
                        id=name,
                        label=name,
                        value=record.value,
                        min_value=record.options.min if record.options else None,
                        max_value=record.options.max if record.options else None,
                        step=1,
                    )

    def get_ordered_components(self) -> List[Component]:
        """Get components in the order they were defined."""
        return [self.components[name] for name in self._component_order if name in self.components]


class IPyWidgetsUI:
    def __init__(self, selection_handler: SelectionHandler):
        self.selection_handler = selection_handler
        self.ui_components: Dict[str, UIComponent] = {}
        self.container = widgets.VBox([])
        self.output = widgets.Output()
        logger.debug("Initialized IPyWidgetsUI")

    def _create_ui_component(self, component: Component) -> UIComponent:
        logger.debug(f"Creating UI component for {component.id}")
        if isinstance(component, SelectComponent):
            return IPySelectComponent(component, self._handle_change)
        elif isinstance(component, NumericComponent):
            return IPyNumericComponent(component, self._handle_change)
        raise ValueError(f"Unsupported component type: {type(component)}")

    def _handle_change(self, component_id: str, new_value: Any):
        logger.debug(f"Handle change for {component_id}: {new_value}")
        with self.output:
            self.selection_handler.update_component(component_id, value=new_value)
            self._update_display()

    def _update_display(self):
        logger.debug("Updating display")
        widgets_list = []

        # Get ordered components from selection handler
        ordered_components = self.selection_handler.get_ordered_components()
        current_component_ids = {comp.id for comp in ordered_components}
        existing_component_ids = set(self.ui_components.keys())

        # Remove UI components that are no longer in the config
        for component_id in existing_component_ids - current_component_ids:
            logger.debug(f"Removing UI component {component_id}")
            self.ui_components.pop(component_id)

        # Update or create UI components in the correct order
        for component in ordered_components:
            logger.debug(f"Processing UI component for {component.id}")
            if component.id not in self.ui_components:
                logger.debug(f"Creating new UI component for {component.id}")
                self.ui_components[component.id] = self._create_ui_component(component)
            else:
                logger.debug(f"Updating existing UI component for {component.id}")
                self.ui_components[component.id].update(component)

            widgets_list.append(self.ui_components[component.id].render())

        logger.debug(f"Final widget list length: {len(widgets_list)}")
        widgets_list.append(self.output)
        self.container.children = widgets_list

    def display(self):
        logger.debug("Displaying UI")
        self._update_display()
        display(self.container)


def create_interactive_config(config_func: Callable) -> Tuple[SelectionHandler, IPyWidgetsUI]:
    """
    Create and display an interactive configuration interface.

    Args:
        config_func: A configuration function decorated with @config

    Returns:
        Tuple of (SelectionHandler, IPyWidgetsUI) if you need to access them later
    """
    handler = SelectionHandler(config_func)
    ui = IPyWidgetsUI(handler)
    ui.display()
    return handler, ui


# Simple wrapper if you don't need access to the handler and UI
def display_interactive_config(config_func: Callable) -> None:
    """
    Display an interactive configuration interface.

    Args:
        config_func: A configuration function decorated with @config
    """
    create_interactive_config(config_func)
