import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Set, Tuple

import ipywidgets as widgets
from IPython.display import HTML, display

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("hypster.interactive")


@dataclass
class ParameterState:
    """Represents the current state and options of a parameter."""

    name: str
    current_value: Any
    available_options: Set[Any]


class SelectionHandler:
    """Handles parameter selections and manages their states."""

    def __init__(self, config_func: Callable) -> None:
        self.config_func = config_func
        self.selected_params: OrderedDict[str, Any] = OrderedDict()
        self.current_options: OrderedDict[str, Set[Any]] = OrderedDict()
        self.parameter_history: list[OrderedDict[str, Any]] = []
        logger.info("Initialized SelectionHandler")

    def update_param(self, param_name: str, value: Any) -> None:
        """
        Update a parameter value and recalculate dependent parameters.

        Args:
            param_name (str): The name of the parameter to update.
            value (Any): The new value for the parameter.
        """
        logger.info(f"Updating parameter '{param_name}' to '{value}'")
        logger.debug(f"Current state before update: {dict(self.selected_params)}")

        # Store current state in history
        old_state = self.selected_params.copy()
        self.parameter_history.append(old_state)

        # Update the parameter
        self.selected_params[param_name] = value

        # Rerun config to update dependent parameters
        self._run_config()

    def _run_config(self) -> None:
        """Run configuration to determine current parameter states."""
        logger.info("Running configuration")
        logger.debug(f"Current selected params: {dict(self.selected_params)}")

        # Store current selections
        current_selections = self.selected_params.copy()
        logger.debug(f"Stored current selections: {dict(current_selections)}")

        # Get the last known values for each parameter
        last_known_values = {}
        for param_name in current_selections:
            for run_id in reversed(self.config_func.db.get_run_ids()):
                record = self.config_func.db.get_latest_record(param_name, run_id)
                if record is not None:
                    last_known_values[param_name] = record.value
                    break

        # Clear current state
        self.selected_params.clear()
        self.current_options.clear()
        logger.debug("Cleared current state")

        # Restore the selections, preferring current values over last known values
        self.selected_params.update(last_known_values)
        self.selected_params.update(current_selections)
        logger.debug(f"Restored selections: {dict(self.selected_params)}")

        # Run the configuration
        try:
            self.config_func(values=self.selected_params)
            logger.debug("Configuration function executed successfully")
        except Exception as e:
            logger.error(f"Error running config: {str(e)}", exc_info=True)
            return

        # Get the latest configuration state
        latest_run_id = self.config_func.db.get_run_ids()[-1]
        latest_records = self.config_func.db.get_records(latest_run_id)
        logger.debug(f"Latest records from DB: {latest_records}")

        # Update state based on configuration results, maintaining order from records
        for record in latest_records.values():
            if record.parameter_type == "select":
                self.current_options[record.name] = set(record.options)
                # Only update value if not already set
                if record.name not in self.selected_params:
                    self.selected_params[record.name] = record.value
                logger.debug(f"Updated parameter '{record.name}':")
                logger.debug(f"  Options: {self.current_options[record.name]}")
                logger.debug(f"  Selected value: {self.selected_params[record.name]}")

        logger.info("Configuration run completed")
        logger.debug(f"Final state - Options: {dict(self.current_options)}")
        logger.debug(f"Final state - Selected: {dict(self.selected_params)}")

    def get_current_parameters(self) -> Dict[str, ParameterState]:
        """
        Retrieve the current parameters and their states.

        Returns:
            Dict[str, ParameterState]: A dictionary of parameter names to their states.
        """
        return {
            name: ParameterState(name=name, current_value=value, available_options=options)
            for name, options in self.current_options.items()
            for value in [self.selected_params.get(name)]
        }


class AbstractValueHandler(ABC):
    @abstractmethod
    def display_ui(self, parameters: Dict[str, ParameterState]) -> None:
        """Display the UI components for parameter selection."""
        pass

    @abstractmethod
    def get_user_selections(self) -> Dict[str, Any]:
        """Retrieve the user's selections from the UI."""
        pass

    @abstractmethod
    def update_ui(self, updated_parameters: Dict[str, ParameterState]) -> None:
        """Update the UI based on new parameter options or defaults."""
        pass


class IPyWidgetsUI(AbstractValueHandler):
    """UI handler using IPyWidgets."""

    def __init__(self, selection_handler: SelectionHandler) -> None:
        self.selection_handler = selection_handler
        self.widgets: Dict[str, widgets.Dropdown] = {}
        self.output = widgets.Output()
        self.container = widgets.VBox([])
        logger.info("Initialized IPyWidgetsUI")
        self.create_widgets()

    def create_widgets(self) -> None:
        """Create widgets for all current parameters."""
        logger.info("Creating widgets")
        parameters = self.selection_handler.get_current_parameters()
        logger.debug(f"Parameters to create widgets for: {parameters}")

        for param in parameters.values():
            logger.debug(f"Creating widget for parameter '{param.name}':")
            logger.debug(f"  Options: {param.available_options}")
            logger.debug(f"  Current value: {param.current_value}")

            widget = widgets.Dropdown(
                options=sorted(param.available_options),
                value=param.current_value,
                description=param.name,
                style={"description_width": "initial"},
            )

            widget.observe(lambda change, name=param.name: self._handle_change(name, change["new"]), names="value")

            self.widgets[param.name] = widget
            logger.debug(f"Created widget for '{param.name}'")

        self.container.children = list(self.widgets.values()) + [self.output]
        logger.info(f"Created {len(self.widgets)} widgets")

    def _handle_change(self, param_name: str, new_value: Any) -> None:
        """
        Handle changes in widget values.

        Args:
            param_name (str): The name of the parameter that changed.
            new_value (Any): The new value of the parameter.
        """
        logger.info(f"Handling change for parameter '{param_name}' to '{new_value}'")
        with self.output:
            try:
                self.selection_handler.update_param(param_name, new_value)
                self.update_ui(self.selection_handler.get_current_parameters())
            except Exception as e:
                logger.error(f"Error handling change: {str(e)}", exc_info=True)
                self.output.append_stdout(f"Error: {str(e)}\n")

    def update_widgets(self) -> None:
        """Update widgets based on the current state."""
        logger.info("Updating widgets")
        parameters = self.selection_handler.get_current_parameters()
        current_param_names = set(parameters.keys())
        existing_param_names = set(self.widgets.keys())

        # Remove widgets that are no longer needed
        for param_name in existing_param_names - current_param_names:
            logger.debug(f"Removing obsolete widget for '{param_name}'")
            self.widgets.pop(param_name)

        # Get the order from the latest run
        latest_run_id = self.selection_handler.config_func.db.get_run_ids()[-1]
        latest_records = self.selection_handler.config_func.db.get_records(latest_run_id)
        ordered_params = list(latest_records.keys())

        # Update or create widgets in the order from the database
        for param_name in ordered_params:
            if param_name in parameters:
                param_state = parameters[param_name]
                if param_name in self.widgets:
                    self._update_widget_values(self.widgets[param_name], param_state)
                else:
                    self._create_widget(param_state)

        # Update the container with widgets in the correct order
        ordered_widgets = [self.widgets[param_name] for param_name in ordered_params if param_name in self.widgets]
        self.container.children = ordered_widgets + [self.output]
        logger.info("Widget update completed")

    def _update_widget_values(self, widget: widgets.Dropdown, param_state: ParameterState) -> None:
        """Update an existing widget's options and value."""
        widget.options = sorted(param_state.available_options)
        if widget.value not in param_state.available_options:
            widget.value = param_state.current_value
        logger.debug(f"Updated widget for '{param_state.name}'")

    def _create_widget(self, param_state: ParameterState) -> None:
        """Create a new widget for a parameter."""
        widget = widgets.Dropdown(
            options=sorted(param_state.available_options),
            value=param_state.current_value,
            description=param_state.name,
            style={"description_width": "initial"},
        )

        widget.observe(lambda change, name=param_state.name: self._handle_change(name, change["new"]), names="value")

        self.widgets[param_state.name] = widget
        logger.debug(f"Created new widget for '{param_state.name}'")

    def display_ui(self) -> None:
        """Display the configuration UI."""
        # Apply custom CSS for styling
        css = """
        <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }
        </style>
        """
        display(HTML(css))

        # Display the widget container
        display(self.container)

    def get_user_selections(self) -> Dict[str, Any]:
        """Retrieve the user's selections from the UI."""
        return {name: widget.value for name, widget in self.widgets.items()}

    def update_ui(self, updated_parameters: Dict[str, ParameterState]) -> None:
        """
        Update the UI based on new parameter options or defaults.

        Args:
            updated_parameters (Dict[str, ParameterState]): The updated parameters.
        """
        self.update_widgets()


def create_interactive_config(config_func: Callable) -> Tuple[SelectionHandler, IPyWidgetsUI]:
    """
    Initialize the interactive configuration UI.

    Args:
        config_func (Callable): The configuration function to manage parameters.

    Returns:
        Tuple[SelectionHandler, IPyWidgetsUI]: Instances of the handler and UI.
    """
    logger.info("Creating interactive configuration")

    # Create and initialize handler
    handler = SelectionHandler(config_func)
    logger.info("Created SelectionHandler")

    # Run initial configuration to populate parameters
    handler._run_config()
    logger.debug(f"Initial parameters: {handler.get_current_parameters()}")

    # Create and initialize UI
    ui = IPyWidgetsUI(handler)
    logger.info("Created IPyWidgetsUI")

    # Display the UI
    ui.display_ui()
    logger.info("UI displayed")

    return handler, ui
