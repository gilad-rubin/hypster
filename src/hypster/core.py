from typing import Any

class Select:
    """Represents a selectable option in the configuration."""
    def __init__(self, name: str):
        self.name = name

def prep(obj: Any) -> Any:
    """Prepares an object for use in the configuration system."""
    return obj