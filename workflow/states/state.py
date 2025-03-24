from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class State:
    """
    Represents the shared state of the agent flow.

    This class uses a dataclass for concise definition and automatic
    generation of methods like __init__, __repr__, etc.
    """
    data: Dict[str, Any] = field(default_factory=dict)
    # Example of other typed fields:
    current_task: Optional[str] = field(default=None)
    plan: Optional[str] = field(default=None)
    results: list = field(default_factory=list)  # Example of a list
    messages: list = field(default_factory=list)  # For conversation history
    # Add other state variables as needed, with appropriate types.

    # --- Optional: Helper methods for accessing/modifying state ---
    # These can improve code readability and encapsulate state updates

    def get_data(self, key: str, default: Any = None) -> Any:
        """Safely gets a value from the data dictionary."""
        return self.data.get(key, default)

    def set_data(self, key: str, value: Any):
        """Sets a value in the data dictionary."""
        self.data[key] = value

    def add_message(self, message: str):
        """Adds a message to the messages list."""
        self.messages.append(message)

    # You can add more specific methods as needed.  For example:
    # def update_plan(self, new_plan: str):
    #    self.plan = new_plan