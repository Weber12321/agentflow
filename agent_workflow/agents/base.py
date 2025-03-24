from abc import ABC, abstractmethod
from states.state import State


class BaseAgent(ABC):

    @abstractmethod
    def run(self, state: State) -> State:
        raise NotImplementedError
