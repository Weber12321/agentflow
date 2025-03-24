from .base import BaseAgent
from state.state import State


class Planner(BaseAgent):
    def __init__(self, llm_model="gpt-3.5-turbo", temperature=0.7, tools=None):
        super().__init__()
        self.llm_model = llm_model
        self.temperature = temperature
        self.tools = tools or []  # Ensure tools is a list

    def run(self, state: State) -> State:
        prompt = state.data["prompt"]
        # In a real application, you'd use an actual LLM here
        plan = f"Plan for: {prompt}\n1. Do something.\n2. Do something else."
        print(f"Planner: {plan}")
        state.data["plan"] = plan
        return state