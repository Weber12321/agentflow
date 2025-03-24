from langgraph.graph import StateGraph, Node, END
from typing import Dict, Any, List, Callable, Optional
from states.state import State


class WorkflowManager:
    def __init__(self):
        self.agent_registry: Dict[str, type] = {}  # Registry for agent types
        self.tool_registry: Dict[str, type] = {}  # Registry for tool types
        self.condition_registry: Dict[str, Callable] = (
            {}
        )  # Registry for conditions
        # The compiled graph and the prompt.
        self.graph = None
        self.prompt = None

    def register_agent(self, agent_class: type):
        """Registers an agent class."""
        self.agent_registry[agent_class.__name__.lower()] = agent_class

    def register_tool(self, tool_class: type):
        """Registers a tool class."""
        self.tool_registry[tool_class.name.lower()] = tool_class

    def register_condition(self, condition_func: Callable, name: str):
        self.condition_registry[name] = condition_func

    def _create_agent_instance(
        self, agent_type: str, config: Dict[str, Any]
    ) -> "Agent":
        """Creates an instance of an agent based on type and config."""
        agent_class = self.agent_registry.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Handle tool instantiation (if specified in config)
        tools = []
        if "tools" in config:
            for tool_name in config["tools"]:
                tool_class = self.tool_registry.get(tool_name)
                if tool_class is None:
                    raise ValueError(f"Unknown tool: {tool_name}")
                tools.append(tool_class())  # Instantiate the tool
            config["tools"] = tools

        # Instantiate the agent, passing the config
        return agent_class(**config)

    def compile_graph(self):
        if self.graph:
            self.app = self.graph.compile()

    def build_graph_from_request(
        self, request_data: Dict[str, Any]
    ) -> StateGraph:
        """Builds a LangGraph from the API request data."""

        # 1.  State Definition (You might need to adjust this based on your needs)
        class DynamicState(State):  # Use a dynamic state
            data: Dict[str, Any] = {}  # store all the data.

            # Add other essential fields if needed
            def __init__(self, data):
                super().__init__()
                self.data = data

        # 2. Create the StateGraph
        workflow = StateGraph(DynamicState)

        # 3. Create Agent Nodes
        agent_instances = {}  # Store agent instances for later use (edges)
        for node_data in request_data["nodes"]:
            agent_id = node_data["id"]
            agent_type = node_data["agent_type"].lower()
            agent_config = node_data.get(
                "config", {}
            )  # Default to empty config

            agent_instance = self._create_agent_instance(
                agent_type, agent_config
            )
            agent_instances[agent_id] = agent_instance

            # Define the node's action (how it processes the state)
            def agent_node_action(
                state: DynamicState, agent=agent_instance
            ) -> Dict[str, Any]:
                next_state = agent.run(state)
                # Combine with the original state data by dict.
                return {"data": next_state.data}

            workflow.add_node(agent_id, agent_node_action)

        # 4. Create Edges
        for edge_data in request_data["edges"]:
            source_id = edge_data["source"]
            target_id = edge_data["target"]
            condition_name = edge_data.get("condition")

            if condition_name:
                # Conditional Edge
                condition_func = self.condition_registry.get(condition_name)
                if not condition_func:
                    raise ValueError(f"Unknown condition: {condition_name}")
                workflow.add_conditional_edges(
                    source_id,
                    target_id,
                    {
                        target_id: condition_func,
                        END: lambda state: not condition_func(
                            state
                        ),  # Important: Handle the "else" case
                    },
                )
            else:
                # Unconditional Edge
                workflow.add_edge(source_id, target_id)

        # 5. Set Entry Point
        workflow.set_entry_point(request_data["entry_point"])
        self.graph = workflow
        return workflow

    def run_graph(
        self,
        initial_state_data: Dict[str, Any],
        processes_output: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if not self.graph:
            raise Exception("No graph, please create the graph first.")

        if not self.app:
            self.compile_graph()
        # Initialize the state
        initial_state = {"data": initial_state_data}
        if not processes_output:
            processes_output = {}

        # Run the graph and processes the output
        for output in self.app.stream(initial_state, {"recursion_limit": 50}):
            for key, value in output.items():
                if key != "__end__":
                    processes_output[key] = value

        processes_output["final_output"] = output["__end__"]
        return processes_output
