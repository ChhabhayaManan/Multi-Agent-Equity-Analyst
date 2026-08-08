"""Conditional routing after each validate node."""

from workflow.state import GraphState

# router factory for the validate nodes: returns a function that takes the current state and returns
def make_router(name: str):
    def route(state: GraphState) -> str:
        status = state["runs"][name]["status"]
        return "join" if status in ("passed", "no_data", "failed_partial") else "retry"

    return route
