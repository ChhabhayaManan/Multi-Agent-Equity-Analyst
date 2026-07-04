"""Conditional routing after each validate node."""

from workflow.state import GraphState


def make_router(name: str):
    def route(state: GraphState) -> str:
        status = state["runs"][name]["status"]
        return "join" if status in ("passed", "failed_partial") else "retry"

    return route
