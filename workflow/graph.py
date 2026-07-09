"""Graph assembly: prepare -> 5 parallel validate-retry branches ->
synthesis (defer=True) -> END."""

from typing import Iterator

from langgraph.graph import END, START, StateGraph

from workflow.edges import make_router
from workflow.nodes import make_run_node, make_validate_node, prepare, synthesis
from workflow.state import AGENTS, GraphState


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("prepare", prepare)
    builder.add_node("synthesis", synthesis, defer=True)
    for name in AGENTS:
        builder.add_node(f"run_{name}", make_run_node(name))
        builder.add_node(f"validate_{name}", make_validate_node(name))
        builder.add_edge("prepare", f"run_{name}")
        builder.add_edge(f"run_{name}", f"validate_{name}")
        builder.add_conditional_edges(
            f"validate_{name}", make_router(name),
            {"retry": f"run_{name}", "join": "synthesis"})
    builder.add_edge(START, "prepare")
    builder.add_edge("synthesis", END)
    return builder.compile()


def _run_config(ticker: str) -> dict:
    return {"tags": [ticker],
            "metadata": {"run_type": "report", "ticker": ticker}}


def generate_report(ticker: str, company_name: str) -> dict:
    """Entry point for the frontend: returns the full GraphState."""
    graph = build_graph()
    return graph.invoke({"ticker": ticker, "company_name": company_name},
                        config=_run_config(ticker))


def stream_report(ticker: str, company_name: str) -> Iterator[dict]:
    """Streaming entry point for the frontend. Runs the graph with
    stream_mode='updates' and yields one dict per node update:
    {node, runs, report, done}. `runs` accumulates across yields; the final
    yield sets done=True and carries the completed report."""
    graph = build_graph()
    runs: dict = {}
    report = None
    last_node = None
    for update in graph.stream(
            {"ticker": ticker, "company_name": company_name},
            stream_mode="updates", config=_run_config(ticker)):
        for node, partial in update.items():
            last_node = node
            if partial and partial.get("runs"):
                runs = {**runs, **partial["runs"]}
            if partial and partial.get("report") is not None:
                report = partial["report"]
            yield {"node": node, "runs": runs, "report": report, "done": False}
    yield {"node": last_node, "runs": runs, "report": report, "done": True}


if __name__ == "__main__":
    import sys

    from utils.tracing import init_tracing
    init_tracing()

    if len(sys.argv) < 2:
        print("usage: python -m workflow.graph TICKER [COMPANY_NAME]")
        raise SystemExit(1)
    ticker = sys.argv[1]
    company = sys.argv[2] if len(sys.argv) > 2 else ticker
    state = generate_report(ticker, company)
    print("\n=== run statuses ===")
    for name, run in state["runs"].items():
        print(f"{name:14s} {run['status']:16s} attempts={run['attempts']} "
              f"fetch_count={run['fetch_count']}")
    report = state.get("report")
    if report:
        print("\n=== executive summary ===\n" + report.exec_summary)
        print("\nmissing sections:", report.missing_sections or "none")
    else:
        print("\nNO REPORT PRODUCED")
