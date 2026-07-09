"""Shared LangGraph state. GraphState IS the 'report state': all five
specialist outputs + run bookkeeping + the final report."""

from typing import Annotated, Literal, Optional, TypedDict

from templates.schemas.outputs import (
    CompetitorOutput, DocsOutput, EventOutput, FundamentalsOutput,
    NewsOutput, ReportOutput)

AGENTS = ("fundamentals", "competitor", "news", "events", "docs")


class AgentRun(TypedDict):
    status: Literal["pending", "running", "passed", "no_data", "failed_partial"]
    attempts: int                 # validator-level attempts (1..3)
    failure_reasons: list[str]    # validator feedback; injected on retry
    fetch_count: int              # raw items fetched; 0 = source empty, -1 = unknown (agent error)


def new_run() -> AgentRun:
    return {"status": "pending", "attempts": 0, "failure_reasons": [], "fetch_count": -1}


def merge_runs(left: dict, right: dict) -> dict:
    """Reducer: parallel branches update disjoint keys of `runs`."""
    return {**left, **right}


class GraphState(TypedDict):
    ticker: str
    company_name: str
    generated_at: str
    namespace_fresh: bool

    fundamentals: Optional[FundamentalsOutput]
    competitor: Optional[CompetitorOutput]
    news: Optional[NewsOutput]
    events: Optional[EventOutput]
    docs: Optional[DocsOutput]

    runs: Annotated[dict[str, AgentRun], merge_runs]
    report: Optional[ReportOutput]
