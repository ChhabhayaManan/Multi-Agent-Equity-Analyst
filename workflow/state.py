"""Shared state schema for the workflow graph"""

from typing import Annotated, Literal, Optional, TypedDict

from templates.schemas.outputs import (
    CompetitorOutput, DocsOutput, EventOutput, FundamentalsOutput,
    NewsOutput, ReportOutput)

AGENTS = ("fundamentals", "competitor", "news", "events", "docs")


class AgentRun(TypedDict):
    status: Literal["pending", "running", "passed", "no_data", "failed_partial"]
    attempts: int                 # no. of current attempt
    failure_reasons: list[str]    # validator feedback
    fetch_count: int              # raw items fetched, 0 = source empty, -1 = unknown (agent error)


def new_run() -> AgentRun: 
    """start a new agent"""
    return {"status": "pending", "attempts": 0, "failure_reasons": [], "fetch_count": -1}


def merge_runs(left: dict, right: dict) -> dict:
    return {**left, **right}

#shared graph state 
class GraphState(TypedDict):
    ticker: str
    company_name: str

    fundamentals: Optional[FundamentalsOutput]
    competitor: Optional[CompetitorOutput]
    news: Optional[NewsOutput]
    events: Optional[EventOutput]
    docs: Optional[DocsOutput]

    runs: Annotated[dict[str, AgentRun], merge_runs] #merge runs defined as a collision policy for the graph state for "runs field"
    report: Optional[ReportOutput]
