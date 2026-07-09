from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    """Common input every specialist agent receives."""

    ticker: str = Field(
        description="yfinance ticker with exchange suffix",
        examples=["HDFCBANK.NS"])
    company_name: str = Field(
        description="Full company name as listed",
        examples=["HDFC Bank Ltd"])
