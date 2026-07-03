"""Structured outputs for every agent. Field descriptions double as LLM
instructions under with_structured_output — keep them imperative and exact."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Intensity = Literal["FIERCE", "STRONG", "MODERATE", "MILD"]
Standing = Literal["AHEAD", "INLINE", "BEHIND"]
Sentiment = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
Significance = Literal["HIGH", "MEDIUM", "LOW"]
EventType = Literal[
    "earnings", "M&A", "SEBI_action", "dividend",
    "pledging", "split", "order_win", "other",
]
Tone = Literal["confident", "cautious", "defensive"]


class PeerComparison(BaseModel):
    """One competitor of the target stock, with comparison verdicts."""

    ticker: str = Field(
        description="Peer's yfinance ticker with exchange suffix",
        examples=["ICICIBANK.NS"])
    name: str = Field(
        description="Peer's full company name",
        examples=["ICICI Bank Ltd"])
    reason_for_inclusion: str = Field(
        description="Why this company is a true competitor: overlapping business lines, same customer base, similar scale",
        examples=["Second-largest private bank; competes directly in retail lending, deposits and cards with a comparable branch network"])
    competition_intensity: Intensity = Field(
        description="One word: how hard this peer competes with the target. FIERCE = head-on in core segments, STRONG = competes directly across most segments, MODERATE = meaningful overlap in some segments, MILD = marginal overlap",
        examples=["FIERCE"])
    target_standing: Standing = Field(
        description="One word: target stock vs THIS peer on combined recent returns + fundamentals. AHEAD / INLINE / BEHIND",
        examples=["INLINE"])
    metrics: dict[str, Optional[float]] = Field(
        description="Tool-fetched numbers for this peer — passed through, never invented. Keys: pe, roe, debt_equity, mktcap_cr, ret_1m, ret_3m, ret_6m",
        examples=[{"pe": 17.8, "roe": 18.4, "debt_equity": None,
                   "mktcap_cr": 895000.0, "ret_1m": 2.1, "ret_3m": 8.9,
                   "ret_6m": 12.4}])


class CompetitorOutput(BaseModel):
    """Competitor Intelligence agent result."""

    peers: list[PeerComparison] = Field(
        min_length=3, max_length=5,
        description="3-5 true competitors, best match first")
    comparison_summary: str = Field(
        description="Competitive-landscape narrative: where the target leads, lags, and why. Cites metrics from the peer table",
        examples=["HDFC Bank trades at a premium P/E to ICICI and Axis but has lagged both on 6-month returns as merger-related deposit costs compress margins."])
    overall_standing: Standing = Field(
        description="One word: target vs the whole peer set overall. AHEAD = leads the peer set, INLINE = roughly in line with the peer set, BEHIND = lags the peer set",
        examples=["AHEAD"])


class NewsItem(BaseModel):
    """One analyzed news article about the target stock."""

    title: str = Field(
        description="Simple, concise version of the article headline",
        examples=["HDFC Bank Q3 net profit rises 2.2% to Rs 16,736 crore"])
    date: str = Field(
        description="Article publish date, YYYY-MM-DD",
        examples=["2026-07-01"])
    source_url: str = Field(
        description="Link to the original article",
        examples=["https://economictimes.indiatimes.com/example"])
    summary: str = Field(
        description="2-3 sentences: what the article actually says",
        examples=["The bank reported Q3 net profit of Rs 16,736 crore, up 2.2% YoY, slightly below street estimates. NIM held at 3.4%."])
    impact_on_stock: str = Field(
        description="One line: what this news means for the target ticker",
        examples=["Margin pressure may cap near-term upside despite stable asset quality"])
    sector_impact: str = Field(
        description="One line: what this news signals for the sector",
        examples=["Private banks broadly face deposit-cost pressure this quarter"])
    sentiment: Sentiment = Field(
        description="Sentiment toward the TARGET stock, not the market. POSITIVE = favorable for the stock, NEGATIVE = unfavorable for the stock, NEUTRAL = no clear directional signal",
        examples=["NEUTRAL"])
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="-1.0 strongly negative to +1.0 strongly positive toward the target stock; must agree with `sentiment`",
        examples=[0.1])


class NewsOutput(BaseModel):
    """News Analysis agent result."""

    items: list[NewsItem] = Field(
        description="One entry per relevant article. Empty list allowed when the fetch returned nothing relevant")
    narrative: str = Field(
        description="Overall story of the last 48h of news for the ticker, citing articles by title. If items is empty, states that no relevant recent news was found",
        examples=["Coverage centred on the Q3 print: profit growth slowed while asset quality held steady."])
    overall_sentiment: Sentiment = Field(
        description="Aggregate sentiment across items, weighted by relevance",
        examples=["NEUTRAL"])


class TimelineEvent(BaseModel):
    """One corporate event on the timeline."""

    date: str = Field(
        description="Event/filing date, YYYY-MM-DD",
        examples=["2026-05-14"])
    type: EventType = Field(
        description="Event class",
        examples=["dividend"])
    significance: Significance = Field(
        description="How much this event matters for the stock",
        examples=["HIGH"])
    summary: str = Field(
        description="One line: what happened",
        examples=["Board declared final dividend of Rs 22/share for FY26"])
    what_it_meant: str = Field(
        description="Interpretation: why the company did this / what it signals",
        examples=["Signals confident capital position after two quarters of elevated provisioning"])
    how_it_affected: str = Field(
        description="Effect on the company/stock, referencing the tool-computed price moves when available",
        examples=["Stock rose 1.8% next session; yield support at current levels"])
    price_move_1d: Optional[float] = Field(
        description="Tool-computed % move, event close vs previous close. Passed through from price_move_around, never invented",
        examples=[1.8])
    price_move_5d: Optional[float] = Field(
        description="Tool-computed % move over the 5 trading days after the event",
        examples=[3.2])
    filing_ref: Optional[str] = Field(
        description="BSE/NSE filing identifier or announcement title from screener",
        examples=["BSE Ann. 2026-05-14: Outcome of Board Meeting"])


class EventOutput(BaseModel):
    """Event Timeline agent result."""

    events: list[TimelineEvent] = Field(
        description="Chronological (oldest first) timeline from ~90 days of announcements. Empty allowed if no announcements found")
    highlights: list[str] = Field(
        max_length=3,
        description="Top up-to-3 most significant events, one line each",
        examples=[["Rs 22/share final dividend declared (14 May)",
                   "CFO resignation announced (02 Jun)"]])


class GuidanceItem(BaseModel):
    """One forward-looking management statement."""

    metric: str = Field(
        description="What is being guided",
        examples=["credit growth"])
    value: str = Field(
        description="The guided value/range as management stated it",
        examples=["17-18% YoY"])
    period: str = Field(
        description="Period the guidance applies to",
        examples=["FY27"])
    source: str = Field(
        description="document type + date",
        examples=["Q4 FY26 concall, 2026-04-19"])
    quote: str = Field(
        min_length=20,
        description="Exact supporting quote from the document",
        examples=["We remain confident of delivering credit growth in the range of 17 to 18 percent for the full year"])


class RiskItem(BaseModel):
    """One risk flagged by management or the annual report."""

    risk: str = Field(
        description="The risk, one line",
        examples=["Deposit repricing lag compressing NIM through H1"])
    source: str = Field(
        description="document type + date",
        examples=["Annual Report FY26, MD&A"])
    quote: str = Field(
        min_length=20,
        description="Exact supporting quote",
        examples=["continued upward repricing of the deposit base is expected to weigh on margins in the near term"])


class DocsOutput(BaseModel):
    """Financial Docs Analyzer result — 3 yrs of concalls + latest annual report."""

    guidance: list[GuidanceItem] = Field(
        description="Management guidance found across the documents")
    risks: list[RiskItem] = Field(
        description="Risks flagged by management")
    strategy_highlights: list[str] = Field(
        description="Key strategic priorities, one line each",
        examples=[["Branch expansion in semi-urban markets",
                   "Merger synergy capture in home loans"]])
    management_tone: Tone = Field(
        description="Overall tone in the most recent documents",
        examples=["confident"])
    tone_trend: str = Field(
        description="How tone shifted across the ~3 years of concalls",
        examples=["Progressively more confident since FY25 as merger integration concerns faded"])
    narrative: str = Field(
        description="Synthesis of what official documents say about where the company is heading; every claim cites source + date")


class FundamentalsOutput(BaseModel):
    """Fundamentals agent result. All numeric dicts are tool-fetched, passed
    through — the LLM writes only `summary`."""

    company_profile: dict = Field(
        description="From get_stock_info: sector, industry, business description",
        examples=[{"sector": "Financial Services",
                   "industry": "Banks - Private",
                   "description": "HDFC Bank provides retail and wholesale banking services across India."}])
    valuation: dict[str, Optional[float]] = Field(
        description="Keys: pe, pb, roe, roce, debt_equity, dividend_yield. None when yfinance lacks the number",
        examples=[{"pe": 19.2, "pb": 2.8, "roe": 16.9, "roce": None,
                   "debt_equity": None, "dividend_yield": 1.1}])
    price_snapshot: dict[str, Optional[float]] = Field(
        description="Keys: price, high_52w, low_52w, ret_1m, ret_6m, ret_1y, mktcap_cr",
        examples=[{"price": 1712.5, "high_52w": 1794.0, "low_52w": 1363.5,
                   "ret_1m": 2.4, "ret_6m": 9.8, "ret_1y": 18.2,
                   "mktcap_cr": 1305000.0}])
    shareholding: dict[str, Optional[float]] = Field(
        description="From fetch_shareholding: promoter, fii, dii, public percentages for the latest quarter",
        examples=[{"promoter": 0.0, "fii": 47.8, "dii": 35.2, "public": 17.0}])
    summary: str = Field(
        description="Quick plain-language summary of the company and the stock's current state, grounded ONLY in the numbers above",
        examples=["India's largest private bank by market cap. Trades near the top of its 52-week range after an 18% one-year run."])


class ReportOutput(BaseModel):
    """Synthesis agent result — the final research report."""

    exec_summary: str = Field(
        description="3-5 sentence executive summary across all sections",
        examples=["HDFC Bank enters FY27 with steady profit growth, a premium valuation versus peers, and management guiding 17-18% credit growth."])
    sections: dict[str, str] = Field(
        description="Markdown body per section. Keys exactly: fundamentals, competitors, events, news, docs. Failed sections carry a one-line 'data unavailable' note",
        examples=[{"fundamentals": "## Company & Fundamentals\n...",
                   "competitors": "## Competitive Landscape\n..."}])
    sources: list[str] = Field(
        description="Deduplicated source references carried over from specialist outputs",
        examples=[["Q4 FY26 concall, 2026-04-19",
                   "https://economictimes.indiatimes.com/example"]])
    missing_sections: list[str] = Field(
        description="Agent names whose branch ended failed_partial",
        examples=[["news"]])


class ChatbotResponse(BaseModel):
    """Research chatbot single-turn result."""

    answer: str = Field(
        description="Guardrail-cleaned answer text with inline citations")
    sources_used: list[str] = Field(
        default_factory=list,
        description="Names of tools invoked during this turn")
    charts: list[str] = Field(
        default_factory=list,
        description="File paths of charts generated during this turn")
