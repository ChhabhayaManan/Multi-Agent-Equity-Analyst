"""Research chatbot session: input guardrail -> create_agent ReAct loop ->
output guardrail (grounded on this turn's tool outputs) -> turn summary."""

import asyncio
import json
from typing import Callable, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from chatbot.chatbot_tools import build_local_tools, load_alphavantage_tools
from chatbot.memory import ConversationMemory, summarize_turn
from guard.input_guardrail import InputGuardrail
from guard.output_guardrail import OutputGuardrail
from templates.prompts.chatbot import CHATBOT_SYSTEM
from templates.schemas.outputs import ChatbotResponse
from tools.pinecone_tools import namespace_exists
from utils.helpers import get_logger
from utils.llm import get_chat_model
from utils.tracing import set_run_metadata, traceable

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 8
ERROR_MESSAGE = "Something went wrong while answering. Please try again."

STATUS_LINES = {
    "input_guard": "Running your question past the compliance bouncer...",
    "thinking": "Consulting the research vault...",
    "output_guard": "Fact-checking every sentence...",
    "search_research": "Digging through the research archives...",
    "get_live_price": "Pinging Dalal Street for a live quote...",
    "get_stock_info": "Pulling the company dossier...",
    "get_fundamentals": "Crunching the fundamentals...",
    "get_price_history": "Rewinding the price tape...",
    "price_move_around": "Zooming into that date on the charts...",
    "resolve_ticker": "Looking up that name on the exchange...",
    "get_recent_news": "Hot off the press — grabbing headlines...",
    "plot_price_chart": "Painting you a chart...",
    "plot_comparison_chart": "Lining up the contenders on one canvas...",
}
AV_LINE = "Summoning Alpha Vantage intelligence..."
DEFAULT_TOOL_LINE = "Fetching data..."


class StatusCallbackHandler(BaseCallbackHandler):
    """Emits a witty status line whenever the agent starts a tool call."""

    def __init__(self, emit: Callable[[str], None], av_tool_names: set):
        self._emit = emit
        self._av = av_tool_names

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "")
        if name in self._av:
            self._emit(AV_LINE)
        else:
            self._emit(STATUS_LINES.get(name, DEFAULT_TOOL_LINE))


def _create_agent(system_prompt: str, tools: list):
    from langchain.agents import create_agent
    return create_agent(model=get_chat_model(), tools=tools,
                        system_prompt=system_prompt)


class ChatSession:
    def __init__(self, ticker: str, company_name: str,
                 status_callback: Optional[Callable[[str], None]] = None):
        if not namespace_exists(ticker):
            raise ValueError(
                f"No indexed research for {ticker}. Generate its research "
                "report first, then start the chatbot.")
        self.ticker = ticker
        self.company_name = company_name
        self.memory = ConversationMemory()
        self._status = status_callback or (lambda line: None)
        self._input_guard = InputGuardrail()
        self._output_guard = OutputGuardrail()

        local_tools = build_local_tools(ticker, company_name)
        av_tools = load_alphavantage_tools()
        self._callback_handler = StatusCallbackHandler(
            self._status, {t.name for t in av_tools})
        system = CHATBOT_SYSTEM.format(ticker=ticker, company_name=company_name)
        self._agent = _create_agent(system, local_tools + av_tools)

    @traceable(run_type="chain", name="chatbot_turn")
    def ask(self, user_message: str) -> ChatbotResponse:
        self._status(STATUS_LINES["input_guard"])
        gate = self._input_guard.validate(
            user_message, self.ticker, self.company_name,
            self.memory.history_text())
        if not gate.passed:
            tag = gate.violations[0] if gate.violations else "blocked"
            self.memory.add_turn(user_message, gate.cleaned_text,
                                 f"[blocked: {tag}]", status="blocked")
            set_run_metadata({"ticker": self.ticker,
                              "run_type": "chatbot_turn",
                              "guardrail_outcome": "blocked"})
            return ChatbotResponse(answer=gate.cleaned_text)

        self._status(STATUS_LINES["thinking"])
        try:
            result = asyncio.run(self._agent.ainvoke(
                {"messages": [*self.memory.context_messages(),
                              ("user", user_message)]},
                config={"callbacks": [self._callback_handler],
                        "recursion_limit": 2 * MAX_TOOL_ROUNDS + 1},
            ))
        except Exception:
            logger.exception("chatbot agent invoke failed")
            self.memory.add_turn(user_message, ERROR_MESSAGE,
                                 "[error: agent failure]", status="error")
            set_run_metadata({"ticker": self.ticker,
                              "run_type": "chatbot_turn",
                              "guardrail_outcome": "error"})
            return ChatbotResponse(answer=ERROR_MESSAGE)

        messages = result["messages"]
        final_text = messages[-1].content if messages else ""
        if isinstance(final_text, list):  # some providers return content blocks
            final_text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in final_text)

        grounding, charts = [], []
        for m in messages:
            if isinstance(m, ToolMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)
                grounding.append(content)
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and data.get("chart_path"):
                        charts.append(data["chart_path"])
                except ValueError:
                    pass

        sources: List[str] = []
        for m in messages:
            for tc in getattr(m, "tool_calls", None) or []:
                if tc["name"] not in sources:
                    sources.append(tc["name"])

        self._status(STATUS_LINES["output_guard"])
        verdict = self._output_guard.validate(final_text, grounding)
        answer = verdict.cleaned_text

        summary = summarize_turn(user_message, answer)
        self.memory.add_turn(user_message, answer, summary)
        outcome = "fixed" if verdict.violations else "pass"
        set_run_metadata({"ticker": self.ticker, "run_type": "chatbot_turn",
                          "guardrail_outcome": outcome})
        return ChatbotResponse(answer=answer, sources_used=sources,
                               charts=charts, retrieved_contexts=grounding)
