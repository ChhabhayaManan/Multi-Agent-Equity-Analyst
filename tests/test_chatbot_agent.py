import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import chatbot.chatbot_agent as ca
from guard.result import GuardrailResult


class FakeAgent:
    """Mimics create_agent's compiled graph: returns preset messages."""

    def __init__(self, messages):
        self._messages = messages
        self.calls = []

    async def ainvoke(self, payload, config=None):
        self.calls.append((payload, config))
        return {"messages": list(payload["messages"]) + self._messages}


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(ca, "namespace_exists", lambda ticker: True)
    monkeypatch.setattr(ca, "build_local_tools", lambda t, c: [])
    monkeypatch.setattr(ca, "load_alphavantage_tools", lambda: [])
    monkeypatch.setattr(ca, "_create_agent", lambda system_prompt, tools: None)
    monkeypatch.setattr(ca, "summarize_turn", lambda q, a: f"sum({q[:20]})")
    monkeypatch.setattr(ca.InputGuardrail, "validate",
                        lambda self, msg, t, c, h: GuardrailResult(True, msg))
    monkeypatch.setattr(ca.OutputGuardrail, "validate",
                        lambda self, text, chunks: GuardrailResult(True, text))
    return ca.ChatSession("HDFCBANK.NS", "HDFC Bank Ltd")


def test_namespace_precondition(monkeypatch):
    monkeypatch.setattr(ca, "namespace_exists", lambda ticker: False)
    with pytest.raises(ValueError):
        ca.ChatSession("UNRESEARCHED.NS", "Nobody Ltd")


def test_blocked_input_short_circuits(monkeypatch, session):
    monkeypatch.setattr(
        ca.InputGuardrail, "validate",
        lambda self, msg, t, c, h: GuardrailResult(
            False, "canned-block", ["advice_request"], "asked to buy"))
    invoked = []
    session._agent = FakeAgent([])
    session._agent.calls = invoked

    resp = session.ask("should I buy?")

    assert resp.answer == "canned-block"
    assert invoked == []                                   # agent never ran
    assert session.memory.turns[-1].status == "blocked"
    assert "[blocked: advice_request]" in session.memory.turns[-1].summary


def test_happy_turn_grounds_on_tool_outputs(monkeypatch, session):
    captured = {}

    def fake_out_validate(self, text, chunks):
        captured["chunks"] = chunks
        return GuardrailResult(True, text)

    monkeypatch.setattr(ca.OutputGuardrail, "validate", fake_out_validate)
    session._agent = FakeAgent([
        AIMessage(content="", tool_calls=[
            {"name": "get_live_price", "args": {"ticker": "TCS.NS"}, "id": "1"}]),
        ToolMessage(content='{"ticker": "TCS.NS", "price": 4120.0}',
                    tool_call_id="1"),
        AIMessage(content="TCS trades at 4120.0 [live: yfinance]."),
    ])

    resp = session.ask("What is TCS price?")

    assert resp.answer == "TCS trades at 4120.0 [live: yfinance]."
    assert resp.sources_used == ["get_live_price"]
    assert '{"ticker": "TCS.NS", "price": 4120.0}' in captured["chunks"]
    assert session.memory.turns[-1].status == "ok"


def test_chart_paths_extracted(session):
    session._agent = FakeAgent([
        AIMessage(content="", tool_calls=[
            {"name": "plot_price_chart", "args": {"ticker": "X"}, "id": "1"}]),
        ToolMessage(content='{"chart_path": "data/charts/X_6mo_1.html", "last_close": 1.0}',
                    tool_call_id="1"),
        AIMessage(content="Here is the chart."),
    ])
    resp = session.ask("plot it")
    assert resp.charts == ["data/charts/X_6mo_1.html"]


def test_output_guardrail_replaces_answer(monkeypatch, session):
    monkeypatch.setattr(
        ca.OutputGuardrail, "validate",
        lambda self, text, chunks: GuardrailResult(
            False, "stripped-canned", ["ungrounded"], "all stripped"))
    session._agent = FakeAgent([AIMessage(content="hallucinated stuff")])
    resp = session.ask("q")
    assert resp.answer == "stripped-canned"


def test_agent_error_survives(monkeypatch, session):
    class BoomAgent:
        async def ainvoke(self, payload, config=None):
            raise RuntimeError("provider down")

    session._agent = BoomAgent()
    resp = session.ask("q")
    assert resp.answer == ca.ERROR_MESSAGE
    assert session.memory.turns[-1].status == "error"


def test_status_callback_fires_phases_and_tools(monkeypatch, session):
    lines = []
    session._status = lines.append
    session._callback_handler = ca.StatusCallbackHandler(lines.append, set())
    session._agent = FakeAgent([AIMessage(content="fine")])

    session.ask("q")

    assert ca.STATUS_LINES["input_guard"] in lines
    assert ca.STATUS_LINES["thinking"] in lines
    assert ca.STATUS_LINES["output_guard"] in lines


def test_status_handler_tool_lines():
    lines = []
    handler = ca.StatusCallbackHandler(lines.append, av_tool_names={"EARNINGS"})
    handler.on_tool_start({"name": "get_live_price"}, "")
    handler.on_tool_start({"name": "EARNINGS"}, "")
    handler.on_tool_start({"name": "mystery_tool"}, "")
    assert lines == [ca.STATUS_LINES["get_live_price"], ca.AV_LINE,
                     ca.DEFAULT_TOOL_LINE]


def test_ask_records_contexts_and_pass_outcome(monkeypatch, session):
    captured = {}
    monkeypatch.setattr(ca, "set_run_metadata", lambda md: captured.update(md))
    session._agent = FakeAgent([
        AIMessage(content="", tool_calls=[
            {"name": "get_live_price", "args": {"ticker": "TCS.NS"}, "id": "1"}]),
        ToolMessage(content='{"ticker":"TCS.NS","price":10}', tool_call_id="1"),
        AIMessage(content="Price is 10 [live: yfinance]"),
    ])

    resp = session.ask("what is the price?")

    assert resp.retrieved_contexts
    assert '{"ticker":"TCS.NS","price":10}' in resp.retrieved_contexts
    assert captured["run_type"] == "chatbot_turn"
    assert captured["guardrail_outcome"] in {"pass", "fixed"}


def test_ask_records_blocked_outcome(monkeypatch, session):
    captured = {}
    monkeypatch.setattr(ca, "set_run_metadata", lambda md: captured.update(md))
    monkeypatch.setattr(
        ca.InputGuardrail, "validate",
        lambda self, msg, t, c, h: GuardrailResult(
            False, "canned", ["advice_request"], "asked to buy"))

    session.ask("should I buy?")

    assert captured["guardrail_outcome"] == "blocked"
