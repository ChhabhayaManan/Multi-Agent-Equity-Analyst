import pytest

from templates.schemas.outputs import EventOutput, TimelineEvent

ANNS = [
    {"date": "2026-05-14", "title": "Dividend declared", "pdf_url": "https://x/1.pdf"},
    {"date": "2026-06-02", "title": "Board meeting outcome", "pdf_url": "https://x/2.pdf"},
]

CANNED = EventOutput(
    events=[
        TimelineEvent(date="2026-05-14", type="dividend", significance="HIGH",
                      summary="Dividend declared", what_it_meant="Capital comfort",
                      how_it_affected="Stock firmed up",
                      price_move_1d=999.0, price_move_5d=999.0,  # must be overwritten
                      filing_ref="Dividend declared"),
        TimelineEvent(date="2026-06-02", type="other", significance="LOW",
                      summary="Board outcome", what_it_meant="Routine",
                      how_it_affected="No visible reaction",
                      price_move_1d=None, price_move_5d=None,
                      filing_ref="Board meeting outcome"),
    ],
    highlights=["Dividend declared (14 May)"])


@pytest.fixture
def patched(monkeypatch):
    from agents import event_timeline_creator as mod
    monkeypatch.setattr(mod, "fetch_bse_announcements", lambda t, days=90: list(ANNS))
    monkeypatch.setattr(mod, "price_move_around",
                        lambda t, d: {"pct_1d": 1.8, "pct_5d": 3.2} if d == "2026-05-14"
                        else {"pct_1d": -0.2, "pct_5d": None})
    monkeypatch.setattr(mod, "store_to_pinecone", lambda *a, **k: None)

    class FakeLLM:
        def invoke(self, prompt):
            return CANNED

    monkeypatch.setattr(mod, "get_llm", lambda schema=None: FakeLLM())
    return mod


def test_price_moves_overwritten_from_tools(patched):
    out, fetch_count = patched.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert fetch_count == 2
    assert out.events[0].price_move_1d == 1.8      # not the LLM's 999.0
    assert out.events[0].price_move_5d == 3.2
    assert out.events[1].price_move_1d == -0.2
    assert out.events[1].price_move_5d is None


def test_unknown_llm_date_gets_none_moves(patched, monkeypatch):
    from agents import event_timeline_creator as mod
    hallucinated = CANNED.model_copy(deep=True)
    hallucinated.events[0].date = "2026-01-01"     # no announcement on this date

    class FakeLLM:
        def invoke(self, prompt):
            return hallucinated

    monkeypatch.setattr(mod, "get_llm", lambda schema=None: FakeLLM())
    out, _ = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert out.events[0].price_move_1d is None and out.events[0].price_move_5d is None
