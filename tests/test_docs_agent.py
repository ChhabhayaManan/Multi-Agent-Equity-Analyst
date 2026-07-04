import pytest

from templates.schemas.outputs import DocsOutput, GuidanceItem, RiskItem

CANNED = DocsOutput(
    guidance=[GuidanceItem(metric="credit growth", value="17-18%", period="FY27",
                           source="Q4 FY26 concall, May 2026",
                           quote="we expect credit growth of 17 to 18 percent")],
    risks=[RiskItem(risk="NIM compression", source="AR FY26",
                    quote="margins are expected to remain under pressure")],
    strategy_highlights=["branch expansion"], management_tone="confident",
    tone_trend="steadier", narrative="Management guides to strong growth.")


@pytest.fixture
def patched(monkeypatch):
    from agents import financial_docs_analyzer as mod
    calls = {"indexed": [], "queries": []}
    transcripts = [{"date": f"Q{i}", "url": f"https://x/t{i}.pdf"} for i in range(15)]
    monkeypatch.setattr(mod, "fetch_concall_transcripts", lambda t: transcripts)
    monkeypatch.setattr(mod, "fetch_annual_report_url", lambda t: "https://x/ar.pdf")
    monkeypatch.setattr(mod, "index_pdf_document",
                        lambda url, t, st, meta=None: calls["indexed"].append(url))
    monkeypatch.setattr(mod, "wait_for_vectors", lambda t, timeout_s=90: True)
    monkeypatch.setattr(mod, "query_pinecone",
                        lambda t, q, st, k=5: (calls["queries"].append(q) or
                                               [{"text": f"passage for {q}", "score": 0.9,
                                                 "metadata": {}}]))

    class FakeLLM:
        def invoke(self, prompt):
            return CANNED

    monkeypatch.setattr(mod, "get_llm", lambda schema=None: FakeLLM())
    return mod, calls


def test_caps_at_12_transcripts_plus_annual_report(patched):
    mod, calls = patched
    out, fetch_count = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert fetch_count == 13                       # 12 transcripts + 1 annual report
    assert len(calls["indexed"]) == 13
    assert "https://x/ar.pdf" in calls["indexed"]
    assert len(calls["queries"]) == 5              # five fixed research queries
    assert out.management_tone == "confident"


def test_survives_individual_pdf_failures(patched, monkeypatch):
    mod, calls = patched

    def flaky(url, t, st, meta=None):
        if url.endswith("t0.pdf"):
            raise ValueError("bad pdf")
        calls["indexed"].append(url)

    monkeypatch.setattr(mod, "index_pdf_document", flaky)
    _, fetch_count = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert fetch_count == 12                       # one transcript failed


def test_no_annual_report_is_nonfatal(patched, monkeypatch):
    mod, _ = patched

    def boom(t):
        raise ValueError("No annual reports panel")

    monkeypatch.setattr(mod, "fetch_annual_report_url", boom)
    _, fetch_count = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert fetch_count == 12
