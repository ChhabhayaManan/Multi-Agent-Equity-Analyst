import importlib
import json

from templates.schemas.outputs import (CompetitorOutput, FundamentalsOutput,
                                       PeerComparison, ReportOutput)


def _report():
    return ReportOutput(
        exec_summary="Summary.",
        sources=["Q4 FY26 concall, 2026-04-19"],
        missing_sections=[])


def _specialists():
    return {
        "fundamentals": FundamentalsOutput(
            company_profile={"sector": "Banks"}, valuation={"pe": 19.2},
            price_snapshot={"price": 1712.5}, shareholding={"promoter": 25.0},
            summary="A large private bank near its 52-week high."),
        "competitor": CompetitorOutput(
            peers=[PeerComparison(ticker="ICICIBANK.NS", name="ICICI Bank",
                                  reason_for_inclusion="overlapping retail book",
                                  competition_intensity="FIERCE",
                                  target_standing="INLINE",
                                  metrics={"pe": 17.8, "roe": None})],
            comparison_summary="Trades at a premium to the peer set.",
            overall_standing="AHEAD"),
        "news": None, "events": None, "docs": None,
    }


def _runs():
    return {"fundamentals": {"status": "passed", "attempts": 1,
                             "fetch_count": 12, "failure_reasons": []}}


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path))
    import app.report_store as rs
    importlib.reload(rs)
    return rs


def test_namespace_of_sanitizes(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    assert rs.namespace_of("HDFCBANK.NS") == "HDFCBANK"
    assert rs.namespace_of("tcs.bo") == "TCS"


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    rs.save_report("HDFCBANK.NS", "HDFC Bank Ltd", _report(),
                   "2026-07-04T10:00:00", _specialists(), _runs())
    loaded = rs.load_report("HDFCBANK.NS")
    assert loaded["company_name"] == "HDFC Bank Ltd"
    assert loaded["generated_at"] == "2026-07-04T10:00:00"
    assert loaded["schema_version"] == 2
    assert isinstance(loaded["report"], ReportOutput)
    assert loaded["report"].exec_summary == "Summary."
    assert isinstance(loaded["specialists"]["fundamentals"], FundamentalsOutput)
    assert isinstance(loaded["specialists"]["competitor"], CompetitorOutput)
    assert loaded["specialists"]["competitor"].peers[0].metrics["roe"] is None
    assert loaded["specialists"]["news"] is None
    assert loaded["runs"]["fundamentals"]["fetch_count"] == 12


def test_save_without_specialists_stores_nulls(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    rs.save_report("TCS.NS", "TCS Ltd", _report(), "2026-07-04T11:00:00")
    loaded = rs.load_report("TCS.NS")
    assert set(loaded["specialists"]) == set(rs.SPECIALIST_CLASSES)
    assert all(v is None for v in loaded["specialists"].values())
    assert loaded["runs"] == {}


def test_legacy_file_is_flagged_not_parsed(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    legacy = {"ticker": "SBIN", "company_name": "State Bank of India",
              "generated_at": "2026-07-01T09:00:00",
              "report": {"exec_summary": "old", "sections": {"news": "## N"},
                         "sources": [], "missing_sections": []}}
    name = rs.namespace_of("SBIN.NS")
    (tmp_path / f"{name}.json").write_text(json.dumps(legacy), encoding="utf-8")
    loaded = rs.load_report("SBIN.NS")
    assert loaded["legacy"] is True
    assert loaded["company_name"] == "State Bank of India"
    assert "report" not in loaded


def test_load_missing_returns_none(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    assert rs.load_report("NONE") is None


def test_list_reports_includes_legacy(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    rs.save_report("HDFCBANK.NS", "HDFC Bank Ltd", _report(),
                   "2026-07-04T10:00:00", _specialists(), _runs())
    rs.save_report("TCS.NS", "TCS Ltd", _report(), "2026-07-04T11:00:00")
    tickers = {r["ticker"] for r in rs.list_reports()}
    assert tickers == {rs.namespace_of("HDFCBANK.NS"), rs.namespace_of("TCS.NS")}
