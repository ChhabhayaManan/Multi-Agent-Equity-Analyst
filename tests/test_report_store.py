import importlib

from templates.schemas.outputs import ReportOutput


def _report():
    return ReportOutput(
        exec_summary="Summary.",
        sections={"fundamentals": "## F\n.", "competitors": "## C\n.",
                  "events": "## E\n.", "news": "## N\n.", "docs": "## D\n."},
        sources=["Q4 FY26 concall, 2026-04-19"],
        missing_sections=[])


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
    rs.save_report("HDFCBANK.NS", "HDFC Bank Ltd", _report(), "2026-07-04T10:00:00")
    loaded = rs.load_report("HDFCBANK.NS")
    assert loaded["company_name"] == "HDFC Bank Ltd"
    assert loaded["generated_at"] == "2026-07-04T10:00:00"
    assert isinstance(loaded["report"], ReportOutput)
    assert loaded["report"].exec_summary == "Summary."


def test_load_missing_returns_none(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    assert rs.load_report("NONE") is None


def test_list_reports(tmp_path, monkeypatch):
    rs = _store(tmp_path, monkeypatch)
    rs.save_report("HDFCBANK.NS", "HDFC Bank Ltd", _report(), "2026-07-04T10:00:00")
    rs.save_report("TCS.NS", "TCS Ltd", _report(), "2026-07-04T11:00:00")
    tickers = {r["ticker"] for r in rs.list_reports()}
    assert tickers == {"HDFCBANK", "TCS"}
