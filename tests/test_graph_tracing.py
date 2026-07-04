import workflow.graph as g


class _FakeGraph:
    def __init__(self):
        self.invoke_config = None
        self.stream_config = None

    def invoke(self, inp, config=None):
        self.invoke_config = config
        return {"ticker": inp["ticker"], "report": None, "runs": {}}

    def stream(self, inp, stream_mode=None, config=None):
        self.stream_config = config
        return iter([])


def test_generate_report_tags_run(monkeypatch):
    fake = _FakeGraph()
    monkeypatch.setattr(g, "build_graph", lambda: fake)
    g.generate_report("HDFCBANK.NS", "HDFC Bank")
    assert fake.invoke_config["tags"] == ["HDFCBANK.NS"]
    assert fake.invoke_config["metadata"] == {
        "run_type": "report", "ticker": "HDFCBANK.NS"}


def test_stream_report_tags_run(monkeypatch):
    fake = _FakeGraph()
    monkeypatch.setattr(g, "build_graph", lambda: fake)
    list(g.stream_report("HDFCBANK.NS", "HDFC Bank"))
    assert fake.stream_config["tags"] == ["HDFCBANK.NS"]
    assert fake.stream_config["metadata"]["run_type"] == "report"
