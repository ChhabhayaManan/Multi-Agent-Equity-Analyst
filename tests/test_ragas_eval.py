from eval.golden import GoldenItem
from eval import ragas_eval


def test_build_dataset_shape():
    items = [GoldenItem("direct-factual", "Q1?", "GT1"),
             GoldenItem("unanswerable", "Q2?", "I don't have that information")]

    def ask_fn(q):
        return (f"ans-{q}", [f"ctx-{q}"])

    rows = ragas_eval.build_dataset(items, ask_fn)
    assert rows == [
        {"user_input": "Q1?", "response": "ans-Q1?",
         "retrieved_contexts": ["ctx-Q1?"], "reference": "GT1"},
        {"user_input": "Q2?", "response": "ans-Q2?",
         "retrieved_contexts": ["ctx-Q2?"],
         "reference": "I don't have that information"},
    ]


def test_run_eval_uses_stubs(monkeypatch, tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"ticker": "HDFCBANK.NS", "questions": [
        {"type": "direct-factual", "question": "Q?", "ground_truth": "A"}]}),
        encoding="utf-8")

    # Defense-in-depth: neutralize the @traceable run + metadata sink so that
    # even a leaked LANGCHAIN_TRACING_V2 from another test can't fire a real
    # LangSmith POST during this stubbed run.
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setattr(ragas_eval, "set_run_metadata", lambda md: None)
    monkeypatch.setattr(ragas_eval, "init_tracing", lambda: False)
    monkeypatch.setattr(ragas_eval, "_ensure_indexed", lambda t: None)
    monkeypatch.setattr(ragas_eval, "_session_ask_fn",
                        lambda ticker: (lambda q: ("ans", ["ctx"])))

    class _FakeResult:
        def to_pandas(self):
            import types
            return types.SimpleNamespace(
                to_dict=lambda **k: {},
                mean=lambda numeric_only=True: {"faithfulness": 0.9,
                                                "llm_context_precision_with_reference": 0.8})

    monkeypatch.setattr(ragas_eval, "_evaluate", lambda rows: _FakeResult())
    scores = ragas_eval.run_eval(str(p))
    assert scores["faithfulness"] == 0.9
