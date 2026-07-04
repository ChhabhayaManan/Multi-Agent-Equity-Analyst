# tests/test_tracing.py
import importlib
import utils.tracing as tracing


def _reload():
    importlib.reload(tracing)
    return tracing


def test_init_tracing_noop_without_key(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setattr("utils.tracing.load_dotenv", lambda *a, **k: None)
    t = _reload()
    monkeypatch.setattr(t, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert t.init_tracing() is False
    assert os_env_missing("LANGCHAIN_TRACING_V2")


def test_init_tracing_enables_with_key(monkeypatch):
    t = _reload()
    monkeypatch.setattr(t, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "fake-key")
    # setenv (not delenv) so monkeypatch records the original and restores it
    # on teardown; delenv on an already-absent var registers no restoration,
    # letting init_tracing's os.environ["LANGCHAIN_TRACING_V2"]="true" leak
    # process-wide and trip a real LangSmith POST in a later @traceable test.
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    assert t.init_tracing() is True
    import os
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "stock-research"


def test_set_run_metadata_noop_when_no_run(monkeypatch):
    t = _reload()
    monkeypatch.setattr(
        "langsmith.run_helpers.get_current_run_tree", lambda: None,
        raising=False)
    t.set_run_metadata({"x": 1})  # must not raise


def test_set_run_metadata_attaches_when_run_present(monkeypatch):
    t = _reload()
    captured = {}

    class _Tree:
        def add_metadata(self, md):
            captured.update(md)

    monkeypatch.setattr(
        "langsmith.run_helpers.get_current_run_tree", lambda: _Tree(),
        raising=False)
    t.set_run_metadata({"x": 1})
    assert captured == {"x": 1}


def test_traceable_is_noop_decorator_when_langsmith_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "langsmith":
            raise ImportError("no langsmith")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    t = _reload()

    @t.traceable(run_type="chain", name="x")
    def f(v):
        return v + 1

    assert f(1) == 2


def os_env_missing(key):
    import os
    return os.environ.get(key) != "true"
