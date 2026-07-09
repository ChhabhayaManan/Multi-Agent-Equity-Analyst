# tests/test_push_prompts.py
from langchain_core.prompts import BasePromptTemplate

from eval.scripts import push_prompts


def test_prompts_registry_complete():
    assert len(push_prompts.PROMPTS) == 8
    for name, prompt in push_prompts.PROMPTS.items():
        assert name.startswith("stock-research-")
        assert isinstance(prompt, BasePromptTemplate)


def test_push_all_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(push_prompts, "init_tracing", lambda: False)
    # Client must never be constructed when disabled.
    def _boom(*a, **k):
        raise AssertionError("Client should not be built when tracing off")
    monkeypatch.setattr(push_prompts, "_client", _boom)
    assert push_prompts.push_all() == 0


def test_push_all_pushes_each(monkeypatch):
    monkeypatch.setattr(push_prompts, "init_tracing", lambda: True)
    pushed = []

    class _FakeClient:
        def push_prompt(self, name, object=None):
            pushed.append(name)

    monkeypatch.setattr(push_prompts, "_client", lambda: _FakeClient())
    n = push_prompts.push_all()
    assert n == 8
    assert len(pushed) == 8
