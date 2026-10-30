"""Runner CLI tests: offline end-to-end with a stub target via the registry."""

from types import SimpleNamespace

import pytest

from agent_redteam.recorder.store import TraceStore
from agent_redteam.runner import run as run_mod
from tests.test_support import make_llm


class StubTarget:
    """Protocol-shaped placeholder to exercise the runner plumbing."""

    name = "stub"
    counter = 0

    def __init__(self, llm_config=None, model_fn=None, tool_fn=None):
        type(self).counter += 1
        StubTarget.counter = type(self).counter
        self.base_model_fn = model_fn or self._fake_model_fn
        self.model_fn = self.base_model_fn
        self.base_tool_fn = lambda name, args: {"reply": name}
        self.tool_fn = self.base_tool_fn

    @staticmethod
    def _fake_model_fn(messages):
        return SimpleNamespace(
            message=SimpleNamespace(content="done", tool_calls=None), usage=None
        )

    def run(self, task: str) -> None:
        self.model_fn([{"role": "user", "content": task}])
        self.tool_fn("stub_tool", {"x": 1})
        # second llm turn, so the timeline proves both wrappers interleave
        self.model_fn([{"role": "user", "content": task}])

    def tools(self):
        return []

    def state(self):
        return {}

    def inject_points(self):
        return []

    def reset(self):
        pass


@pytest.fixture
def stub_registry(monkeypatch):
    registry = {"stub": StubTarget}
    monkeypatch.setattr(run_mod, "_registry", lambda: registry)
    return StubTarget


def test_registry_missing_target():
    with pytest.raises(ValueError):
        run_mod.build_target("no-such", make_llm())


def test_registry_builds_by_name(monkeypatch, stub_registry):
    stub_registry.counter = 0
    target = run_mod.build_target("stub", make_llm())
    assert target.name == "stub"
    assert StubTarget.counter == 1


def test_cli_run_records_steps(tmp_path, capsys, stub_registry, monkeypatch):
    # config comes from env, like a real run
    monkeypatch.setenv("REDTEAM_MODEL", "fake-model")
    db = str(tmp_path / "traces.db")
    code = run_mod.main(["run", "--target", "stub", "hello world", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "[00] LLM" in out and "answer: done" in out
    assert "[01] TOOL" in out and "stub_tool" in out
    assert "[02] LLM" in out
    assert "Recorded run" in out

    # the trace persisted and finished
    store = TraceStore(db)
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].target == "stub"
    assert runs[0].task == "hello world"
    assert runs[0].model == "fake-model"
    assert runs[0].finished_at is not None
    assert [s.step_index for s in store.steps(runs[0].run_id)] == [0, 1, 2]


def test_cli_unknown_target_errors(capsys, monkeypatch, stub_registry):
    with pytest.raises(SystemExit):
        run_mod.main(["run", "--target", "nonexistent", "task"])
