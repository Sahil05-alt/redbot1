"""Contract tests for the Target protocol (core/target.py)."""

import pytest

from agent_redteam.core.target import Target


class FakeTarget:
    """Duck-typed minimal target used only to exercise the protocol shape."""

    name = "fake"

    def tools(self):
        return [{"type": "function", "function": {"name": "noop"}}]

    def run(self, task: str) -> None:
        return None

    def state(self) -> dict:
        return {"key": "value"}

    def inject_points(self) -> list[str]:
        return ["point_a"]

    def reset(self) -> None:
        pass


@pytest.mark.parametrize("attr", ["tools", "run", "state", "inject_points", "reset"])
def test_protocol_methods(attr):
    assert callable(getattr(Target, attr)), f"Target protocol missing method: {attr}"


def test_protocol_name_is_declared():
    assert "name" in Target.__annotations__


def test_runtime_checkable_conformance():
    """A duck-typed class qualifies via isinstance, no subclassing needed."""
    assert isinstance(FakeTarget(), Target)


def test_non_target_fails_conformance():
    class NotATarget:
        name = "nope"  # has name, but none of the required methods

    assert not isinstance(NotATarget(), Target)


def test_target_protocol_class_conforms():
    target = FakeTarget()
    assert target.name == "fake"
    assert target.tools() == [{"type": "function", "function": {"name": "noop"}}]
    assert target.state() == {"key": "value"}
    assert target.inject_points() == ["point_a"]
    assert target.reset() is None
