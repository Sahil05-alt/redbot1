"""Tool-layer tests (target: email).

Mirrors how the agent loop and future model wrapper will use this module.
"""

import pytest

from agent_redteam.core.guard import Guard, recipient_allowlist
from agent_redteam.targets.email.env import Env
from agent_redteam.targets.email import tools


@pytest.fixture
def env() -> Env:
    e = Env()
    e.reset()
    return e


@pytest.fixture
def guard() -> Guard:
    g = Guard(max_calls=20)
    g.add_rule("send_email", recipient_allowlist("to", ["corp.example"]))
    return g


def all_tool_names() -> list[str]:
    return sorted(tools.TOOL_FUNCS)


# t1: the three tool kinds behave differently in this module

def test_basic_tool_types(env, guard):
    names = all_tool_names()
    assert names == ["create_event", "read_file", "read_inbox", "search_notes", "send_email"]
    assert not (set(tools.TOOL_FUNCS) - set(names))


def test_read_tools_cannot_mutate_env(env, guard):
    before = env.state()
    tools.dispatch_tool("read_inbox", {}, env, guard)
    tools.dispatch_tool("read_file", {"name": "notes.txt"}, env, guard)
    tools.dispatch_tool("search_notes", {"query": "sync"}, env, guard)
    assert env.state() == before


def test_all_mutating_tools_are_guarded(env, guard):
    """Mutating tools touch the guard; read tools don't (by contract code)."""
    guard2 = Guard(max_calls=3)
    guard2.add_rule("send_email", recipient_allowlist("to", ["corp.example"]))
    # everything allowed: both mutating tools go through and count
    tools.dispatch_tool("send_email", {"to": "x@corp.example", "subject": "s", "body": "b"}, env, guard2)
    tools.dispatch_tool("create_event", {"title": "T", "time": "10:00"}, env, guard2)
    assert env.state()["sent"] and env.state()["calendar"]
    assert guard2.calls_used() == 2

    # deny send_email in a third guard: creating event still works
    env2 = Env()
    env2.reset()
    guard3 = Guard(max_calls=10)
    guard3.add_rule("send_email", recipient_allowlist("to", ["corp.example"]))
    out = tools.dispatch_tool(
        "send_email", {"to": "x@blocked.com", "subject": "s", "body": "b"}, env2, guard3
    )
    assert out["status"] == "blocked"
    assert env2.state()["sent"] == []
    assert guard3.calls_used() == 0  # blocked calls don't consume budget


# t2: tool schemas — the values the ToolsBox serialises into OpenAI format

def test_read_tool_payloads(env, guard):
    """Read tools return payloads the model can use (and stay read-only)."""
    inbox = tools.dispatch_tool("read_inbox", {}, env, guard)
    assert inbox["count"] == 5
    assert inbox["emails"][0]["subject"] == "Team sync on Thursday"

    found = tools.dispatch_tool("read_file", {"name": "notes.txt"}, env, guard)
    assert "checklist" in found["content"]

    missing = tools.dispatch_tool("read_file", {"name": "nope.txt"}, env, guard)
    assert "file not found" in missing["error"]
    assert set(missing["files"]) == set(env.files)

    hits = tools.dispatch_tool("search_notes", {"query": "groceries"}, env, guard)
    assert hits["count"] == 1 and hits["matches"][0]["name"] == "groceries"

    unknown = tools.dispatch_tool("teleport", {}, env, guard)
    assert "unknown tool" in unknown["error"]


def test_tool_schemas(env, guard):
    specs = tools.tool_schemas()
    names = [s["function"]["name"] for s in specs]
    assert names == ["read_inbox", "read_file", "search_notes", "send_email", "create_event"]
    for s in specs:
        assert s["type"] == "function"
        for key in ("description", "parameters"):
            assert s["function"][key]
        assert s["function"]["parameters"]["type"] == "object"
        assert "required" in s["function"]["parameters"]


def test_json_serialisable():
    specs = tools.tool_schemas()
    import json

    for target in specs:
        json.dumps(target)  # must not raise
