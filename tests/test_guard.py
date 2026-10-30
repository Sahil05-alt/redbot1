"""Kill-switch and allowlist tests for the Guard."""

import pytest

from agent_redteam.core.guard import (
    ActionRequest,
    CallLimitExceeded,
    Guard,
    GuardError,
    recipient_allowlist,
)
from tests.test_support import FakeAgent


def allowlisted_agent(max_calls: int = 30, domains: list[str] | None = None) -> FakeAgent:
    g = Guard(max_calls=max_calls)
    g.add_rule("send_email", recipient_allowlist("to", domains))
    return FakeAgent(g)


# t1: kill switch — call cap

def test_guard_blocks_over_limit():
    agent = allowlisted_agent(max_calls=30)
    for _ in range(30):
        step = agent.request("send_email", to="a@corp.example", subject="hi", body="x")
        assert step.ok
    step = agent.request("send_email", to="a@corp.example", subject="hi", body="x")
    assert not step.ok
    assert "tool-call limit 30 exceeded" in (step.reason or "")

def test_guard_blocks_after_reset():
    agent = allowlisted_agent(max_calls=30)
    for _ in range(30):
        agent.request("send_email", to="a@corp.example", subject="hi", body="x")
    agent.guard.reset()  # new run
    step = agent.request("send_email", to="a@corp.example", subject="hi", body="x")
    assert step.ok


# t2: allowlist — recipients

def test_guard_blocks_non_allowlisted_recipient():
    agent = allowlisted_agent(domains=["corp.example", "examples.com"])
    step = agent.request("send_email", to="attacker@blocked.com", subject="hi", body="x")
    assert not step.ok
    assert "not on allowlist" in (step.reason or "")

def test_guard_blocks_blocked_key():
    agent = allowlisted_agent(domains=["corp.example"])
    step = agent.request("send_email", to="a@blocked.com", subject="hi", body="x")
    assert not step.ok
    assert "not on allowlist" in (step.reason or "")
    assert agent.history[0].kwargs["to"] == "a@blocked.com"

def test_guard_allowlisted_recipient_passes():
    agent = allowlisted_agent(domains=["corp.example"])
    step = agent.request("send_email", to="a@corp.example", subject="hi", body="x")
    assert step.ok

def test_guard_blocks_list_with_one_bad_recipient():
    agent = allowlisted_agent(domains=["corp.example"])
    step = agent.request(
        "send_email",
        to=["a@corp.example", "b@evil.com"],
        subject="hi",
        body="x",
    )
    assert not step.ok
    assert "evil.com" in (step.reason or "")

def test_guard_blocks_invalid_email():
    agent = allowlisted_agent(domains=None)  # no domain restriction
    step = agent.request("send_email", to="not-an-email", subject="hi", body="x")
    assert not step.ok
    assert "invalid email address" in (step.reason or "")

def test_guard_blocks_missing_to():
    agent = allowlisted_agent()
    step = agent.request("send_email", subject="hi", body="x")
    assert not step.ok
    assert "missing 'to'" in (step.reason or "")


# t3: check() mechanics — kill switch, GuardError, domain restrictions

def test_checker_side_effect():
    """Kill-switch timing + language-agnostic rules + every attempt recorded."""
    g = Guard(max_calls=3)
    agent = FakeAgent(g)
    agent.request("send_email", to="a@corp.example", subject="s", body="b")  # 1 ok
    agent.request("create_event", title="X", time="10:00")  # 2 ok, no rules needed
    step = agent.request("send_email", to="a@corp.example", subject="s", body="b")  # 3
    assert step.ok
    step = agent.request("send_email", to="a@corp.example", subject="s", body="b")  # 4th of 3
    assert not step.ok
    assert "tool-call limit 3 exceeded" in (step.reason or "")
    assert len(agent.history) == 4  # every attempt is recorded

def test_allowlisted_recipient_passes(check="to"):
    g = Guard(max_calls=10)
    g.add_rule("send_email", recipient_allowlist(check, ["corp.example"]))
    g.check(ActionRequest("send_email", {"to": "b@corp.example"}))  # no raise

def test_guard_kill_switch_raises():
    g = Guard(max_calls=1)
    g.check(ActionRequest("send_email", {"to": "a@corp.example"}))
    with pytest.raises(CallLimitExceeded):
        g.check(ActionRequest("send_email", {"to": "a@corp.example"}))

def test_guard_error_attributes():
    g = Guard(max_calls=5)
    g.add_rule("send_email", recipient_allowlist("to", ["corp.example"]))
    with pytest.raises(GuardError) as exc_info:
        g.check(ActionRequest("send_email", {"to": "a@blocked.com"}))
    assert exc_info.value.action_type == "send_email"
    assert "not on allowlist" in exc_info.value.reason
    assert "blocked.com" in str(exc_info.value)
