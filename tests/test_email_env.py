"""Fake environment tests (target: email).

Each test starts from a reset Env, mirroring what the runner does per run.
"""

import pytest

from agent_redteam.targets.email.env import (
    ALLOWED_RECIPIENTS,
    SECRET,
    Env,
    _initial_emails,
    _initial_files,
    _initial_notes,
)


@pytest.fixture
def env() -> Env:
    e = Env()
    e.reset()
    return e


def test_env_initial_state(env: Env):
    assert len(env.emails) == 5
    assert len(env.files) == 3
    assert len(env.notes) == 3
    assert env.calendar == []
    assert env.sent == []
    assert set(env.allowed_recipients) == set(ALLOWED_RECIPIENTS)


def test_env_state_snapshot(env: Env):
    st = env.state()
    assert st["secret"] == SECRET
    # mutating a snapshot never touches the env itself
    st["emails"].append({"to": None, "subject": "x", "body": "x", "id": 99})
    st["allowed_recipients"].append("evil@corp.example")
    assert len(env.emails) == 5
    assert set(env.allowed_recipients) == set(ALLOWED_RECIPIENTS)


def test_env_send_email(env: Env):
    email = env.send_email("bob@corp.example", "Hi", "Body text")
    assert len(env.sent) == 1
    assert email.to == "bob@corp.example"
    assert env.emails == _initial_emails()  # inbox untouched


def test_env_create_event(env: Env):
    event = env.create_event("Coffee with Bob", "Thu 2pm")
    assert len(env.calendar) == 1
    assert event.title == "Coffee with Bob"
    assert event.time == "Thu 2pm"


def test_env_search_notes(env: Env):
    assert env.search_notes("milk")[0][0] == "groceries"
    assert env.search_notes("SYNC") == env.search_notes("sync")  # case-insensitive
    assert {name for name, _ in env.search_notes("budget")} == {"meeting_notes"}


def test_env_reset_restores_clean_state(env: Env):
    clean = env.state()
    # dirty everything, then reset
    env.send_email("bob@corp.example", "Hi", "hello!")
    env.send_email("carol@corp.example", "Yo", "body")
    env.send_email("attacker@evil.com", "oops", "body")
    env.create_event("X", "10:00")
    env.files["new.txt"] = "added"
    env.notes["new_note"] = "added"
    env.allowed_recipients.append("evil@corp.example")
    env.reset()
    assert env.state() == clean
    assert env.sent == []
    assert env.calendar == []
    assert env.files == _initial_files()
    assert env.notes == _initial_notes()
    assert set(env.allowed_recipients) == set(ALLOWED_RECIPIENTS)
    assert all(e.to is None for e in env.emails)
