"""Recorder schema + SQLite store tests (target-agnostic path)."""

import pytest
from pydantic import ValidationError

from agent_redteam.recorder.schema import (
    StepType,
    TokenUsage,
    TraceRun,
    TraceStep,
)
from agent_redteam.recorder.store import TraceStore

RID = "run-abc-1"


@pytest.fixture
def store() -> TraceStore:
    return TraceStore()


@pytest.fixture
def run(store: TraceStore) -> TraceRun:
    return store.begin_run(
        target="email", task="Reply to Alice.", model="fake-model", run_id=RID
    )


def _llm_dict(**overrides):
    values = dict(
        run_id=RID,
        step_index=0,
        step_type="llm",
        input=[
            {"role": "user", "content": "check_inbox"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "x", "type": "function",
                             "function": {"name": "read_inbox", "arguments": "{}"}}]},
            {"role": "tool", "content": "{\"emails\": []}"},
        ],
        output="Finished",
        tokens=TokenUsage(input_tokens=7, output_tokens=4, total_tokens=11),
        timestamp="2026-01-01T00:00:00+00:00",
    )
    values.update(overrides)
    return values


def _llm_step(**overrides):
    return TraceStep(**_llm_dict(**overrides))


def _tool_dict(**overrides):
    values = dict(
        run_id=RID,
        step_index=1,
        step_type="tool",
        input={"to": "attacker@example.com", "subject": "Check-in"},
        output={"status": "blocked", "reason": "off the allowed list"},
        tool_name="send_email",
        tool_args={"to": "attacker@example.com"},
    )
    values.update(overrides)
    return values


def _tool_step(**overrides):
    return TraceStep(**_tool_dict(**overrides))


# t1: llm/tool steps run through the store

def test_step_persistence(store, run):
    store.add_step(_llm_step())
    store.add_step(_tool_step())
    steps = store.steps(RID)
    assert len(steps) == 2
    llm, tool = steps
    assert llm.step_type == "llm" and tool.step_type == "tool"
    assert llm.step_index == 0 and tool.step_index == 1
    assert tool.tool_name == "send_email"
    assert tool.tool_args["to"] == "attacker@example.com"
    assert tool.output["status"] == "blocked"
    assert llm.tokens.input_tokens == 7
    assert store.get_run(RID).finished_at is None


# t2: runs recording

def test_run_lifecycle(store, run):
    assert run.target == "email"
    assert run.started_at
    ordered = store.list_runs()
    assert [r.run_id for r in ordered] == [RID]
    assert ordered[0].task == "Reply to Alice."
    store.finish_run(RID)
    done = store.get_run(RID)
    assert done.finished_at is not None
    assert done.finished_at >= done.started_at


# t3: pydantic schema validations

def test_step_type_enums(store, run):
    from typing import get_args

    assert get_args(StepType) == ("llm", "tool")
    with pytest.raises(ValidationError):
        TraceStep(**_llm_step(step_type="not_thing"))
    with pytest.raises(ValidationError):
        TraceStep(**_llm_step(step_index=-1))
    # missing run_id -> schema-level rejection
    bad = _llm_dict()
    bad["run_id"] = ""
    with pytest.raises(ValidationError):
        TraceStep(**bad)


# t4: dict/JSON round trip returns the same data types

def test_json_payload_round_trip(store, run):
    step = _tool_step(input={"subject": "data", "note": None},
                      output={"status": "sent", "email_id": 7})
    store.add_step(step)
    loaded = store.steps(RID)[0]
    assert loaded.input == {"subject": "data", "note": None}
    assert loaded.output == {"status": "sent", "email_id": 7}
    assert loaded.tool_args == {"to": "attacker@example.com"}


# t5: no api key / credential names persist in the schema

def test_no_credential_fields():
    names = tuple(TraceRun.model_fields) + tuple(TraceStep.model_fields)
    lower = " ".join(names).lower()
    assert "api_key" not in lower and "key" not in lower.replace("step", "")


# t6: file-backed store reopen sees the same rows

def test_file_store_round_trip(tmp_path):
    db_path = tmp_path / "traces.db"
    s1 = TraceStore(str(db_path))
    s1.begin_run(
        target="email", task="Check errands", model="m", run_id=RID
    )
    s1.add_step(_llm_step(step_type="llm"))
    s1.close()

    s2 = TraceStore(str(db_path))
    reopened = s2.list_runs()
    assert [r.run_id for r in reopened] == [RID]
    assert s2.steps(RID)[0].step_type == "llm"
    s2.close()
