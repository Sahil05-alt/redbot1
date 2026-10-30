"""Shared helpers for tests that exercise a Guard through an agent. fake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_redteam.core.guard import ActionRequest, Guard, GuardError


@dataclass
class Step:
    """One attempted action, as the target's tools layer would perform it."""

    action_type: str
    kwargs: dict[str, Any]
    ok: bool
    reason: str | None = None


class FakeAgent:
    """Simulates the target's tools layer against a Guard.

    Mirrors tools.py's job: pack the model's raw kwargs into an
    ActionRequest, call guard.check() FIRST, and mutate state only if
    the guard allows it.
    """

    def __init__(self, guard: Guard):
        self.guard = guard
        self.history: list[Step] = []

    def request(self, action_type: str, **kwargs: Any) -> Step:
        req = ActionRequest(action_type=action_type, kwargs=kwargs)
        try:
            self.guard.check(req)
        except GuardError as e:
            step = Step(action_type, kwargs, ok=False, reason=e.reason)
        else:
            step = Step(action_type, kwargs, ok=True)
            # (real target would mutate env state here)
        self.history.append(step)
        return step


# ---------------------------------------------------------------------------
# Scripted-model fakes (shared by agent-loop, wrapper and CLI tests)
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from agent_redteam.core.config import LLMConfig as _LLMConfig  # noqa: E402


def make_llm(**overrides) -> _LLMConfig:
    """Config for offline tests; the endpoint is never contacted."""
    values = {
        "model": "fake-model",
        "base_url": "http://localhost:0/v1",
        "api_key": "unused",
        "max_model_calls": 20,
        "max_tool_calls": 30,
    }
    values.update(overrides)
    return _LLMConfig(**values)


def model_reply(content=None, tool_calls=None):
    """OpenAI-SDK-shaped response object without the SDK."""
    call_dicts = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in (tool_calls or [])
    ]

    def dump() -> dict:
        d = {"role": "assistant", "content": content}
        if call_dicts:
            d["tool_calls"] = call_dicts
        return d

    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
        model_dump=dump,
    )
    return SimpleNamespace(message=message, usage=None)


def tool_call(name: str, **args) -> SimpleNamespace:
    """An OpenAI-SDK-shaped tool call object."""
    args_json = _json.dumps(args, sort_keys=True)
    return SimpleNamespace(
        id=f"call_{name}_{args_json}",
        type="function",
        function=SimpleNamespace(name=name, arguments=args_json),
    )


def scripted_model(replies: list):
    """Returns (model_fn, received_histories) that replays a fixed script."""
    histories: list[list] = []

    def model_fn(messages):
        histories.append([dict(m) for m in messages])
        return replies[len(histories) - 1]

    return model_fn, histories
