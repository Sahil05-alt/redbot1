"""Pydantic schemas for the flight recorder.

The recorder is target-agnostic by construction: a run has a linear list
of steps, and every step is either a `llm` call or a `tool` call. The
full message list / tool arguments go into JSON payload columns
(sqlite3 repr) so nothing about any specific target leaks into this
module, and later targets plug in unchanged.

Deliberately NO credential fields exist on any schema and the recorder
is never handed the API key (config is built at runner level).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

StepType = Literal["llm", "tool"]


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp, used for wall-clock ordering fields."""
    return datetime.now(timezone.utc).isoformat()


class TokenUsage(BaseModel):
    """Reported token counts, when the provider supplies them."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TraceRun(BaseModel):
    """One executed attack scenario over one target."""

    run_id: str = Field(min_length=1)
    target: str  # e.g. "email"
    task: str  # the user task text
    model: str  # model identifier from config
    started_at: str  # ISO-8601 datetime
    finished_at: str | None = None


class TraceStep(BaseModel):
    """One recorded event inside a run: a model call or a tool call.

    llm steps: input=messages list, output=assistant message content,
               tokens=provider usage (or None).
    tool steps: tool_name + tool_args, output=the tool result dict fed
               back to the model (includes guard "blocked" outcomes).
    """

    run_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)  # position inside the run, from 0
    step_type: StepType
    input: Any = None   # llm: list of messages | tool: raw argument dict
    output: Any = None  # llm: reply text | tool: result dict (incl. blocked)
    tool_name: str | None = None
    tool_args: dict | None = None
    timestamp: str = Field(default_factory=utcnow_iso)
    duration_ms: float | None = None
    tokens: TokenUsage | None = None

    @field_validator("step_type")
    @classmethod
    def _type_must_be_llm_or_tool(cls, value: str) -> str:
        if value not in ("llm", "tool"):
            raise ValueError("step_type must be 'llm' or 'tool'")
        return value

    @field_validator("tool_args", "tokens", "input", "output")
    @classmethod
    def _copy_dict(cls, value: Any) -> Any:
        """Defensive copy so later mutations can't rewrite history."""
        if isinstance(value, dict):
            return {k: v for k, v in value.items()}
        return value
