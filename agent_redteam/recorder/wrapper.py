"""Recording wrappers: turn every model/tool call into one step row.

Both wrappers are transparent — they return exactly what the wrapped
callable returned, only writing to the store first. The runner installs
them on a fresh target per run:

    target.model_fn = RecordingModel(store, run_id, target.base_model_fn, index)
    target.tool_fn  = RecordingTool(store, run_id, target.base_tool_fn, index)

`index` is shared so steps form ONE linear timeline:
llm=0, tool=1, tool=2, llm=3 ... (replay reads it in order).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from agent_redteam.recorder.schema import TokenUsage, TraceStep

ModelFn = Callable[[list[dict]], Any]
ToolFn = Callable[[str, dict], dict]


class StepIndex:
    """Shared zero-based counter across llm and tool steps."""

    def __init__(self) -> None:
        self._next = 0

    def allocate(self) -> int:
        n = self._next
        self._next += 1
        return n


class RecordingModel:
    """Adds one llm step per model call."""

    def __init__(self, store, run_id: str, model_fn: ModelFn, index: StepIndex):
        self._store = store
        self._run_id = run_id
        self._inner = model_fn
        self._index = index

    def __call__(self, messages: list[dict]) -> Any:
        start = time.perf_counter()
        resp = self._inner(messages)
        duration_ms = (time.perf_counter() - start) * 1000.0

        msg = resp.message
        output: dict = {"content": msg.content}
        if getattr(msg, "tool_calls", None):
            output["tool_calls"] = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]

        # OpenAI usage fields when the provider reports them.
        tokens = None
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens = TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        self._store.add_step(
            TraceStep(
                run_id=self._run_id,
                step_index=self._index.allocate(),
                step_type="llm",
                input=[dict(m) for m in messages],
                output=output,
                duration_ms=duration_ms,
                tokens=tokens,
            )
        )
        return resp  # recording is transparent


class RecordingTool:
    """Adds one tool step per tool call (guard 'blocked' outcomes included)."""

    def __init__(self, store, run_id: str, tool_fn: ToolFn, index: StepIndex):
        self._store = store
        self._run_id = run_id
        self._inner = tool_fn
        self._index = index

    def __call__(self, name: str, args: dict) -> dict:
        start = time.perf_counter()
        result = self._inner(name, args)
        duration_ms = (time.perf_counter() - start) * 1000.0
        self._store.add_step(
            TraceStep(
                run_id=self._run_id,
                step_index=self._index.allocate(),
                step_type="tool",
                input=dict(args),
                output=result,
                tool_name=name,
                tool_args=dict(args),
                duration_ms=duration_ms,
            )
        )
        return result
