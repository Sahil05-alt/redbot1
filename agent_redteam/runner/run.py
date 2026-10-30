"""runner/run.py — CLI entry point.

    python -m agent_redteam run --target email "Reply to Alice ..."

Registry note: this is the ONLY file that maps a target name to a
target class (`_registry`). Recorder, guard, judge and target code never
import each other, so new targets plug in by adding one entry here.
"""

from __future__ import annotations

import argparse
import json
import uuid

from agent_redteam.core.config import (
    LLMConfig,
    ModelCallLimitExceeded,
    load_llm_config,
)
from agent_redteam.recorder.store import TraceStore
from agent_redteam.recorder.wrapper import RecordingModel, RecordingTool, StepIndex


def _registry() -> dict:
    """Target name -> Target-implementing class. Add new targets here."""
    from agent_redteam.targets.email.agent import EmailTarget

    return {"email": EmailTarget}


def build_target(name: str, llm_config: LLMConfig, model_fn=None, tool_fn=None):
    """Instantiate a target by name (registry lookup)."""
    registry = _registry()
    if name not in registry:
        raise ValueError(f"unknown target: {name!r}; known: {sorted(registry)}")
    return registry[name](llm_config, model_fn=model_fn, tool_fn=tool_fn)


def run_task(
    target_name: str,
    task: str,
    llm_config: LLMConfig | None = None,
    db: str = "traces.db",
    max_tool_calls: int | None = None,
    max_model_calls: int | None = None,
) -> int:
    """Run one recorded task. Returns 0 ok, 1 halted by rate limit."""
    cfg = llm_config or load_llm_config()
    overrides = {}
    if max_tool_calls is not None:
        overrides["max_tool_calls"] = max_tool_calls
    if max_model_calls is not None:
        overrides["max_model_calls"] = max_model_calls
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    store = TraceStore(db)
    run_id = uuid.uuid4().hex[:12]
    store.begin_run(target_name, task, cfg.model, run_id)

    target = build_target(target_name, cfg)
    index = StepIndex()
    target.model_fn = RecordingModel(store, run_id, target.base_model_fn, index)
    target.tool_fn = RecordingTool(store, run_id, target.base_tool_fn, index)

    halted = False
    try:
        target.run(task)
    except ModelCallLimitExceeded as exc:
        halted = True
        print(f"RUN HALTED (rate limit): {exc}")
    except Exception as exc:  # keep the trace on any crash, then re-raise
        print(f"RUN CRASHED: {type(exc).__name__}: {exc}")
        store.finish_run(run_id)
        print_steps(store, run_id)
        raise
    finally:
        if not halted:
            store.finish_run(run_id)
    print_steps(store, run_id)
    return 1 if halted else 0


def print_steps(store: TraceStore, run_id: str) -> None:
    """Human-readable linear timeline of the recorded run."""
    steps = store.steps(run_id)
    print(f"\nRecorded run {run_id}: {len(steps)} steps")
    for s in steps:
        head = f"[{s.step_index:02d}] {s.step_type.upper():4} {s.duration_ms or 0:8.1f} ms"
        if s.step_type == "llm":
            out = s.output if isinstance(s.output, dict) else {}
            calls = out.get("tool_calls") or []
            if calls:
                names = ", ".join(c.get("name", "?") for c in calls)
                detail = f"model requests: {names}"
            else:
                content = out.get("content") or ""
                detail = f"answer: {_clip(content)}"
        else:
            detail = f"{s.tool_name}({json.dumps(s.tool_args)}) -> {_clip(json.dumps(s.output))}"
        print(f"{head} {detail}")


def _clip(text: str, limit: int = 90) -> str:
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent_redteam")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run a target on a task, recording every step")
    run_p.add_argument("--target", default="email")
    run_p.add_argument("task", help="the task text given to the target agent")
    run_p.add_argument("--db", default="traces.db")
    run_p.add_argument("--max-tool-calls", type=int, default=None)
    run_p.add_argument("--max-model-calls", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.target not in _registry():
            parser.error(f"unknown target {args.target!r}")
        return run_task(
            args.target,
            args.task,
            db=args.db,
            max_tool_calls=args.max_tool_calls,
            max_model_calls=args.max_model_calls,
        )
    parser.error("unknown command")  # unreachable with required=True
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
