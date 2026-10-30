"""SQLite storage for the flight recorder (stdlib sqlite3 only).

Tables (design option A — generic rows + JSON payloads):

  runs(...)  id, target, task, model, started_at, finished_at|null
  steps(...) one row per llm/tool call, payloads as JSON text

The store knows nothing about email, attacks or the agent loop. It only
speaks pydantic schemas in / rows out, and JSON-encoders anything that
is not a basic type.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent_redteam.recorder.schema import TokenUsage, TraceRun, TraceStep, utcnow_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    target      TEXT NOT NULL,
    task        TEXT NOT NULL,
    model       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    step_index  INTEGER NOT NULL,
    step_type   TEXT NOT NULL CHECK (step_type IN ('llm', 'tool')),
    input       TEXT,
    output      TEXT,
    tool_name   TEXT,
    tool_args   TEXT,
    timestamp   TEXT NOT NULL,
    duration_ms REAL,
    tokens      TEXT,
    PRIMARY KEY (run_id, step_index)
);
"""


def _json(value: Any) -> str | None:
    """Everything non-basic goes in as compact JSON text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(value: str | None) -> Any:
    if value is None or not value.startswith(("{", "[", '"')):
        return value
    return json.loads(value)


class TraceStore:
    """Read/write access to the runs + steps tables."""

    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def begin_run(self, target: str, task: str, model: str, run_id: str) -> TraceRun:
        run = TraceRun(
            run_id=run_id, target=target, task=task, model=model,
            started_at=utcnow_iso(),
        )
        self._conn.execute(
            "INSERT INTO runs (run_id, target, task, model, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (run.run_id, run.target, run.task, run.model, run.started_at),
        )
        self._conn.commit()
        return run

    def finish_run(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at = ? WHERE run_id = ?",
            (utcnow_iso(), run_id),
        )
        self._conn.commit()

    def add_step(self, step: TraceStep) -> None:
        """Insert one validated step row (payloads JSON-encoded)."""
        if not isinstance(step, TraceStep):
            step = TraceStep(**step)
        self._conn.execute(
            "INSERT INTO steps (run_id, step_index, step_type, input, output, "
            "tool_name, tool_args, timestamp, duration_ms, tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                step.run_id,
                step.step_index,
                step.step_type,
                _json(step.input),
                _json(step.output),
                step.tool_name,
                _json(step.tool_args),
                step.timestamp,
                step.duration_ms,
                _json(step.tokens.model_dump() if step.tokens else None),
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> TraceRun | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return TraceRun(**row) if row else None

    def list_runs(self) -> list[TraceRun]:
        """All runs, newest first (started_at DESC, run_id tiebreak)."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC, run_id DESC"
        ).fetchall()
        return [TraceRun(**r) for r in rows]

    def steps(self, run_id: str) -> list[TraceStep]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
        ).fetchall()
        return [
            TraceStep(
                run_id=r["run_id"],
                step_index=r["step_index"],
                step_type=r["step_type"],
                input=_load_json(r["input"]),
                output=_load_json(r["output"]),
                tool_name=r["tool_name"],
                tool_args=_load_json(r["tool_args"]),
                timestamp=r["timestamp"],
                duration_ms=r["duration_ms"],
                tokens=None
                if r["tokens"] is None
                else TokenUsage(**json.loads(r["tokens"])),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
