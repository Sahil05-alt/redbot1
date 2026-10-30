"""The Target protocol — the seam between the framework and any agent.

Anything on the right of this line is target-specific: recorder, runner,
guard, config know nothing about email/support/coding agents. A target
exposes:

- its callable tools (JSON schemas for the model),
- one run() that drives the raw tool-calling loop end to end,
- a state() snapshot the judge can inspect deterministically,
- inject_points(): where untrusted text can enter (attack weeks use
  these names), and
- reset(): restoring a clean environment between runs.

Targets package their own attack files and judge rules alongside their
code; those stay out of core.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Target(Protocol):
    """Contract every target agent must satisfy.

    Attributes
    ----------
    name:
        Short unique identifier used for runtime selection,
        e.g. "email", "support", "coding".
    """

    name: str

    def tools(self) -> list[dict]:
        """JSON-schema'd tool descriptions handed to the model.

        One dict per tool with OpenAI tool-calling fields (name,
        description, parameters). All tools must be side-effect-safe:
        anything that mutates state must route through the guard.
        """
        ...

    def run(self, task: str) -> None:
        """Drive the agent on one task until a final answer or a stop.

        All model calls and tool calls must pass through the recorder
        (via the RecordingModel/RecordingTool wrappers, step 6). No
        return value: results live in the recorder + state().
        """
        ...

    def state(self) -> dict:
        """Snapshot of the environment for a judge to inspect.

        Must be a deep copy: mutating the dict must not affect the live
        environment. Judge rules are deterministic checks over this.
        """
        ...

    def inject_points(self) -> list[str]:
        """Where untrusted text can enter the agent's context.

        Names of prompt-injection locations the attack engine later
        uses, e.g. ["email_body", "file_content", "note_text"].
        """
        ...

    def reset(self) -> None:
        """Restore a clean environment before/after each run."""
        ...
