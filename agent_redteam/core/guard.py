"""core/guard.py — one chokepoint for every state-changing action.

The model can never bypass this layer: target tool code packs each
side-effecting call into an ActionRequest and is allowed to mutate env
state ONLY after guard.check() returns. Enforcement is in code, not in
prompts (the model never even sees the allowlist).

Two policies live here:

1. Kill switch — hard cap on state-changing tool calls per run
   (CallLimitExceeded).
2. Allowlist — per-action rules (e.g. send_email recipients must be on
   an allowed domain list) written as plain Python CheckFns.

This module knows nothing about email: register rules for any
action_type ("send_email", "create_event", "file_write", ...).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# Checks receive one ActionRequest; return None to pass, or a string
# with the reason it is denied.
CheckFn = Callable[["ActionRequest"], str | None]


class GuardError(Exception):
    """A request was denied by policy."""

    def __init__(self, action_type: str, reason: str):
        super().__init__(f"{action_type}: {reason}")
        self.action_type = action_type
        self.reason = reason


class CallLimitExceeded(GuardError):
    """The per-run kill switch fired."""

    def __init__(self, calls_used: int, limit: int):
        super().__init__(
            "run",
            f"tool-call limit {limit} exceeded ({calls_used} calls already made)",
        )
        self.calls_used = calls_used
        self.limit = limit


@dataclass(frozen=True)
class ActionRequest:
    """A state-changing action that wants to happen.

    action_type: which registered ruleset applies ("send_email", ...)
    kwargs:      raw arguments as the model supplied them
    """

    action_type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Built-in checkers
# ---------------------------------------------------------------------------


def recipient_allowlist(key: str = "to", domains: Iterable[str] | None = None) -> CheckFn:
    """Allowlist emails in kwargs[key].

    Accepts a single address string or a list/tuple. Every address must
    be well-formed and, if `domains` is given, on those domains
    (case-insensitive). The check runs in code; the model never sees it.
    """
    allowed = {d.lower().strip() for d in domains or ()}

    def check(request: ActionRequest) -> str | None:
        value = request.kwargs.get(key)
        if value is None:
            return f"missing '{key}'"
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = [str(v) for v in value]
        else:
            return f"'{key}' must be str or list, got {type(value).__name__}"
        for raw in values:
            addr = raw.strip().lower()
            if not addr:
                return f"empty '{key}' address"
            if _EMAIL_RE.match(addr) is None:
                return f"invalid email address: {raw!r}"
            domain = addr.split("@", 1)[1]
            if allowed and domain not in allowed:
                return f"recipient {raw!r} not on allowlist (domain {domain!r})"
        return None

    return check


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class Guard:
    """Kill switch + allowlist for state-changing actions.

    Order of checks per call: kill switch first (so a runaway loop stops
    even if the action would otherwise be allowlisted), then rules, then
    the call is counted.
    """

    def __init__(self, max_calls: int = 30):
        self._limit = max_calls
        self._used = 0
        self._rules: dict[str, list[CheckFn]] = {}

    def add_rule(self, action_type: str, check: CheckFn) -> None:
        """Register a rule for an action type (e.g. "send_email")."""
        self._rules.setdefault(action_type, []).append(check)

    def check(self, request: ActionRequest) -> None:
        """Validate one action. Raises GuardError on any denial."""
        if self._used >= self._limit:
            raise CallLimitExceeded(self._used, self._limit)
        for check in self._rules.get(request.action_type, ()):
            reason = check(request)
            if reason is not None:
                raise GuardError(request.action_type, reason)
        self._used += 1

    def calls_used(self) -> int:
        return self._used

    def limit(self) -> int:
        return self._limit

    def reset(self) -> None:
        """Clear the call counter (new run)."""
        self._used = 0
