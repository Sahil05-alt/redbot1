"""Model-facing tools for the email target.

This is the only place the model can touch the environment. Contract:

- Every tool returns a JSON-serialisable dict to feed back to the model.
- READ tools (read_inbox, read_file, search_notes) never touch the
  guard and never mutate state.
- SIDE-EFFECT tools (send_email, create_event) pack the model's raw
  kwargs into an ActionRequest and call guard.check() FIRST; only if it
  passes is the Env primitive invoked. A blocked call is returned to the
  model as {"status": "blocked", "reason": ...} and nothing is applied.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_redteam.core.guard import ActionRequest, Guard, GuardError
from agent_redteam.targets.email.env import Env

# Dispatch table: tool name -> executor(args, env, guard) -> dict.
TOOL_FUNCS: dict[str, Callable[..., dict]] = {}


def _register(schema: dict) -> Callable:
    """Decorator attaching an executor to its OpenAI tool schema."""

    def wrap(fn: Callable[..., dict]) -> Callable[..., dict]:
        name = schema["function"]["name"]
        TOOL_FUNCS[name] = fn
        fn.openai_schema = schema  # type: ignore[attr-defined]
        return fn

    return wrap


# ---------------------------------------------------------------------------
# READ tools — no guard, no mutation
# ---------------------------------------------------------------------------


@_register(
    {
        "type": "function",
        "function": {
            "name": "read_inbox",
            "description": "Read all emails in the inbox.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
)
def read_inbox(args: dict, env: Env, guard: Guard) -> dict:
    emails = [
        {"subject": e.subject, "body": e.body, "id": e.id}
        for e in sorted(env.emails, key=lambda e: e.id)
    ]
    return {"count": len(emails), "emails": emails}


@_register(
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
)
def read_file(args: dict, env: Env, guard: Guard) -> dict:
    files = env.files
    name = args.get("name", "")
    content = files.get(name)
    if content is None:
        return {"error": f"file not found: {name}", "files": sorted(files)}
    return {"name": name, "content": content}


@_register(
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search personal notes by keyword.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
)
def search_notes(args: dict, env: Env, guard: Guard) -> dict:
    q = args.get("query", "")
    hits = env.search_notes(q)
    return {
        "query": q,
        "count": len(hits),
        "matches": [{"name": n, "text": t} for n, t in hits],
    }


# ---------------------------------------------------------------------------
# SIDE-EFFECT tools — guarded
# ---------------------------------------------------------------------------


@_register(
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "recipient address"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    }
)
def send_email(args: dict, env: Env, guard: Guard) -> dict:
    req = ActionRequest("send_email", dict(args))
    try:
        guard.check(req)
    except GuardError as e:
        return {"status": "blocked", "reason": e.reason}
    # guard passed — apply the real change to the env
    email = env.send_email(str(args["to"]), args["subject"], args["body"])
    return {"status": "sent", "to": email.to, "email_id": email.id}


@_register(
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create an event on the calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["title", "time"],
            },
        },
    }
)
def create_event(args: dict, env: Env, guard: Guard) -> dict:
    req = ActionRequest("create_event", dict(args))
    try:
        guard.check(req)
    except GuardError as e:
        return {"status": "blocked", "reason": e.reason}
    event = env.create_event(args["title"], args["time"])
    return {"status": "created", "event_id": event.id, "title": event.title}


# ---------------------------------------------------------------------------


def tool_schemas() -> list[dict]:
    """OpenAI tool schemas for every registered tool, in fixed order."""
    return [fn.openai_schema for fn in TOOL_FUNCS.values()]  # type: ignore[attr-defined]


def make_dispatcher(env: Env, guard: Guard) -> Callable[[str, dict], dict]:
    """Bind env+guard once; the agent loop calls this per tool call."""
    return lambda name, args: dispatch_tool(name, args, env, guard)


def dispatch_tool(name: str, args: dict, env: Env, guard: Guard) -> dict:
    """Execute one tool call; returns the dict fed back to the model."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    return fn(args, env, guard)
