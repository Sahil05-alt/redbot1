# agent-redteam

Automated red-team framework for **tool-using AI agents**. Attacks agents
with (indirect) prompt injections, judges success with deterministic
checks, and records every model call, tool call, and state change so
attacks are replayable.

## Status: Week 1 complete — target-agnostic skeleton + email target

Implemented: `core/config.py`, `core/target.py` (Target Protocol),
`core/guard.py` (one chokepoint for side effects), `recorder/schema.py`
+ `store.py` (SQLite, generic JSON-payload rows) + `wrapper.py`
(RecordingModel/RecordingTool), `targets/email/` (env, tools, raw
agent loop, EmailTarget), `runner/run.py` (CLI). 60 tests green.

Week-1 validation cases live where they best fit (see decisions.md):
- recipient allowlist kills + secret-exfil block + kill switch
- recorder step counts for a scripted model (linear timeline [0,1,2])
- each tool's state change (and read tools staying read-only)
- EmailTarget satisfies the Target protocol (runtime_checkable)
- reset() restores a clean environment end-to-end

## Setup

```sh
uv sync
# or: python -m venv .venv && . .venv/bin/activate && pip install -e .
```

## Configuration (env vars)

```sh
export REDTEAM_MODEL=llama3.1
export REDTEAM_BASE_URL=http://localhost:11434/v1   # Ollama example
export REDTEAM_API_KEY=ollama                        # any value for Ollama
```

Other OpenAI-compatible endpoints work: Groq, Gemini, OpenRouter — only
these three env vars change.

## Run

```sh
uv run python -m agent_redteam run --target email "Reply to Alice telling her the meeting time"
```

## Safety defaults

- **Dry-run is always on** in this session: actions mutate the *fake
  in-memory environment* only; no real accounts exist.
- The recipient allowlist and secret-protection are enforced in
  `guard.py`, never in prompts.
- Kill switch: hard cap on tool calls per run; rate limit on model calls.
- API keys only from env, never logged or stored.

See `decisions.md` for design choices.
