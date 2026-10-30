# decisions.md — design log

## Week 1, step 1: project setup

- **Package layout**: single package `agent_redteam/` at the project root
  so `python -m agent_redteam run ...` works and nothing needs
  namespace packages. `core/`, `recorder/`, `targets/`, `runner/`,
  `tests/` as specified.
- **Config in `core/config.py`**: three env vars (REDTEAM_MODEL,
  REDTEAM_BASE_URL, REDTEAM_API_KEY) + optional call caps; api key is a
  `SecretStr` so prompting/logging it can never leak.
- **Defaults chosen** (Ollama-friendly): base_url
  `http://localhost:11434/v1`, model `llama3.1`, api_key `ollama` — so a
  local Ollama run works with zero env vars. Switching providers is just
  changing the three vars.
- Added `max_tool_calls` / `max_model_calls` to config now (defaults 30 /
  60) because both are env-configurable limits that guard.py (step 3) and
  the agent loop (step 6) will consume.
- Toolchain: `uv` + `pyproject.toml`, deps `openai` + `pydantic` only,
  dev dep `pytest`. No agent frameworks.

## Week 1, step 3: `core/guard.py`

- **Guard is a policy object, not an executor.** It only validates
  (`check(ActionRequest) -> None`, raises `GuardError`); the target's
  tools layer applies state changes *after* the check passes. Keeps the
  guard generic — the later coding agent reuses it for `file_write`
  etc. without touching core.
- **Kill switch checked before rules, call counted only when allowed**
  and none of these call it. A blocked action does not consume budget;
  every attempt (blocked or not) is recorded.
- **Immutable `ActionRequest` dataclass** (frozen): the guard gets a
  snapshot of exactly what the model sent.
- **`recipient_allowlist()` is a plain `CheckFn`** (returns `None` or a
  reason string). Rules are per-`action_type` lists so new targets
  register their own rules without changing `Guard`.
- Env vars: check kill switch (`REDTEAM_MAX_TOOL_CALLS`, default 30)
  → allowlist → record. Well-formedness checked before allowlist so
  the error says *why* an address is rejected (invalid vs not allowed).
- `GuardError` carries `action_type` + `reason` so later judge rules can
  detect "blocked by policy" as a signal deterministically.

## Week 1, step 4: `targets/email/env.py`

- **Env = state + primitives.** `send_email`, `create_event`,
  `search_notes`, plus `state()` (deep copy) and `reset()`. `tools.py`
  (next step) is the model-facing shell; it checks the guard *before*
  calling any mutating primitive, so side effects stay behind one choke
  point while Env stays credential- and network-free.
- **`state()` includes the secret deliberately**: judge rules must
  deterministically check whether the secret leaks into outgoing mail,
  and per-target judges read the target's `state()`. Snapshot is a deep
  copy, so writes to it never touch the live env.
- **Sent mail is recorded in `env.sent`** — dry-run applies actions to
  the fake env (a judge scans `sent`, not the inbox); `emails` stays the
  pristine inbox.
- Seed data is deliberately benign for now (5 emails, 3 files, 3 notes);
  adversarial content arrives in the attack weeks.
- **Note for the user**: `core/target.py` (Step 2) is missing on disk —
  the earlier step-2 summary was never backed by a real write. Backfill
  it before step 6 (`agent.py`) since EmailTarget implements it.

## Week 1, step 2 (backfill): `core/target.py`

- `@runtime_checkable class Target(Protocol)` with exactly the six
  members agreed in the plan: `name`, `tools()`, `run(task) -> None`,
  `state()`, `inject_points()`, `reset()`.
- `@runtime_checkable` so the runner can `isinstance(target, Target)`
  before loading a target with no subclassing required by targets.
- Docstrings encode the two rules that matter most: side effects must go
  through the guard; results live in the recorder + `state()`, not in
  `run()`'s return value. That keeps the recorder as the single
  source of truth for replay/judging.
- Tests cover the full member map, runtime conformance of a duck-typed
  class and *non*-conformance of a class missing required members. An
  `EmailTarget` conformance test will be added when the EmailTarget
  exists (step 6).
- Lesson (recorded): test protocol attribute declarations via
  `Target.__annotations__`, not `hasattr` — annotation-only members
  don't exist as class attributes.

## Week 1, step 5: `targets/email/tools.py`

- **Decorator registry** (`@_register(schema)`) keeps each tool's OpenAI
  JSON schema next to its executor, and builds `tool_schemas()` +
  `dispatch_tool(name, args...)` in fixed registration order — no
  hand-maintained dict of schemas.
- **Two tool kinds, one rule**: read-only tools (`read_inbox`,
  `read_file`, `search_notes`) never touch the guard and never mutate;
  side-effect tools (`send_email`, `create_event`) build an
  `ActionRequest` and call `guard.check()` BEFORE env primitives run.
- **Blocked calls are visible to the model** as
  `{"status": "blocked", "reason": ...}` — this is what makes
  "attack attempted but denied" replayable later, and it does NOT
  consume kill-switch budget (the guard never counts denials).
- Tool results are JSON-serialisable dicts (verified in tests) so the
  recorder can store them verbatim in step 7.

## Week 1, step 6: `targets/email/agent.py`

- **Raw tool-calling loop**: `run(task)` builds [system, user], calls
  the model, appends the assistant message (with `tool_calls`, re-derived
  into plain dicts so both SDK messages and test fakes work), executes
  each tool call via `dispatch_tool`, and loops until a final answer, the
  guard kill-switch blocks everything, or `ModelCallLimitExceeded`.
- **Recorder seam = injectable `model_fn`** (constructor arg). Default is
  a thin OpenAI-SDK wrapper over `chat.completions.create` using config.
  Runner (step 8) can pass a recording wrapper; the loop and EmailTarget
  never import recorder code, and the recorder never imports email code.
- **Rate limit = raise-first** (`max_model_calls` from config,
  counter incremented before each call). kill switch stays inside the
  guard where blocked calls return `{"status": "blocked", ...}` to the
  model and never raise at loop level; `reset()` clears env + call
  counters + `last_answer`.
- **Allowlist nuance (bug found by tests)**: `recipient_allowlist()` takes
  DOMAINS; `ALLOWED_RECIPIENTS` are addresses, so agent.py derives
  `ALLOWED_DOMAINS = {addr.split("@",1)[1] ...}`. Test lesson: a fake
  history slice for turn N contains all of turn N-1's appended messages
  too — assert against the right index.
- `SYSTEM_PROMPT` deliberately says nothing about allowed recipients or
  the secret: enforcement lives in `guard.py`, not the prompt.

## Week 1, step 7: `recorder/schema.py` + `recorder/store.py`

- **Trace schema: chose option A** (two generic tables, JSON payload
  columns, pydantic at the boundary) over B (normalized per-event
  tables). Rationale: the recorder must stay target-agnostic forever;
  JSON-columns never need a migration when a target adds a field, and
  later weeks need linear order + timestamps, not relational joins.
- `TraceRun` (run_id, target, task, model, started_at, finished_at) and
  `TraceStep` (run_id, step_index, step_type llm|tool, input, output,
  tool_name, tool_args, timestamp, duration_ms, tokens).
  `step_type` is constrained twice: pydantic literal + SQL CHECK.
- **Store contract**: in = schema objects, out = schema objects; payloads
  JSON-encode anything non-basic. `steps()` orders by step_index;
  `list_runs()` newest-first. `TraceStep` validators defensively copy
  dicts so later mutation can't rewrite history before insert.
- **No credential fields anywhere**: the API key never reaches this
  layer (config is built at runner level and stays a SecretStr); a test
  asserts no key-ish field names exist on the schemas.

## Week 1, step 8: `recorder/wrapper.py` + `runner/run.py`

- **Recording wrappers are transparent**: `RecordingModel` /
  `RecordingTool` return exactly what the wrapped call returned, writing
  one step each. A shared `StepIndex` counter makes all steps ONE linear
  timeline (llm=0, tool=1, tool=2, …) — the seam current replay needs.
- **Agent seam change (smallest possible)**: EmailTarget now keeps
  `base_model_fn` / `base_tool_fn` (raw) and `model_fn` / `tool_fn`
  (replaceable). The runner installs recording wrappers on public
  attributes; the loop code itself is unchanged. `reset()` unwraps.
- **`ModelCallLimitExceeded` moved to `core/config.py`** (next to the
  limit knobs, re-exported from agent.py) so the runner catches it
  without importing email code — the runner must stay target-agnostic.
- **Registry in the runner only**: `_registry()` maps "email" ->
  EmailTarget. New targets add one line there; nothing else in the
  framework changes. The CLI errors (usage exit 2) for unknown names.
- `run_task()` records runs begin/finish even when the loop crashes or
  is rate-limited (finally), prints the step timeline with durations,
  and returns 1 when HALTED so batch runs can detect failures.
- CLI: `python -m agent_redteam run --target email "<task>"` with
  `--db` (default `traces.db`), `--max-tool-calls`, `--max-model-calls`
  which override the env-configured caps via `model_copy`.
- Test refactor: scripted-model fakes moved to tests/test_support.py so
  wrapper, agent and CLI tests share one source of truth. Lesson: the
  CLI test must set REDTEAM_MODEL in env (config is read at run time,
  not test time).

## Week 1, step 9: test sweep + closeout

- **Five required scenarios land where they test best**, not in one
  mega-file: recipient-allowlist kills → test_guard + agent-level
  secret-exfil + recorder-level blocked-record; step counts for a mocked
  model → test_shared_index_linear + offline CLI end-to-end; per-tool
  state changes → test_email_env/test_email_tools; Target-protocol
  conformance → test_target_protocol + test_email_agent (EmailTarget
  isinstance Target); reset() → env-level and agent-level tests.
- 60 tests total, all offline (no network, no credentials anywhere).
- **Gap closed this step**: read-tool return payloads (read_inbox count,
  read_file found/missing + unknown-tool error, search_notes hits) and a
  tautological assert in test_llm_step_payload replaced with a real
  JSON parse of the recorded tool arguments.
- Final Week-1 layout: config / Target Protocol / guard in core; generic
  recorder (schema, store, recording wrappers); registry-only runner;
  the email target implementing the protocol. Next weeks (attacks,
  deterministic judge rules, replay, viewer) extend via the registry and
  per-target rules folders — nothing core gets rewritten.
- **Found + fixed in closeout**: `agent_redteam/__main__.py` was missing,
  so `python -m agent_redteam run ...` failed at the last step. Smoke
  test against a dead endpoint confirms the whole chain: CLI ends the
  run cleanly on crash (RUN CRASHED + trace persisted with finished_at,
  0 steps printed). With Ollama/any OpenAI-compatible endpoint up, the
  same command records a full timeline.
