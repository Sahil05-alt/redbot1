"""Runtime configuration, read from environment variables only.

Design notes:
- Credentials come from env vars only. The API key is a SecretStr so it
  never prints, never leaks into logs, and is never written to the
  database or files.
- BASE_URL / MODEL are plain env vars so it is easy to switch between
  OpenAI-compatible providers (Ollama, Groq, Gemini, OpenRouter).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, SecretStr


class ModelCallLimitExceeded(RuntimeError):
    """The per-run model-call rate limit fired (agent loops raise this).

    Lives here (next to the limit knobs) so generic code like the runner
    can catch it without importing any specific target.
    """


class LLMConfig(BaseModel):
    """How the agent loop talks to a model."""

    model: str
    base_url: str
    api_key: SecretStr
    # Kill switch: hard cap on tool calls per run (enforced in guard.py).
    max_tool_calls: int = Field(default=30, ge=1)
    # Rate limit: max model calls per run (enforced in the agent loop).
    max_model_calls: int = Field(default=60, ge=1)

    # Never let the key appear in any repr/log output.
    model_config = {"repr": True}


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_llm_config() -> LLMConfig:
    """Build an LLMConfig from REDTEAM_* environment variables.

    Env vars (all optional except where noted):
      REDTEAM_MODEL         e.g. "llama3.1", "mixtral-8x7b-32768"
      REDTEAM_BASE_URL      e.g. "http://localhost:11434/v1" (Ollama)
      REDTEAM_API_KEY       provider key (Ollama accepts any dummy value)
      REDTEAM_MAX_TOOL_CALLS   optional, default 30
      REDTEAM_MAX_MODEL_CALLS  optional, default 60
    """
    return LLMConfig(
        model=_env("REDTEAM_MODEL", "llama3.1"),
        base_url=_env("REDTEAM_BASE_URL", "http://localhost:11434/v1"),
        api_key=SecretStr(os.environ.get("REDTEAM_API_KEY", "ollama")),
        max_tool_calls=int(os.environ.get("REDTEAM_MAX_TOOL_CALLS", "30")),
        max_model_calls=int(os.environ.get("REDTEAM_MAX_MODEL_CALLS", "60")),
    )
