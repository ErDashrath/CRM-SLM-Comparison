"""
API backend for the teacher (Phase 1), judge (Phase 4), and research agent
(Phase 5) -- every non-local-inference LLM call in this project goes through
here, selected by the TEACHER_BACKEND env var (default: "openai").

Same generate(system_prompt, user_prompt, max_tokens) -> str signature as
models/inference.py::VariantHandle, so eval/run_eval.py can treat the
teacher/judge and the three local variants uniformly wherever that's useful.

Backend choice, 2026-09-10: no ANTHROPIC_API_KEY exists on this machine yet
(confirmed -- not in the shell env, not in any .env, not in a shell
profile). The user chose to proceed with OpenAIAPIBackend (gpt-4o-mini) for
now rather than wait -- OPENAI_API_KEY is already configured (copied from
SalesIntelligence's .env, where it was used successfully for the same kind
of offline dataset drafting). ClaudeAPIBackend is implemented and ready:
switching TEACHER_BACKEND=claude once a key is added is the only change
needed anywhere in this project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # exact snapshot ID, not the bare alias
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMBackend(Protocol):
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512
    ) -> str: ...


class ClaudeAPIMissingKeyError(RuntimeError):
    pass


class OpenAIAPIMissingKeyError(RuntimeError):
    pass


class ClaudeAPIBackend:
    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeAPIMissingKeyError(
                "TEACHER_BACKEND=claude requires ANTHROPIC_API_KEY. Set it "
                "in this project's .env, or use TEACHER_BACKEND=openai "
                "(already configured) instead."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )


class OpenAIAPIBackend:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIAPIMissingKeyError(
                "TEACHER_BACKEND=openai requires OPENAI_API_KEY. Set it in "
                "this project's .env."
            )
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


def get_teacher_backend(name: Optional[str] = None) -> LLMBackend:
    """Return the backend selected by TEACHER_BACKEND env var (or the `name`
    override). Not cached -- these are cheap to construct (just an API
    client), unlike models/inference.py's VRAM-heavy local variants."""
    backend_name = name or os.environ.get("TEACHER_BACKEND", "claude")
    if backend_name == "openai":
        return OpenAIAPIBackend()
    if backend_name == "claude":
        return ClaudeAPIBackend()
    raise ValueError(
        f"Unknown TEACHER_BACKEND={backend_name!r}; expected 'openai' or 'claude'."
    )
