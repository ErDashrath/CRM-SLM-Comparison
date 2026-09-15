"""Context-window and session-budget decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from common.memory.summarizer import summarize_history


@dataclass(frozen=True)
class BudgetPolicy:
    context_window: int = 5120
    reserved_output: int = 600
    soft_limit_ratio: float = 0.75
    max_session_tokens: int = 100_000

    @property
    def prompt_budget(self) -> int:
        return self.context_window - self.reserved_output


def estimate_tokens(text: str) -> int:
    """Conservative fallback counter; exact model tokenizer can replace it."""
    return max(1, (len(text) + 3) // 4) if text else 0


def prepare_prompt_context(
    question: str,
    history: list[dict],
    build_prompt: Callable[[str, list[dict], dict, list[dict]], str],
    tool_result: dict,
    retrieved: list[dict],
    policy: BudgetPolicy | None = None,
) -> tuple[list[dict], dict[str, int | bool | str]]:
    """Trim oldest turns only when the assembled prompt exceeds its budget."""
    policy = policy or BudgetPolicy()
    working = list(history)
    prompt = build_prompt(question, working, tool_result, retrieved)
    initial_tokens = estimate_tokens(prompt)
    dropped = 0
    summarized = False
    while working and estimate_tokens(prompt) > policy.prompt_budget:
        if not summarized and len(working) > 2:
            split_at = max(2, len(working) // 2)
            summary = summarize_history(working[:split_at])
            working = [{"role": "assistant", "content": f"[Earlier conversation summary]\n{summary}"}] + working[split_at:]
            summarized = True
        else:
            if summarized and working:
                # Keep the summary as the durable memory anchor while
                # removing remaining verbatim turns. If even the summary is
                # too large, shrink its prose rather than dropping memory.
                if len(working) > 1:
                    working = [working[0]] + working[2:]
                    dropped += 1
                else:
                    max_chars = max(160, policy.prompt_budget * 4)
                    content = working[0].get("content", "")
                    working[0] = {**working[0], "content": content[:max_chars]}
                    prompt = build_prompt(question, working, tool_result, retrieved)
                    break
            else:
                working = working[2:] if len(working) > 1 else working[1:]
                dropped += 1
        prompt = build_prompt(question, working, tool_result, retrieved)
    final_tokens = estimate_tokens(prompt)
    hard_exceeded = final_tokens > policy.prompt_budget
    status = "ok"
    if dropped or summarized:
        status = "history_compacted"
    if hard_exceeded:
        status = "context_limit_reached"
    return working, {
        "prompt_tokens_estimate": final_tokens,
        "initial_prompt_tokens_estimate": initial_tokens,
        "context_window": policy.context_window,
        "reserved_output_tokens": policy.reserved_output,
        "history_turns_dropped": dropped,
        "history_summary_created": summarized,
        "soft_limit_reached": initial_tokens >= int(policy.context_window * policy.soft_limit_ratio),
        "hard_limit_reached": hard_exceeded,
        "status": status,
    }
