"""Deterministic conversation compaction for the local POC."""

from __future__ import annotations


def summarize_history(history: list[dict], max_chars: int = 1800) -> str:
    lines = []
    for message in history:
        role = str(message.get("role", "")).upper()
        content = " ".join(str(message.get("content", "")).split())
        if content:
            lines.append(f"{role}: {content[:600]}")
    summary = "\n".join(lines)
    return summary if len(summary) <= max_chars else summary[: max_chars - 1].rstrip() + "…"
