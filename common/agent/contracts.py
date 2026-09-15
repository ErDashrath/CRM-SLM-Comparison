"""Stable contracts shared by the local agent loop and the UI adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    name: str
    status: str
    answer: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    truncated: bool = False


@dataclass
class AgentTurn:
    question: str
    variant: str
    answer: str
    tool_result: dict[str, Any]
    retrieved: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    model_used: bool = True
    is_casual: bool = False
    limits: dict[str, Any] = field(default_factory=dict)
    debug_trace: dict[str, Any] = field(default_factory=dict)

    def as_ui_result(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_result": self.tool_result,
            "retrieved": self.retrieved,
            "tool_trace": self.tool_trace,
            "variant": self.variant,
            "model_used": self.model_used,
            "is_casual": self.is_casual,
            "limits": self.limits,
            "debug_trace": self.debug_trace,
        }
