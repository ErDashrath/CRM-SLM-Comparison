"""Allow-listed tool metadata and execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: dict[str, str]
    handler: Callable[[dict[str, Any], str | None], Any]


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs = {spec.name: spec for spec in specs or []}

    def names(self) -> set[str]:
        return set(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description, "arguments": spec.arguments}
            for spec in self._specs.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any], account_scope: str | None = None) -> Any:
        spec = self._specs.get(name)
        if spec is None:
            raise ValueError(f"Unknown CRM tool: {name}")
        return spec.handler(arguments, account_scope)


def _crm(arguments: dict[str, Any], account_scope: str | None) -> dict:
    from common.crm_tools import run_crm_tool
    from common.crm_retrieval import hybrid_search
    from common.crm_query import schema_catalog

    question = str(arguments.get("question", "")).strip()[:1000]
    plan = arguments.get("plan") if isinstance(arguments.get("plan"), dict) else None
    structured = run_crm_tool(question, account_scope=account_scope, plan=plan)
    effective_plan = structured.get("plan") or {}
    evidence = []
    if structured.get("status") == "ok" and effective_plan.get("include_evidence", True):
        evidence = hybrid_search(question, account_id=account_scope, limit=6)
    return {
        "tool": "crm",
        "structured_tool": structured.get("tool"),  # compatibility label for the UI
        "status": structured.get("status", "ok"),
        "answer": structured.get("answer", ""),
        "rows": structured.get("rows", []),
        "structured": structured.get("structured", {}),
        "plan": effective_plan,
        "schema_catalog": schema_catalog(),
        "authoritative": structured.get("authoritative", False),
        "coverage_fields": structured.get("coverage_fields", []),
        "evidence": evidence,
        "evidence_count": len(evidence),
    }


CRM_TOOL_REGISTRY = ToolRegistry([
    ToolSpec(
        name="crm",
        description=(
            "Execute one safe semantic CRM query. Provide the user's question and, "
            "when possible, a validated plan with entity, operation, filters, and fields. "
            "The result contains authoritative structured records plus relevant evidence."
        ),
        arguments={
            "question": "string",
            "plan": "object (optional: entity, operation, account_id, filters, requested_fields, group_by, limit)",
        },
        handler=_crm,
    ),
])
