"""Schema-driven CRM query service.

The conversational models see one ``crm`` tool. They produce a small
semantic query plan; this module validates that plan and executes only SQL
declared in the schema registry. Natural-language compilation exists only
as a compatibility path for callers that do not yet have a planner.

There is deliberately no question-specific SQL routing here. Adding a field
or relationship means changing the registry, not adding another branch to a
large router.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common.crm_store import query_rows


Entity = Literal["accounts", "contacts", "leads", "opportunities", "communications"]
Operation = Literal["lookup", "list", "count", "aggregate", "search"]


class QueryPlan(BaseModel):
    """Safe semantic request emitted by a model or compatibility compiler."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    entity: Entity
    operation: Operation = "lookup"
    account_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    requested_fields: list[str] = Field(default_factory=list, max_length=30)
    group_by: list[str] = Field(default_factory=list, max_length=4)
    limit: int = Field(default=20, ge=1, le=50)
    include_evidence: bool = True


class QueryPlanError(ValueError):
    """Raised when a model plan cannot be executed safely."""


class FieldSpec:
    def __init__(self, sql: str, label: str):
        self.sql = sql
        self.label = label


class EntitySpec:
    def __init__(
        self,
        *,
        source: str,
        fields: dict[str, FieldSpec],
        default_fields: tuple[str, ...],
        filters: dict[str, str],
    ):
        self.source = source
        self.fields = fields
        self.default_fields = default_fields
        self.filters = filters


SCHEMA: dict[str, EntitySpec] = {
    "accounts": EntitySpec(
        source="customers c",
        fields={
            "account_id": FieldSpec("c.name", "account_id"),
            "account": FieldSpec("c.customer_name", "account"),
            "industry": FieldSpec("c.industry", "industry"),
            "segment": FieldSpec("c.customer_group", "segment"),
            "region": FieldSpec("c.territory", "region"),
            "account_health": FieldSpec("c.custom_account_health", "account_health"),
            "owner": FieldSpec("c.account_manager", "owner"),
            "account_owner": FieldSpec("c.account_manager", "account_owner"),
            "account_owner_role": FieldSpec("c.custom_account_owner_role", "account_owner_role"),
            "employee_count": FieldSpec("c.custom_employee_count", "employee_count"),
            "annual_revenue": FieldSpec("c.custom_annual_revenue", "annual_revenue"),
        },
        default_fields=("account_id", "account", "industry", "account_health", "owner"),
        filters={
            "account_id": "c.name",
            "account": "c.customer_name",
            "account_name": "c.customer_name",
            "account_health": "c.custom_account_health",
            "industry": "c.industry",
        },
    ),
    "contacts": EntitySpec(
        source=(
            "contacts c JOIN contact_links cl ON cl.contact = c.name "
            "AND cl.link_doctype = 'Customer' "
            "JOIN customers a ON a.name = cl.link_name"
        ),
        fields={
            "contact_id": FieldSpec("c.name", "contact_id"),
            "name": FieldSpec("c.full_name", "name"),
            "account": FieldSpec("a.customer_name", "account"),
            "account_id": FieldSpec("a.name", "account_id"),
            "role": FieldSpec("c.designation", "role"),
            "contact_type": FieldSpec("c.custom_contact_type", "contact_type"),
            "engagement_level": FieldSpec("c.custom_engagement_level", "engagement_level"),
            "notes": FieldSpec("c.custom_notes", "notes"),
        },
        default_fields=("account", "name", "role", "contact_type", "engagement_level"),
        filters={
            "account_id": "a.name",
            "account": "a.customer_name",
            "account_name": "a.customer_name",
            "contact_type": "c.custom_contact_type",
            "role": "c.designation",
            "engagement_level": "c.custom_engagement_level",
        },
    ),
    "leads": EntitySpec(
        source="leads l JOIN customers a ON a.name = l.customer",
        fields={
            "lead_id": FieldSpec("l.name", "lead_id"),
            "name": FieldSpec("l.lead_name", "name"),
            "account": FieldSpec("a.customer_name", "account"),
            "account_id": FieldSpec("a.name", "account_id"),
            "company_name": FieldSpec("l.company_name", "company_name"),
            "status": FieldSpec("l.status", "status"),
            "industry": FieldSpec("l.industry", "industry"),
            "territory": FieldSpec("l.territory", "territory"),
        },
        default_fields=("name", "account", "company_name", "status", "industry", "territory"),
        filters={"account_id": "a.name", "account": "a.customer_name", "status": "l.status"},
    ),
    "opportunities": EntitySpec(
        source="opportunities o JOIN customers a ON a.name = o.customer",
        fields={
            "opportunity_id": FieldSpec("o.name", "opportunity_id"),
            "opportunity": FieldSpec("o.title", "opportunity"),
            "account": FieldSpec("a.customer_name", "account"),
            "account_id": FieldSpec("a.name", "account_id"),
            "stage": FieldSpec("o.sales_stage", "stage"),
            "status": FieldSpec("o.status", "status"),
            "owner": FieldSpec("o.opportunity_owner", "owner"),
            "deal_value_inr": FieldSpec("o.opportunity_amount", "deal_value_inr"),
            "win_probability_pct": FieldSpec("o.probability", "win_probability_pct"),
            "expected_close_date": FieldSpec("o.expected_closing", "expected_close_date"),
            "notes": FieldSpec("o.notes", "notes"),
        },
        default_fields=("account", "opportunity", "stage", "deal_value_inr", "win_probability_pct", "expected_close_date"),
        filters={
            "account_id": "a.name",
            "account": "a.customer_name",
            "account_name": "a.customer_name",
            "stage": "o.sales_stage",
            "status": "o.status",
            "owner": "o.opportunity_owner",
        },
    ),
    "communications": EntitySpec(
        source=(
            "communications m LEFT JOIN opportunities o ON o.name = m.reference_name "
            "LEFT JOIN customers a ON a.name = o.customer"
        ),
        fields={
            "communication_id": FieldSpec("m.name", "communication_id"),
            "subject": FieldSpec("m.subject", "subject"),
            "medium": FieldSpec("m.communication_medium", "medium"),
            "type": FieldSpec("m.communication_type", "type"),
            "sender": FieldSpec("m.sender", "sender"),
            "recipients": FieldSpec("m.recipients", "recipients"),
            "content": FieldSpec("m.content", "content"),
            "date": FieldSpec("m.communication_date", "date"),
            "account": FieldSpec("a.customer_name", "account"),
            "account_id": FieldSpec("a.name", "account_id"),
            "opportunity": FieldSpec("o.title", "opportunity"),
        },
        default_fields=("account", "subject", "medium", "sender", "date", "content"),
        filters={
            "account_id": "a.name",
            "account": "a.customer_name",
            "account_name": "a.customer_name",
            "medium": "m.communication_medium",
            "type": "m.communication_type",
        },
    ),
}


def schema_catalog() -> dict[str, Any]:
    """Compact catalog shown to the planner, generated from the registry."""
    return {
        entity: {
            "operations": ["lookup", "list", "count", "search"] + (["aggregate"] if entity == "opportunities" else []),
            "fields": list(spec.fields),
            "filters": list(spec.filters),
        }
        for entity, spec in SCHEMA.items()
    }


def _account_id_from_text(text: str) -> str | None:
    lowered = text.lower()
    for row in query_rows("SELECT name AS account_id, customer_name AS account FROM customers"):
        if row["account_id"].lower() in lowered or row["account"].lower() in lowered:
            return row["account_id"]
    return None


# Compatibility rules are data, not SQL routing. Model-generated plans bypass
# them; they exist so old callers can migrate without breaking.
_QUESTION_RULES = (
    (r"how many|\bcount\b|\bnumber of\b", r"risk|at.?risk|stalled|threat", "accounts", "count", ("account", "account_health", "owner")),
    (r"risk|at.?risk|stalled|threat", r"account|customer|client|portfolio|which|list|show", "accounts", "search", ("account", "account_health", "owner")),
    (r"how many|\bcount\b|\bnumber of\b", r"e-?mail|communication|activity|activities", "communications", "count", ("medium",)),
    (r"how many|\bcount\b|\bnumber of\b", r"contact|person|people|stakeholder", "contacts", "count", ()),
    (r"how many|\bcount\b|\bnumber of\b", r"account|customer|client", "accounts", "count", ()),
    (r"how many|\bcount\b|\bnumber of\b", r"opportunit|deal", "opportunities", "count", ()),
    (r"in charge|who owns|account manager|managed by|responsible", r".*", "accounts", "lookup", ("account_owner", "account_owner_role")),
    (r"pipeline|stage", r".*", "opportunities", "aggregate", ("stage", "deal_value_inr")),
    (r"risk|at.?risk|stalled|threat", r".*", "opportunities", "search", ("account", "opportunity", "stage", "win_probability_pct", "notes")),
    (r"lead", r".*", "leads", "list", ()),
    (r"contact|person|people|stakeholder|buyer|champion|decision.?maker", r".*", "contacts", "list", ()),
    (r"email|communication|activity|transcript|call|meeting|follow.?up", r".*", "communications", "list", ()),
    (r"opportunit|deal|forecast|win.?prob|close.?date|deadline|timeline", r".*", "opportunities", "list", ()),
    (r"account|customer|client|portfolio", r".*", "accounts", "lookup", ()),
)


def compile_question(question: str, account_scope: str | None = None) -> QueryPlan:
    if not question.strip():
        raise QueryPlanError("A CRM question is required")
    for first, second, entity, operation, fields in _QUESTION_RULES:
        if re.search(first, question, re.IGNORECASE) and re.search(second, question, re.IGNORECASE):
            break
    else:
        entity, operation, fields = "accounts", "search", ()
    account_id = account_scope or _account_id_from_text(question)
    filters: dict[str, Any] = {}
    if account_id:
        filters["account_id"] = account_id
    if entity == "communications" and re.search(r"e-?mail", question, re.IGNORECASE):
        filters["medium"] = "Email"
    if entity == "accounts" and re.search(r"risk|at.?risk|stalled|threat", question, re.IGNORECASE):
        filters["account_health"] = "at_risk"
    return QueryPlan(
        question=question,
        entity=entity,
        operation=operation,
        account_id=account_id,
        filters=filters,
        requested_fields=list(fields),
        group_by=["stage"] if operation == "aggregate" else [],
        include_evidence=(operation not in {"count", "aggregate"} and entity == "communications")
        or operation not in {"count", "aggregate", "list"},
    )


def validate_plan(raw: QueryPlan | dict[str, Any], question: str, account_scope: str | None = None) -> QueryPlan:
    try:
        if isinstance(raw, QueryPlan):
            plan = raw.model_copy(update={"question": question})
        else:
            payload = {**raw, "question": question}
            plan = QueryPlan.model_validate(payload)
    except ValidationError as exc:
        raise QueryPlanError(str(exc)) from exc
    # The application scope is authoritative. A model cannot widen a scoped
    # view by placing another account in the semantic plan or its filters.
    if account_scope:
        filters = {**plan.filters, "account_id": account_scope}
        plan = plan.model_copy(update={"account_id": account_scope, "filters": filters})
    elif plan.account_id and plan.filters.get("account_id") != plan.account_id:
        plan = plan.model_copy(update={"filters": {**plan.filters, "account_id": plan.account_id}})
    spec = SCHEMA[plan.entity]
    if plan.operation == "aggregate" and plan.entity != "opportunities":
        raise QueryPlanError("Aggregate queries are currently supported for opportunities only")
    for key, value in plan.filters.items():
        values = value if isinstance(value, list) else [value]
        if len(values) > 50 or any(not isinstance(item, (str, int, float, bool)) and item is not None for item in values):
            raise QueryPlanError(f"Filter {key} must contain scalar values")
        if any(isinstance(item, str) and len(item) > 500 for item in values):
            raise QueryPlanError(f"Filter {key} is too long")
    unknown_fields = set(plan.requested_fields) - set(spec.fields)
    unknown_filters = set(plan.filters) - set(spec.filters)
    unknown_groups = set(plan.group_by) - set(spec.fields)
    if unknown_fields or unknown_filters or unknown_groups:
        raise QueryPlanError(
            f"Unsupported CRM fields: fields={sorted(unknown_fields)}, "
            f"filters={sorted(unknown_filters)}, groups={sorted(unknown_groups)}"
        )
    if plan.operation == "aggregate" and not plan.group_by:
        raise QueryPlanError("Aggregate queries require group_by")
    return plan


def _where(plan: QueryPlan, spec: EntitySpec) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if plan.account_id and "account_id" not in plan.filters:
        clauses.append("c.name = ?" if plan.entity == "accounts" else "a.name = ?")
        params.append(plan.account_id)
    for key, value in plan.filters.items():
        column = spec.filters[key]
        values = value if isinstance(value, list) else [value]
        if len(values) == 1:
            clauses.append(f"{column} = ?")
            params.append(values[0])
        else:
            clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
            params.extend(values)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _select(spec: EntitySpec, fields: list[str]) -> str:
    return ", ".join(f"{spec.fields[field].sql} AS {field}" for field in fields)


def _lookup_related(row: dict[str, Any], plan: QueryPlan) -> None:
    if plan.entity != "accounts" or not row.get("account_id"):
        return
    account_id = row["account_id"]
    row["opportunity_owners"] = query_rows(
        "SELECT title AS opportunity, opportunity_owner FROM opportunities WHERE customer = ? ORDER BY title",
        (account_id,),
    )
    row["customer_contacts"] = query_rows(
        "SELECT c.full_name AS name, c.designation AS role, c.custom_contact_type AS contact_type, "
        "c.custom_engagement_level AS engagement_level FROM contacts c "
        "JOIN contact_links cl ON cl.contact = c.name "
        "WHERE cl.link_doctype = 'Customer' AND cl.link_name = ? ORDER BY c.custom_contact_type, c.full_name",
        (account_id,),
    )
    row["opportunities"] = query_rows(
        "SELECT title AS opportunity, sales_stage AS stage, probability AS win_probability_pct, "
        "opportunity_amount AS deal_value_inr FROM opportunities WHERE customer = ? ORDER BY title",
        (account_id,),
    )
    row["risk_factors"] = query_rows(
        "SELECT o.title AS opportunity, r.risk_type, r.severity, r.description "
        "FROM opportunity_risk_factors r JOIN opportunities o ON o.name = r.opportunity "
        "WHERE o.customer = ? ORDER BY r.severity DESC, o.title",
        (account_id,),
    )


def _canonical_name(plan: QueryPlan) -> str:
    if plan.entity == "accounts" and plan.filters.get("account_health") == "at_risk":
        return "at-risk accounts"
    if plan.entity == "communications":
        return "emails" if plan.filters.get("medium") == "Email" else "communications"
    return plan.entity


def _compat_tool_name(plan: QueryPlan) -> str:
    """Stable labels for existing UI/tests; semantics come from the plan."""
    if plan.operation == "count":
        return {
            "accounts": "count_accounts",
            "contacts": "count_contacts",
            "opportunities": "count_opportunities",
            "communications": "count_emails" if plan.filters.get("medium") == "Email" else "count_communications",
        }.get(plan.entity, "count_records")
    if plan.entity == "accounts" and plan.operation == "lookup" and "account_owner" in plan.requested_fields:
        return "account_ownership"
    if plan.entity == "contacts" and plan.operation in {"list", "search"}:
        return "contacts_by_account" if plan.account_id else "contacts_all"
    if plan.entity == "leads" and plan.operation in {"list", "search"}:
        return "leads_all"
    if plan.entity == "opportunities" and plan.operation == "aggregate":
        return "pipeline_summary"
    if plan.entity == "opportunities" and plan.operation in {"list", "search"}:
        return "opportunities_all"
    if plan.entity == "communications" and plan.operation in {"list", "search"}:
        return "communications_all"
    return f"{plan.operation}_{plan.entity}"


def _answer(plan: QueryPlan, rows: list[dict[str, Any]]) -> str:
    if plan.operation == "count":
        count = (rows[0].get("count") if rows else 0) or 0
        scope = " for this account" if plan.account_id else " across the portfolio"
        return f"There are {count} {_canonical_name(plan)}{scope}."
    if plan.entity == "accounts" and rows and plan.operation == "lookup":
        owner = rows[0].get("account_owner") or rows[0].get("owner")
        account = rows[0].get("account") or rows[0].get("account_id")
        if owner:
            return f"{account} is owned by {owner}. Customer-side contacts are listed separately in the structured data."
    if plan.operation == "aggregate":
        return f"{plan.entity.title()} grouped by {', '.join(plan.group_by)}."
    if plan.operation in {"list", "search"}:
        return f"{len(rows)} {plan.entity} returned."
    return f"CRM lookup returned {len(rows)} {plan.entity}."


def execute_query(raw_plan: QueryPlan | dict[str, Any], question: str | None = None, account_scope: str | None = None) -> dict[str, Any]:
    question = question or (raw_plan.question if isinstance(raw_plan, QueryPlan) else str(raw_plan.get("question", "CRM query")))
    plan = validate_plan(raw_plan, question, account_scope)
    spec = SCHEMA[plan.entity]
    fields = list(plan.requested_fields) or list(spec.default_fields)
    if plan.entity == "accounts" and "account_id" not in fields:
        fields = list(dict.fromkeys(["account_id", "account", *fields]))
    if plan.operation == "count":
        select = "COUNT(*) AS count"
    elif plan.operation == "aggregate":
        select = ", ".join([_select(spec, plan.group_by), "COUNT(*) AS opportunities", "COALESCE(SUM(o.opportunity_amount), 0) AS value_inr"])
    else:
        select = _select(spec, fields)
    where, params = _where(plan, spec)
    group_sql = f" GROUP BY {', '.join(spec.fields[field].sql for field in plan.group_by)}" if plan.operation == "aggregate" else ""
    if plan.operation == "aggregate":
        order_sql = " ORDER BY value_inr DESC"
    elif plan.entity == "communications":
        order_sql = " ORDER BY m.communication_date DESC"
    else:
        order_sql = ""
    sql = f"SELECT {select} FROM {spec.source}{where}{group_sql}{order_sql} LIMIT ?"
    rows = query_rows(sql, tuple(params) + (plan.limit,))
    if plan.entity == "accounts":
        for row in rows:
            _lookup_related(row, plan)
    return {
        "tool": _compat_tool_name(plan),
        "status": "ok",
        "answer": _answer(plan, rows),
        "rows": rows,
        "structured": {"entity": plan.entity, "operation": plan.operation, "records": rows, "total": len(rows)},
        "plan": plan.model_dump(mode="json"),
        "authoritative": True,
        "coverage_fields": fields,
    }


def execute_question(question: str, account_scope: str | None = None, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a model plan, or compile a legacy free-form question."""
    try:
        if plan is not None:
            raw = {**plan, "question": question}
        else:
            raw = compile_question(question, account_scope)
        return execute_query(raw, question=question, account_scope=account_scope)
    except (QueryPlanError, ValidationError) as exc:
        return {
            "tool": "crm_query",
            "status": "needs_clarification",
            "answer": f"I could not safely map that CRM request: {exc}",
            "rows": [],
            "structured": {"entity": None, "operation": None, "records": [], "total": 0},
            "plan": plan or {},
            "authoritative": False,
            "coverage_fields": [],
        }
