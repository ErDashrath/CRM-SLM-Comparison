"""Small, validated tool-calling layer for the conversational CRM UI.

The assistant should not receive the whole CRM on every turn.  It first
selects one or more read-only tools, the application executes those tools,
and only the returned evidence is sent to the answer model.

The planner is deliberately optional.  Small local models are not always
reliable at emitting JSON, so an invalid plan falls back to the existing
deterministic CRM router rather than producing an ungrounded answer.
"""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from common.conversation_memory import resolve_question
from common.crm_query import schema_catalog
from common.tools.registry import CRM_TOOL_REGISTRY
from common.telemetry import TelemetrySink


TOOL_DEFINITIONS = CRM_TOOL_REGISTRY.definitions()


def tool_prompt(question: str, account_id: str | None = None) -> str:
    scope = account_id or "portfolio-wide"
    return (
        "Use the single read-only CRM tool to answer the question. "
        "Return JSON only in this exact shape: "
        '{"tool_calls":[{"name":"crm","arguments":{"question":"...","plan":{...}}}]}. '
        "The plan is semantic, never SQL. Choose one entity, one operation, and only fields from the catalog. "
        "Use count for exact totals, list/search for records, lookup for one account, and aggregate for grouped totals. "
        "The CRM tool combines authoritative structured data with relevant evidence. "
        "Call it at most once. Never invent tool names or arguments.\n\n"
        f"Scope: {scope}\nQuestion: {question}\n"
        f"Available tool: {json.dumps(TOOL_DEFINITIONS)}\n"
        f"Schema catalog: {json.dumps(schema_catalog(), separators=(',', ':'))}"
    )


def _json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_tool_calls(raw: str) -> list[dict[str, Any]]:
    """Validate the model's plan against the allow-list."""
    obj = _json_object(raw)
    if not obj or not isinstance(obj.get("tool_calls"), list):
        return []
    calls = []
    for call in obj["tool_calls"][:2]:
        if not isinstance(call, dict) or call.get("name") not in CRM_TOOL_REGISTRY.names():
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            continue
        if call["name"] == "crm" and isinstance(args.get("question"), str):
            safe_args = {"question": args["question"][:1000]}
            if isinstance(args.get("plan"), dict):
                safe_args["plan"] = args["plan"]
            calls.append({"name": "crm", "arguments": safe_args})
    return calls[:1]


def execute_tool_calls(
    calls: list[dict[str, Any]],
    question: str,
    account_id: str | None = None,
    *,
    telemetry: TelemetrySink | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    variant_name: str | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    """Execute the one CRM tool and split its unified result for the UI."""
    tool_started = perf_counter()
    call = calls[0] if calls else {"name": "crm", "arguments": {"question": question}}
    args = call.get("arguments", {"question": question})
    if call.get("name") != "crm":
        call = {"name": "crm", "arguments": {"question": question}}
        args = call["arguments"]
    tool_result = CRM_TOOL_REGISTRY.execute("crm", args, account_scope=account_id)
    hits = list(tool_result.get("evidence") or [])
    trace = [{
        "tool": "crm",
        "status": tool_result.get("status", "ok"),
        "arguments": args,
        "structured_tool": tool_result.get("structured_tool"),
        "result_count": len(tool_result.get("rows") or []),
        "evidence_count": len(hits),
        "fallback": not calls,
    }]
    if telemetry:
        telemetry.emit(
            "tool.execute",
            session_id=session_id,
            turn_id=turn_id,
            variant=variant_name,
            tool="crm",
            status=tool_result.get("status", "ok"),
            structured_tool=tool_result.get("structured_tool"),
            row_count=len(tool_result.get("rows") or []),
            evidence_count=len(hits),
            fallback=not calls,
            duration_ms=round((perf_counter() - tool_started) * 1000, 2),
            arguments=args,
        )
        telemetry.emit(
            "retrieval.run",
            session_id=session_id,
            turn_id=turn_id,
            variant=variant_name,
            query=args.get("question", question),
            account_scope=account_id,
            candidate_count=len(hits),
            retrieval_skipped=not hits,
        )
    return tool_result, hits, trace


def retrieve(
    question: str,
    account_id: str | None = None,
    planner_handle: Any | None = None,
    conversation_history: list[dict] | None = None,
    *,
    telemetry: TelemetrySink | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    variant_name: str | None = None,
    debug_trace: dict[str, Any] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    """Plan, validate, and execute tools; always return a safe fallback."""
    effective_question, memory = resolve_question(question, conversation_history, account_id)
    if debug_trace is not None:
        debug_trace.update(
            {
                "route": "crm",
                "original_question": question,
                "effective_question": effective_question,
                "memory": {
                    "subject": memory.subject,
                    "intent": memory.intent,
                    "resolved": effective_question != question,
                },
            }
        )
    calls: list[dict] = []
    planning_error = None
    if planner_handle is not None:
        planner_started = perf_counter()
        planner_system_prompt = "You are a CRM tool planner. Output JSON only."
        planner_user_prompt = tool_prompt(effective_question, account_id)
        if debug_trace is not None:
            debug_trace["planner_system_prompt"] = planner_system_prompt
            debug_trace["planner_user_prompt"] = planner_user_prompt
        try:
            raw = planner_handle.generate(
                planner_system_prompt,
                planner_user_prompt,
                max_tokens=220,
            )
            if debug_trace is not None:
                debug_trace["planner_raw_output"] = raw
            calls = parse_tool_calls(raw)
            if not calls:
                planning_error = "planner_output_invalid"
        except Exception as exc:  # model planning must never block retrieval
            planning_error = type(exc).__name__
        if telemetry:
            telemetry.emit(
                "model.tool_decision",
                session_id=session_id,
                turn_id=turn_id,
                variant=variant_name,
                status="ok" if calls else "fallback",
                parse_status="parsed" if calls else planning_error,
                selected_tools=[call["name"] for call in calls],
                duration_ms=round((perf_counter() - planner_started) * 1000, 2),
            )
    if not calls:
        calls = [{"name": "crm", "arguments": {"question": effective_question}}]
    else:
        calls = calls[:1]
    result, hits, trace = execute_tool_calls(
        calls,
        effective_question,
        account_id,
        telemetry=telemetry,
        session_id=session_id,
        turn_id=turn_id,
        variant_name=variant_name,
    )
    if debug_trace is not None:
        debug_trace["validated_tool_calls"] = calls
        debug_trace["tool_result"] = result
        debug_trace["tool_trace"] = trace
        debug_trace["retrieved_documents"] = hits
    if planning_error:
        trace.insert(0, {"tool": "planner", "status": "fallback", "reason": planning_error})
    trace.insert(0, {
        "tool": "conversation_memory",
        "status": "resolved" if effective_question != question else "direct",
        "subject": memory.subject,
        "intent": memory.intent,
        "resolved_question": effective_question,
    })
    return result, hits, trace
