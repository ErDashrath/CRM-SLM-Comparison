"""Per-model agent loop.

The selected local model performs both the tool decision and final answer
generation. There is no external runtime orchestrator. The loop accepts a
model handle so the UI can preserve its one-GGUF-at-a-time VRAM policy.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any
from uuid import uuid4

from common.agent.contracts import AgentTurn
from common.crm_agent import retrieve
from common.memory.budget import BudgetPolicy, prepare_prompt_context
from common.telemetry import TelemetrySink, default_sink, text_fingerprint


def run_agent_turn(
    handle: Any,
    *,
    question: str,
    variant_name: str,
    account_id: str | None,
    conversation_history: list[dict],
    system_prompt: str,
    build_prompt: Callable[[str, list[dict], dict, list[dict]], str],
    clean_answer: Callable[[str, dict], str],
    max_tokens: int = 600,
    budget_policy: BudgetPolicy | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    telemetry: TelemetrySink | None = None,
) -> AgentTurn:
    """Run planning, validated tool execution, and final synthesis once."""
    telemetry = telemetry or default_sink()
    turn_id = turn_id or str(uuid4())
    debug_trace: dict[str, Any] = {
        "turn_id": turn_id,
        "variant": variant_name,
        "account_scope": account_id,
        "model_max_tokens": max_tokens,
    }
    telemetry.emit(
        "crm.turn.started",
        session_id=session_id,
        turn_id=turn_id,
        variant=variant_name,
        account_scope=account_id,
        question=question,
    )
    started = perf_counter()
    tool_result, retrieved, tool_trace = retrieve(
        question,
        account_id=account_id,
        planner_handle=handle,
        conversation_history=conversation_history,
        telemetry=telemetry,
        session_id=session_id,
        turn_id=turn_id,
        variant_name=variant_name,
        debug_trace=debug_trace,
    )
    telemetry.emit(
        "evidence.collected",
        session_id=session_id,
        turn_id=turn_id,
        variant=variant_name,
        tool=tool_result.get("tool"),
        tool_status=tool_result.get("status"),
        tool_row_count=len(tool_result.get("rows") or []),
        retrieved_count=len(retrieved),
        retrieved_ids=[item.get("document_id") for item in retrieved if item.get("document_id")],
        trace_steps=len(tool_trace),
    )
    if tool_result.get("status") == "needs_clarification" and not retrieved:
        telemetry.emit(
            "crm.turn.completed",
            session_id=session_id,
            turn_id=turn_id,
            variant=variant_name,
            model_used=False,
            status="needs_clarification",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            answer=tool_result.get("answer", ""),
        )
        return AgentTurn(
            question=question,
            variant=variant_name,
            answer=tool_result.get("answer", ""),
            tool_result=tool_result,
            retrieved=[],
            tool_trace=tool_trace,
            model_used=False,
            debug_trace={**debug_trace, "status": "needs_clarification"},
        )

    history_for_prompt, limits = prepare_prompt_context(
        question,
        conversation_history,
        build_prompt,
        tool_result,
        retrieved,
        policy=budget_policy or BudgetPolicy(reserved_output=max_tokens),
    )
    telemetry.emit(
        "budget.checked",
        session_id=session_id,
        turn_id=turn_id,
        variant=variant_name,
        **limits,
    )
    if limits["hard_limit_reached"]:
        telemetry.emit(
            "crm.turn.completed",
            session_id=session_id,
            turn_id=turn_id,
            variant=variant_name,
            model_used=False,
            status="context_limit_reached",
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        return AgentTurn(
            question=question,
            variant=variant_name,
            answer=(
                "This request is too large for the current model context. "
                "Please narrow the account, date range, or question."
            ),
            tool_result=tool_result,
            retrieved=retrieved,
            tool_trace=tool_trace,
            model_used=False,
            limits=limits,
            debug_trace={
                **debug_trace,
                "status": "context_limit_reached",
                "budget_limits": limits,
            },
        )

    prompt = build_prompt(question, history_for_prompt, tool_result, retrieved)
    debug_trace.update(
        {
            "status": "model_generation",
            "budget_limits": limits,
            "final_system_prompt": system_prompt,
            "final_prompt": prompt,
            "prompt_history": history_for_prompt,
        }
    )
    model_started = perf_counter()
    raw = handle.generate(system_prompt, prompt, max_tokens=max_tokens)
    cleaned = clean_answer(raw, tool_result)
    debug_trace["final_raw_output"] = raw
    debug_trace["final_cleaned_output"] = cleaned
    telemetry.emit(
        "model.final_answer",
        session_id=session_id,
        turn_id=turn_id,
        variant=variant_name,
        input_tokens_estimate=limits.get("prompt_tokens_estimate"),
        output_chars=len(raw or ""),
        duration_ms=round((perf_counter() - model_started) * 1000, 2),
        answer=raw or "",
    )
    telemetry.emit(
        "crm.turn.completed",
        session_id=session_id,
        turn_id=turn_id,
        variant=variant_name,
        model_used=True,
        status="ok",
        duration_ms=round((perf_counter() - started) * 1000, 2),
        answer=raw or "",
        answer_fingerprint=text_fingerprint(raw or ""),
    )
    return AgentTurn(
        question=question,
        variant=variant_name,
        answer=cleaned,
        tool_result=tool_result,
        retrieved=retrieved,
        tool_trace=tool_trace,
        model_used=True,
        limits=limits,
        debug_trace=debug_trace,
    )
