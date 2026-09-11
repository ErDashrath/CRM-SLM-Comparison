"""Tool-first CRM assistant orchestration."""

from __future__ import annotations

import json
import re

from common.crm_retrieval import hybrid_search
from common.crm_tools import run_crm_tool
from common.formatting import load_system_prompt
from models.inference import load

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def answer_crm_question(question: str, account_id: str | None = None) -> dict:
    """Run structured tools and RAG, then ask the local model to synthesize.

    Exact counts and schema clarifications remain authoritative tool outputs;
    the model receives them as evidence and is explicitly forbidden to invent
    missing CRM entities or values.
    """
    tool_result = run_crm_tool(question)
    retrieved = hybrid_search(question, account_id=account_id)
    if tool_result["status"] == "needs_clarification":
        return {"answer": tool_result["answer"], "tool_result": tool_result, "retrieved": retrieved, "model_used": False}

    evidence = {
        "structured_tool_result": tool_result,
        "retrieved_evidence": retrieved,
    }
    user_prompt = (
        "Answer the user's CRM question using ONLY the evidence below. "
        "Do not invent fields, counts, leads, or actions. If evidence is "
        "insufficient, ask one precise follow-up question. Be concise and "
        "mention which tool/retrieved records support the answer.\n\n"
        f"Question: {question}\nEvidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    handle = load("base")
    try:
        answer = handle.generate(load_system_prompt(), user_prompt, max_tokens=500)
    finally:
        handle.close()
    answer = _THINK_RE.sub("", answer).strip()
    return {"answer": answer, "tool_result": tool_result, "retrieved": retrieved, "model_used": True}