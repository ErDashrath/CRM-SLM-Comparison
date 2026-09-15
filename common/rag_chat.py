"""
RAG + tool-calling chat pipeline for the direct conversational UI.

This is the NEW pipeline used by the chat interface — completely separate
from the NextBestAction/guardrails pipeline in common/formatting.py and
eval/run_eval.py. That pipeline remains intact for the eval harness and
model-comparison research.

This pipeline:
1. Runs a structured CRM tool (keyword-routed SQL) → structured rows + answer
2. Runs hybrid BM25 RAG retrieval → relevant account/opp/email/transcript docs
3. Assembles both into a compact evidence block
4. Loads the selected model variant (base / kd / sft) and generates a
   plain-language response using the v2_chat conversational system prompt
5. Returns {answer, tool_result, retrieved, model_used, variant}

The model answers conversationally — no JSON output, no guardrail checks,
no NextBestAction parsing. Those belong to the research pipeline.

VRAM note: same load/close pattern as run_one_variant in the UI — one
variant at a time, explicit close() after generate(). Do NOT cache_resource
here; see the ui/app.py module docstring for why.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from uuid import uuid4

import yaml

from common.agent.loop import run_agent_turn
from common.telemetry import default_sink

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHAT_PROMPT_PATH = PROJECT_ROOT / "common" / "system_prompts" / "v2_chat.yaml"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_COMPACT_EVIDENCE_CHARS = 2400  # slightly more than the NextBestAction pipeline;
                                 # the conversational prompt is shorter so we have room

# ---------------------------------------------------------------------------
# Intent classifier — keeps casual messages out of the CRM model
# ---------------------------------------------------------------------------
# The CRM-trained 1.7B models hallucinate CRM content in response to ANY input,
# including "hii". We detect casual / non-CRM messages here and return a direct
# reply without loading the model at all.

_GREETING_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|howdy|greetings|sup|what'?s up|yo+|hola|namaste|"
    r"good\s*(morning|afternoon|evening|night|day)|"
    r"how are you|how r u|how'?s it going|how do you do|"
    r"nice to meet|pleased to meet|"
    r"thanks?|thank you|thx|ty|cheers|great|ok+|okay|cool|got it|noted|"
    r"bye|goodbye|see you|cya|later|"
    r"yes|no|yeah|nope|sure|absolutely|definitely|"
    r"what can you do|what are you|who are you|help me|what do you know)\W*$",
    re.IGNORECASE,
)

_CASUAL_REPLIES: dict[str, str] = {
    "greeting": (
        "Hi! I'm your CRM assistant for Nimbus Systems. "
        "Ask me about accounts, deals, pipeline status, risks, or contacts — "
        "I'll search the CRM and give you a grounded answer."
    ),
    "thanks": "You're welcome! Let me know if you have any CRM questions.",
    "bye": "Goodbye! Come back when you need CRM insights.",
    "help": (
        "I can help with:\n"
        "- **Account health** — which accounts are at risk?\n"
        "- **Deals & pipeline** — stage, value, close dates\n"
        "- **Contacts** — key stakeholders at an account\n"
        "- **Risks** — competitor mentions, stalled deals, discount flags\n\n"
        "Just ask naturally, e.g. *\"What's the status of the Acme deal?\"*"
    ),
    "affirmation": "Got it! What would you like to know about the CRM?",
    "identity": (
        "I'm a CRM assistant powered by Nimbus Systems' local small-language models "
        "(Base, KD, and SFT variants of Qwen3-1.7B). I answer questions about your "
        "accounts, opportunities, contacts, and pipeline using RAG retrieval and "
        "structured CRM queries."
    ),
}


_CRM_KEYWORDS_RE = re.compile(
    r"\b(account|deal|opportunit|pipeline|risk|contact|stage|forecast|"
    r"discount|close|revenue|health|acme|technova|globex|prospect|"
    r"customer|client|email|transcript|win|lose|stall|competitor)\b",
    re.IGNORECASE,
)


def _classify_casual(text: str) -> str | None:
    """
    Return a reply key if the message is casual/non-CRM, else None.
    Returns one of: 'greeting', 'thanks', 'bye', 'help', 'affirmation', 'identity'.
    Returning None means: run the full RAG+model pipeline.
    """
    stripped = text.strip()

    # Always run the full pipeline if there's a CRM keyword
    if _CRM_KEYWORDS_RE.search(stripped):
        return None

    # Handle short greeting typos such as "heloo", "helo", and "hiii".
    # These must be caught before any local CRM model is loaded, otherwise a
    # small fine-tuned model may invent CRM content for a casual message.
    compact = re.sub(r"[^a-z]", "", stripped.lower())
    if len(compact) <= 16:
        greeting_words = ("hi", "hello", "hey", "howdy", "hola", "namaste")
        if any(difflib.SequenceMatcher(None, compact, word).ratio() >= 0.72 for word in greeting_words):
            return "greeting"

    # No CRM keyword — check casual patterns
    if not _GREETING_RE.match(stripped):
        return None  # Unknown short message — let the pipeline handle it

    low = stripped.lower()
    if any(w in low for w in ("bye", "goodbye", "cya", "see you", "later")):
        return "bye"
    if any(w in low for w in ("thank", "thx", "ty", "cheers")):
        return "thanks"
    if any(w in low for w in ("help", "what can you", "what do you")):
        return "help"
    if any(w in low for w in ("who are you", "what are you", "what is this")):
        return "identity"
    if any(w in low for w in ("yes", "no", "yeah", "nope", "sure", "ok", "okay",
                               "cool", "got it", "noted", "great", "absolutely")):
        return "affirmation"
    return "greeting"


def _load_chat_system_prompt() -> str:
    with open(_CHAT_PROMPT_PATH) as f:
        spec = yaml.safe_load(f)
    return spec["system_prompt"]


def _format_tool_result(tool_result: dict) -> str:
    """Turn the structured tool result into a compact text block for the prompt."""
    tool_label = tool_result.get("tool", "crm")
    if tool_result.get("structured_tool"):
        tool_label = f"{tool_label} / {tool_result['structured_tool']}"
    if tool_result["status"] == "needs_clarification":
        return f"[Tool: {tool_label}]\n{tool_result['answer']}"
    lines = [f"[Tool: {tool_label}] {tool_result['answer']}"]
    rows = tool_result.get("rows") or []
    if rows:
        # Show the complete bounded result (the service caps it at 20 rows).
        for row in rows[:20]:
            lines.append("  • " + ", ".join(f"{k}={v}" for k, v in dict(row).items() if v is not None))
        if len(rows) > 20:
            lines.append(f"  … and {len(rows) - 20} more rows")
    return "\n".join(lines)


def _format_rag_hits(hits: list[dict]) -> str:
    """Turn RAG hits into compact evidence blocks."""
    if not hits:
        return "(no retrieved documents)"
    blocks = []
    for hit in hits:
        header = f"[{hit['source_type']} | {hit.get('title', '')} | score {hit.get('score', 0):.2f}]"
        # Truncate very long docs
        text = hit.get("text", "")[:600]
        if len(hit.get("text", "")) > 600:
            text += "…"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _ensure_structured_coverage(answer: str, tool_result: dict) -> str:
    """Prevent the small model from silently dropping query-result records.

    Coverage is driven by the semantic result contract, not by a growing list
    of question-specific tool names.
    """
    rows = tool_result.get("rows") or []
    structured = tool_result.get("structured") or {}
    operation = structured.get("operation") or (tool_result.get("plan") or {}).get("operation")
    if not rows or operation not in {"list", "search", "aggregate"}:
        return answer

    missing: list[str] = []
    for row in rows:
        entity = structured.get("entity")
        label_keys = {
            "accounts": ("account", "name"),
            "contacts": ("name", "account"),
            "leads": ("name", "account"),
            "opportunities": ("opportunity", "account", "name"),
            "communications": ("subject", "account"),
        }.get(entity, ("name", "account", "opportunity", "subject", "stage"))
        label = next((row.get(key) for key in label_keys if row.get(key)), None)
        if not label:
            continue
        label_text = str(label)
        if label_text.lower() not in answer.lower():
            details = []
            for key in (
                "role", "contact_type", "engagement_level", "stage", "account_health",
                "win_probability_pct", "deal_value_inr", "expected_close_date", "date",
            ):
                value = row.get(key)
                if value is not None and str(value) != label_text:
                    details.append(f"{key.replace('_', ' ')}: {value}")
            risks = row.get("risk_factors") or []
            if risks:
                risk_names = [str(item.get("risk_type")) for item in risks if item.get("risk_type")]
                if risk_names:
                    details.append("risk factors: " + ", ".join(dict.fromkeys(risk_names)))
            missing.append(f"- {label_text}" + (f" ({'; '.join(details)})" if details else ""))
    if missing:
        return answer.rstrip() + "\n\nAdditional records from the CRM:\n" + "\n".join(missing)
    return answer


def _clean_model_answer(raw: str, tool_result: dict) -> str:
    """Apply conversational output cleanup after the per-model agent loop."""
    answer = _THINK_RE.sub("", raw or "").strip()
    # Aggregates and counts are authoritative SQL results. The model sees the
    # same result but must not replace an exact value with a top-k estimate.
    structured = tool_result.get("structured") or {}
    operation = structured.get("operation") or (tool_result.get("plan") or {}).get("operation")
    # ``structured_tool`` is retained for callers from the previous contract;
    # the semantic operation is preferred whenever the new envelope exists.
    if not operation and str(tool_result.get("structured_tool", "")).startswith("count_"):
        operation = "count"
    if (tool_result.get("authoritative") or tool_result.get("structured_tool")) and operation in {"count", "aggregate"}:
        return str(tool_result.get("answer") or answer)
    return _ensure_structured_coverage(answer, tool_result)


def _build_chat_user_prompt(
    question: str,
    conversation_history: list[dict],
    tool_result: dict,
    rag_hits: list[dict],
) -> str:
    """
    Assemble the user-turn prompt with:
    - Prior conversation turns (for context continuity)
    - Structured tool result
    - RAG evidence
    - The current question
    """
    # Recent history (last 4 turns to keep context manageable)
    history_lines = []
    for msg in conversation_history[-4:]:
        role = msg["role"].upper()
        content = msg.get("content", "")[:400]  # truncate long history
        history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "(no prior conversation)"

    tool_block = _format_tool_result(tool_result)
    rag_block = _format_rag_hits(rag_hits)

    return (
        "=== Conversation history ===\n"
        f"{history_block}\n\n"
        "=== Structured CRM tool result ===\n"
        f"{tool_block}\n\n"
        "=== Retrieved CRM evidence (RAG) ===\n"
        f"{rag_block}\n\n"
        "=== Current question ===\n"
        f"{question}\n\n"
        "Answer concisely, grounded in the evidence above. "
        "If the evidence is insufficient, say so and ask a follow-up. "
        "For exact counts, the structured CRM tool result is authoritative; "
        "never replace its number with a count inferred from retrieved documents. "
        "For count, list, portfolio, or comparison questions, include every "
        "record returned by the structured CRM tool; do not silently omit a "
        "record while summarizing."
    )


def rag_chat(
    question: str,
    variant_name: str = "base",
    account_id: str | None = None,
    conversation_history: list[dict] | None = None,
    max_tokens: int = 600,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> dict:
    """
    Run a single question through the RAG + tool-calling chat pipeline.

    Args:
        question: The user's current question.
        variant_name: Which model variant to use ("base", "kd", "sft").
        account_id: Scope retrieval to a specific account, or None for portfolio.
        conversation_history: Prior messages [{role, content}, ...] for context.
        max_tokens: Maximum tokens for the response.

    Returns:
        {
            "answer": str,          # plain-language response (think block stripped)
            "tool_result": dict,    # structured tool output
            "retrieved": list,      # RAG hits
            "variant": str,         # which variant was used
            "model_used": bool,     # False if tool returned needs_clarification early
        }
    """
    from models.inference import load, VariantNotBuiltError

    if conversation_history is None:
        conversation_history = []
    turn_id = turn_id or str(uuid4())
    telemetry = default_sink()

    # Step 0: Casual / non-CRM intent check
    # The small CRM-trained models hallucinate CRM content for ANY input.
    # Short-circuit before loading the model for greetings, thanks, etc.
    casual_key = _classify_casual(question)
    if casual_key is not None:
        telemetry.emit(
            "crm.turn.completed",
            session_id=session_id,
            turn_id=turn_id,
            variant=variant_name,
            account_scope=account_id,
            model_used=False,
            status="casual",
            answer=_CASUAL_REPLIES[casual_key],
        )
        return {
            "answer": _CASUAL_REPLIES[casual_key],
            "tool_result": {},
            "retrieved": [],
            "variant": variant_name,
            "model_used": False,
            "is_casual": True,
            "debug_trace": {
                "route": "casual",
                "casual_intent": casual_key,
                "model_calls": [],
            },
        }

    # Load one model handle and reuse it for planning and final synthesis.
    # This keeps the flow cheap on the 4 GB GPU and makes the tool trace
    # visible without loading a second model instance.
    try:
        handle = load(variant_name)
    except VariantNotBuiltError as e:
        telemetry.emit(
            "crm.turn.completed",
            session_id=session_id,
            turn_id=turn_id,
            variant=variant_name,
            account_scope=account_id,
            model_used=False,
            status="model_unavailable",
            answer=str(e),
        )
        return {
            "answer": f"Model '{variant_name}' is not available yet: {e}",
            "tool_result": {},
            "retrieved": [],
            "variant": variant_name,
            "model_used": False,
            "debug_trace": {
                "route": "model_unavailable",
                "error": str(e),
                "model_calls": [],
            },
        }

    try:
        turn = run_agent_turn(
            handle,
            question=question,
            variant_name=variant_name,
            account_id=account_id,
            conversation_history=conversation_history,
            system_prompt=_load_chat_system_prompt(),
            build_prompt=_build_chat_user_prompt,
            clean_answer=_clean_model_answer,
            max_tokens=max_tokens,
            session_id=session_id,
            turn_id=turn_id,
            telemetry=telemetry,
        )
    finally:
        handle.close()
    return turn.as_ui_result()
