"""Conversation memory and follow-up query resolution for the CRM agent.

The UI stores turns, but retrieval needs a stable semantic state.  This module
derives that state from the conversation and produces a resolved CRM query
before any tool or RAG lookup runs.  It is deliberately model-independent so
memory remains reliable even when a 1.7B planner emits invalid JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.crm_store import query_rows


@dataclass(frozen=True)
class ConversationMemory:
    last_user_question: str = ""
    subject: str = ""
    account_id: str | None = None
    intent: str = ""


_SUBJECT_TERMS = {
    # Risk is a qualifier that must survive follow-ups such as
    # "which are they?" after "which accounts are at risk?".
    "risks": r"risk|at.?risk|stalled|stall|threat|issue",
    "contacts": r"contact|person|people|stakeholder|buyer|champion|decision.?maker",
    "accounts": r"account|customer|client|portfolio",
    "opportunities": r"opportunit|deal|pipeline|forecast|win probability|stage",
    "activities": r"email|transcript|call|meeting|follow.?up|latest|recent|said",
}


def _latest_user(history: list[dict]) -> str:
    return next(
        (str(m.get("content", "")).strip() for m in reversed(history) if m.get("role") == "user"),
        "",
    )


def _subject(text: str) -> str:
    for subject, pattern in _SUBJECT_TERMS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return subject
    return ""


def _account(text: str) -> str | None:
    lowered = text.lower()
    for row in query_rows("SELECT name AS account_id, customer_name AS name FROM customers"):
        if row["account_id"].lower() in lowered or row["name"].lower() in lowered:
            return row["account_id"]
    return None


def _intent(text: str) -> str:
    lowered = text.lower()
    if re.search(r"how many|count|number of", lowered):
        return "count"
    if re.search(r"which|who|list|show", lowered):
        return "list"
    if re.search(r"why|reason", lowered):
        return "explain"
    if re.search(r"what should|recommend|next step|do next", lowered):
        return "recommend"
    return "lookup"


def remember(history: list[dict] | None) -> ConversationMemory:
    turns = history or []
    latest = _latest_user(turns)
    # Walk backwards until a turn gives us a subject/account. This supports
    # several conversational turns, not only the immediately previous one.
    subject = ""
    account_id = None
    for message in reversed(turns):
        if message.get("role") != "user":
            continue
        text = str(message.get("content", ""))
        subject = subject or _subject(text)
        account_id = account_id or _account(text)
        if subject and account_id:
            break
    return ConversationMemory(
        last_user_question=latest,
        subject=subject,
        account_id=account_id,
        intent=_intent(latest),
    )


def resolve_question(
    question: str,
    history: list[dict] | None = None,
    account_scope: str | None = None,
) -> tuple[str, ConversationMemory]:
    """Return a self-contained CRM query plus the memory used to resolve it."""
    memory = remember(history)
    current = question.strip()
    lowered = current.lower()
    account_id = account_scope or memory.account_id
    is_followup = (
        len(lowered.split()) <= 12
        and bool(re.search(r"\b(they|them|those|that|this|which|who|there)\b", lowered))
    )
    if not is_followup or not memory.subject:
        return current, memory

    subject = memory.subject
    if memory.intent == "count" and re.search(r"which|who|list|show|they|them", lowered):
        intent = "list"
    else:
        intent = _intent(current)
    scope = f" for account {account_id}" if account_id else " across the portfolio"
    resolved = f"{intent} {subject}{scope}. User follow-up: {current}"
    return resolved, memory
