"""Transparent deterministic reranking for the small CRM corpus.

This is intentionally dependency-free and CPU-only. It is a measured first
reranker, not a substitute for a cross-encoder. Every component is returned
with the candidate so a benchmark can explain why a document moved.
"""

from __future__ import annotations

import math
import re
from datetime import date

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "what", "how", "and", "or", "to", "for", "of", "in"}

WEIGHTS = {
    "lexical": 0.45,
    "entity": 0.20,
    "source_type": 0.15,
    "recency": 0.10,
    "scope": 0.10,
}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if token not in _STOPWORDS}


def _lexical_component(query: str, candidate: dict) -> float:
    query_terms = _tokens(query)
    doc_terms = _tokens(f"{candidate.get('title', '')} {candidate.get('text', '')}")
    if not query_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)


def _entity_component(query: str, candidate: dict, account_id: str | None) -> float:
    lowered = (query or "").lower()
    candidate_account = str(candidate.get("account_id") or "").lower()
    if account_id and candidate_account == account_id.lower():
        return 1.0
    title = str(candidate.get("title") or "").lower()
    if candidate_account and candidate_account in lowered:
        return 1.0 if candidate_account in title or candidate_account == candidate.get("account_id", "").lower() else 0.5
    return 0.0


def _source_component(query: str, candidate: dict) -> float:
    lowered = (query or "").lower()
    source = candidate.get("source_type")
    if source in {"email", "transcript"} and re.search(r"email|call|said|recent|latest|follow.?up|contract|request|blocking|why", lowered):
        return 1.0
    if source == "opportunity" and re.search(r"deal|opportunit|pipeline|stage|probability|close|blocking|win", lowered):
        return 1.0
    if source == "account" and re.search(r"account|portfolio|health|risk|which", lowered):
        return 1.0
    return 0.0


def _recency_component(candidate: dict, newest_date: date | None) -> float:
    if not newest_date or not candidate.get("date"):
        return 0.0
    try:
        value = date.fromisoformat(str(candidate["date"])[:10])
    except ValueError:
        return 0.0
    age_days = max(0, (newest_date - value).days)
    return math.exp(-age_days / 90.0)


def _scope_component(candidate: dict, account_id: str | None) -> float:
    # hybrid_search already applies the hard account filter. This component
    # remains explicit so a future union retriever can use the same contract.
    if account_id is None:
        return 0.0
    return float(candidate.get("account_id") == account_id)


def rerank(query: str, candidates: list[dict], account_id: str | None = None, limit: int = 6) -> list[dict]:
    """Rerank existing candidates without changing retrieval or DB behavior."""
    if not candidates:
        return []
    max_base = max(float(candidate.get("score", 0.0) or 0.0) for candidate in candidates) or 1.0
    dates = []
    for candidate in candidates:
        if candidate.get("date"):
            try:
                dates.append(date.fromisoformat(str(candidate["date"])[:10]))
            except ValueError:
                pass
    newest_date = max(dates) if dates else None

    ranked = []
    for original_rank, candidate in enumerate(candidates):
        components = {
            "lexical": 0.5 * (float(candidate.get("score", 0.0) or 0.0) / max_base) + 0.5 * _lexical_component(query, candidate),
            "entity": _entity_component(query, candidate, account_id),
            "source_type": _source_component(query, candidate),
            "recency": _recency_component(candidate, newest_date),
            "scope": _scope_component(candidate, account_id),
        }
        rerank_score = sum(WEIGHTS[name] * value for name, value in components.items())
        ranked.append((rerank_score, original_rank, {**candidate, "rerank_score": round(rerank_score, 6), "rerank_components": components}))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in ranked[:limit]]
