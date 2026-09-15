"""Dependency-light hybrid CRM retrieval.

The retriever combines lexical BM25 scoring with CRM metadata boosts. It is
deliberately local and deterministic for this POC; a vector index can be
added later without changing the assistant contract.
"""

from __future__ import annotations

import math
import re

from common.crm_store import query_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "what", "how", "and", "or", "to", "for", "of", "in"}


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _documents() -> list[dict]:
    documents = []
    for row in query_rows(
        "SELECT name AS account_id, customer_name AS name, custom_account_health AS account_health, "
        "customer_details AS notes FROM customers"
    ):
        documents.append({"document_id": f"account:{row['account_id']}", "source_type": "account", "account_id": row["account_id"], "title": row["name"], "text": " ".join(str(v or "") for v in row.values())})
    for row in query_rows(
        "SELECT o.name AS opportunity_id, o.customer AS account_id, o.title AS name, o.sales_stage AS stage, "
        "o.opportunity_amount AS deal_value_inr, o.expected_closing AS expected_close_date, "
        "o.probability AS win_probability_pct, o.notes FROM opportunities o"
    ):
        documents.append({"document_id": f"opportunity:{row['opportunity_id']}", "source_type": "opportunity", "account_id": row["account_id"], "title": row["name"], "text": " ".join(str(v or "") for v in row.values())})
    for row in query_rows(
        "SELECT cm.name AS communication_id, o.customer AS account_id, cm.communication_medium AS medium, "
        "cm.communication_date AS activity_date, "
        "cm.subject AS title, cm.content AS body FROM communications cm "
        "JOIN opportunities o ON o.name = cm.reference_name"
    ):
        activity_type = "email" if row["medium"] == "Email" else "transcript"
        documents.append({"document_id": f"communication:{row['communication_id']}", "source_type": activity_type, "account_id": row["account_id"], "title": row["title"], "date": row["activity_date"], "text": " ".join(str(v or "") for v in (row["title"], row["body"]))})
    return documents


def hybrid_search(query: str, limit: int = 6, account_id: str | None = None) -> list[dict]:
    documents = _documents()
    query_terms = _tokens(query)
    if account_id:
        documents = [doc for doc in documents if doc.get("account_id") == account_id]
    if not documents or not query_terms:
        return []

    tokenized = [_tokens(doc["text"]) for doc in documents]
    document_frequency = {term: sum(term in tokens for tokens in tokenized) for term in set(query_terms)}
    average_length = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    ranked = []
    for document, tokens in zip(documents, tokenized):
        term_counts = {term: tokens.count(term) for term in set(query_terms)}
        score = 0.0
        for term, count in term_counts.items():
            if not count:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = count + 1.5 * (0.25 + 0.75 * len(tokens) / max(1, average_length))
            score += idf * count * 2.5 / denominator
        title_terms = set(_tokens(document.get("title", "")))
        score += 2.0 * len(title_terms.intersection(query_terms))
        if document.get("source_type") == "opportunity" and any(term in query_terms for term in ("deal", "pipeline", "stage", "close", "probability")):
            score += 0.8
        if document.get("source_type") in {"email", "transcript"} and any(term in query_terms for term in ("risk", "why", "recent", "said", "customer", "competitor")):
            score += 0.5
        if document.get("date"):
            score += 0.1
        if score > 0:
            ranked.append((score, document))
    ranked.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, document in ranked[:limit]:
        results.append({**document, "score": round(score, 4)})
    return results
