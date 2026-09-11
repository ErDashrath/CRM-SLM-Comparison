"""Deterministic compression for long, already-formatted CRM contexts."""

from __future__ import annotations

import json
import re


_SECTION_RE = re.compile(
    r"(?=^\[(?:account|opportunity|email|transcript|playbook|product_catalog) \|[^\n]+\]\n)",
    re.MULTILINE,
)
_POLICY_TERMS = re.compile(
    r"discount|economic buyer|competitor|procurement|negotiat|approval|ceiling|risk|sign[- ]off|champion",
    re.IGNORECASE,
)


def _sections(context_text: str) -> list[tuple[str, str]]:
    parts = [part.strip() for part in _SECTION_RE.split(context_text) if part.strip()]
    result = []
    for part in parts:
        header, _, body = part.partition("\n")
        result.append((header, body.strip()))
    return result


def _trim_policy(text: str, budget: int) -> str:
    lines = text.splitlines()
    selected = [line for line in lines if _POLICY_TERMS.search(line)]
    if not selected:
        selected = lines[:]
    result = "\n".join(selected)
    return result[:budget].rstrip()


def _compact_core_body(header: str, body: str) -> str:
    is_account = header.startswith("[account |")
    is_opportunity = header.startswith("[opportunity |")
    if not is_account and not is_opportunity:
        return body
    try:
        data = json.loads(body)
        if is_account:
            fields = (
                "account_id", "name", "industry", "segment", "region", "account_health",
                "account_owner", "contacts", "notes",
            )
            data = {key: data[key] for key in fields if key in data}
            data["contacts"] = [
                {
                    key: contact[key]
                    for key in ("name", "role", "contact_type", "engagement_level", "notes")
                    if key in contact
                }
                for contact in data.get("contacts", [])
            ]
        else:
            fields = (
                "opportunity_id", "account_id", "name", "stage", "deal_value_inr",
                "currency", "last_activity_date", "expected_close_date",
                "win_probability_pct", "milestones", "negotiation_band", "risk_factors", "notes",
            )
            data = {key: data[key] for key in fields if key in data}
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return body


def compact_formatted_context(
    context_text: str, max_chars: int = 7600, summaries: dict[str, str] | None = None
) -> str:
    """Keep decision-critical CRM evidence within an approximate token budget.

    Account and opportunity JSON are kept intact but minified (field
    selection, not summarization -- already dense/structured). Emails,
    transcripts, playbooks, and the product catalog use an LLM-generated
    summary when one is available in ``summaries`` (keyed by the exact
    "[doc_type | account_id | date]" header -- see
    data_gen/build_context_summaries.py, which builds this cache once via
    the teacher backend and reuses it across every query that references
    the same document). Falls back to blind character truncation
    (emails/transcripts) or keyword-line matching (playbooks/catalog) for
    any document NOT in ``summaries`` -- lossier, but keeps this function
    usable without a populated cache (e.g. before
    build_context_summaries.py has run, or for a document added after the
    cache was built). ``max_chars`` is intentionally expressed in
    characters because this helper is tokenizer-independent; 7600 chars is
    approximately 1500-2200 tokens for the Qwen tokenizer used here.
    """
    summaries = summaries or {}
    sections = _sections(context_text)
    if not sections or len(context_text) <= max_chars:
        return context_text

    core = [
        (header, body)
        for header, body in sections
        if header.startswith("[account |") or header.startswith("[opportunity |")
    ]
    evidence = [
        (header, body)
        for header, body in sections
        if header.startswith("[email |") or header.startswith("[transcript |")
    ]
    core_headers = {header for header, _ in core}
    evidence_headers = {header for header, _ in evidence}
    reference = [
        (header, body)
        for header, body in sections
        if header not in core_headers and header not in evidence_headers
    ]

    output: list[str] = []
    used = 0

    for header, body in core:
        block = f"{header}\n{_compact_core_body(header, body)}"
        output.append(block)
        used += len(block) + 2

    remaining = max(0, max_chars - used)
    evidence_budget = min(remaining * 3 // 4, 3200)
    per_evidence = max(240, evidence_budget // max(1, len(evidence)))
    for header, body in evidence:
        if header in summaries:
            block = f"{header}\n{summaries[header]}"
        else:
            block = f"{header}\n{body[:per_evidence].rstrip()}"
        output.append(block)
        used += len(block) + 2

    remaining = max(0, max_chars - used)
    per_reference = max(0, remaining // max(1, len(reference)))
    for header, body in reference:
        if header in summaries:
            compact_body = summaries[header]
        else:
            compact_body = _trim_policy(body, per_reference)
        if compact_body:
            output.append(f"{header}\n{compact_body}")

    return "\n\n".join(output)[:max_chars].rstrip()