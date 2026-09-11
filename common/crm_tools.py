"""Safe, read-only CRM tools exposed to the conversational assistant."""

from __future__ import annotations

import re

from common.crm_store import query_rows


def _account_id_from_text(text: str) -> str | None:
    lowered = text.lower()
    for row in query_rows("SELECT account_id, name FROM accounts"):
        account_id, name = row["account_id"], row["name"]
        if account_id.lower() in lowered or name.lower() in lowered:
            return account_id
    return None


def run_crm_tool(question: str) -> dict:
    """Route a natural-language read request to a bounded SQL tool."""
    lowered = question.lower()
    account_id = _account_id_from_text(question)

    if re.search(r"\blead(s)?\b", lowered):
        return {
            "tool": "schema_clarification",
            "status": "needs_clarification",
            "answer": "This CRM dataset has no leads table or lead records. Do you mean opportunities, contacts, or accounts?",
            "available_entities": ["accounts", "opportunities", "contacts", "emails", "transcripts"],
        }
    if "risk" in lowered and account_id is None:
        rows = query_rows(
            "SELECT a.name, a.account_health, o.name AS opportunity, o.stage, o.win_probability_pct "
            "FROM accounts a LEFT JOIN opportunities o ON o.account_id = a.account_id "
            "WHERE a.account_health = 'at_risk' OR o.win_probability_pct < 60"
        )
        return {"tool": "risk_summary", "status": "ok", "answer": "These accounts or opportunities need attention.", "rows": rows}
    if re.search(r"how many|count|number of", lowered) and "account" in lowered:
        row = query_rows("SELECT COUNT(*) AS count FROM accounts")[0]
        return {"tool": "count_accounts", "status": "ok", "answer": f"There are {row['count']} accounts in the CRM dataset.", "rows": [row]}
    if re.search(r"how many|count|number of", lowered) and ("opportunit" in lowered or "deal" in lowered):
        row = query_rows("SELECT COUNT(*) AS count FROM opportunities")[0]
        return {"tool": "count_opportunities", "status": "ok", "answer": f"There are {row['count']} opportunities in the CRM dataset.", "rows": [row]}
    if "pipeline" in lowered or "stage" in lowered:
        rows = query_rows(
            "SELECT stage, COUNT(*) AS opportunities, COALESCE(SUM(deal_value_inr), 0) AS value_inr "
            "FROM opportunities GROUP BY stage ORDER BY value_inr DESC"
        )
        return {"tool": "pipeline_summary", "status": "ok", "answer": "Here is the pipeline grouped by stage.", "rows": rows}
    if account_id:
        rows = query_rows(
            "SELECT a.name, a.account_health, o.name AS opportunity, o.stage, o.deal_value_inr, "
            "o.expected_close_date, o.win_probability_pct "
            "FROM accounts a LEFT JOIN opportunities o ON o.account_id = a.account_id WHERE a.account_id = ?",
            (account_id,),
        )
        return {"tool": "account_summary", "status": "ok", "answer": f"Here is the current CRM summary for {rows[0]['name']}.", "rows": rows}
    return {
        "tool": "clarification",
        "status": "needs_clarification",
        "answer": "I can search accounts, opportunities, contacts, emails, and transcripts. Which account, entity, or metric should I use?",
        "available_entities": ["accounts", "opportunities", "contacts", "emails", "transcripts"],
    }