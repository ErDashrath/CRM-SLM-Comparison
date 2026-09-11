"""
Deterministic per-account ground-truth summary, extracted directly from
mock_crm/ fields (account_health, contacts, opportunity risk_factors, deal
terms) -- no interpretation, no LLM call, just the facts already sitting in
the JSON.

Exists to solve a real problem with data_gen/curate_sft_subset.py: a human
reviewer can't be expected to have memorized 4 fictional companies' worth of
emails and transcripts before judging whether a drafted response is
correctly grounded. Printing this alongside each candidate lets a reviewer
check "did the draft get the actual facts right" in a few seconds instead of
reading the full account history each time.

The mock CRM's opportunity JSON already carries a ready-made ground-truth
risk list at `risk_factors: [{type, severity, description}, ...]` -- that's
the single most useful field here, since it's literally "what SHOULD this
response mention," pre-labeled by whoever built the mock data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_CRM_DIR = PROJECT_ROOT / "mock_crm"


def _find_opportunity(account_id: str) -> Optional[dict]:
    for f in (MOCK_CRM_DIR / "opportunities").glob("*.json"):
        data = json.loads(f.read_text())
        if data.get("account_id") == account_id:
            return data
    return None


def account_cheatsheet(account_id: str) -> str:
    """A compact, fact-only summary of one account -- ground truth to check
    a drafted response against, not a narrative."""
    account_path = MOCK_CRM_DIR / "accounts" / f"{account_id}.json"
    if not account_path.exists():
        return f"(no account record for {account_id!r})"

    account = json.loads(account_path.read_text())
    opp = _find_opportunity(account_id)

    lines = [
        f"{account['name']} ({account['industry']}, {account['segment']}) "
        f"-- health: {account['account_health']}",
    ]

    key_contacts = {"champion": [], "economic_buyer": [], "procurement": []}
    for c in account.get("contacts", []):
        ctype = c.get("contact_type")
        if ctype in key_contacts:
            key_contacts[ctype].append(c)
    for ctype, label in [
        ("economic_buyer", "Economic buyer"),
        ("champion", "Champion"),
        ("procurement", "Procurement"),
    ]:
        for c in key_contacts[ctype]:
            lines.append(
                f"  {label}: {c['name']} ({c['role']}) -- engagement: {c['engagement_level']}"
            )

    if opp:
        lines.append(
            f"Opportunity: {opp['stage']}, INR {opp['deal_value_inr']:,}, "
            f"win_probability {opp['win_probability_pct']}%, "
            f"expected close {opp['expected_close_date']}"
        )
        band = opp.get("negotiation_band")
        if band:
            lines.append(
                f"  Discount policy: {band['approved_min_discount_pct']}-"
                f"{band['approved_max_discount_pct']}% approved band, "
                f"{band['hard_ceiling_discount_pct']}% hard ceiling"
            )
        risk_factors = opp.get("risk_factors", [])
        if risk_factors:
            lines.append("  GROUND-TRUTH RISK FACTORS (the draft should surface these if relevant to the query):")
            for rf in risk_factors:
                lines.append(f"    - [{rf['severity']}] {rf['type']}: {rf['description']}")
        else:
            lines.append("  GROUND-TRUTH RISK FACTORS: none -- this is a healthy, low-risk account.")

    return "\n".join(lines)


def portfolio_cheatsheet() -> str:
    """All 4 accounts' cheat sheets concatenated, for portfolio-wide queries."""
    account_ids = sorted(p.stem for p in (MOCK_CRM_DIR / "accounts").glob("*.json"))
    return "\n\n".join(account_cheatsheet(a) for a in account_ids)


if __name__ == "__main__":
    print(account_cheatsheet("acme_corp"))
    print()
    print(account_cheatsheet("globex"))
