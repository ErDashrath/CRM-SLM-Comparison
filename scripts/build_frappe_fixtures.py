"""Generate Frappe/ERPNext-doctype-shaped mock CRM fixtures under mock_crm/frappe/.

This script is read-only with respect to the original mock_crm/ source files
(accounts/, opportunities/, emails/, transcripts/, product_catalog.json) --
it only ever writes under mock_crm/frappe/. It exists so the new,
doctype-shaped fixtures are always derived from (and therefore stay in sync
with) the original source data instead of being hand-duplicated.

The original mock_crm/ fixtures remain the input for the old NextBestAction
research pipeline (common/context.py et al.) and are never modified here.
The new mock_crm/frappe/ tree is the input for the conversational RAG layer
(common/crm_store.py and everything downstream of it).

Doctype field names/types below were taken from magnaerp.local's live
tabDocField metadata, trimmed to the fields relevant to a CRM sales
narrative. Project-specific concepts that are not native ERPNext fields
(account health, contact type/engagement, deal risk factors, negotiation
band) are emitted as custom_-prefixed fields/child tables, matching how a
real Frappe deployment would extend the framework for this exact use case.

Existing IDs (account_id, opportunity_id) and enum values (account_health,
contact_type, engagement_level) are carried over unchanged. Any field with
no honest source value is left null rather than invented.

Re-run this script any time the source mock_crm/ fixtures change:
    venv-gpu/bin/python scripts/build_frappe_fixtures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_CRM_DIR = PROJECT_ROOT / "mock_crm"
OUT_DIR = MOCK_CRM_DIR / "frappe"

_SUBJECT_RE = re.compile(r"\*\*Subject:\*\*\s*(.+)")
_TYPE_RE = re.compile(r"\*\*Type:\*\*\s*(.+)")
_FROM_RE = re.compile(r"\*\*From:\*\*\s*(.+)")
_TO_RE = re.compile(r"\*\*To:\*\*\s*(.+)")

_EMPLOYEE_BUCKETS = [
    (10, "1-10"), (50, "11-50"), (200, "51-200"), (500, "201-500"),
    (1000, "500-1000"), (float("inf"), "1000+"),
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"


def _employee_bucket(count: int) -> str:
    for ceiling, label in _EMPLOYEE_BUCKETS:
        if count <= ceiling:
            return label
    return "1000+"


def _load_accounts() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((MOCK_CRM_DIR / "accounts").glob("*.json"))]


def _load_opportunities() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((MOCK_CRM_DIR / "opportunities").glob("*.json"))]


def _load_catalog() -> list[dict[str, Any]]:
    return json.loads((MOCK_CRM_DIR / "product_catalog.json").read_text())["products"]


def _write(doctype_dir: str, doc_name: str, payload: dict[str, Any]) -> None:
    out = OUT_DIR / doctype_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{doc_name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _contact_by_type(account: dict[str, Any], contact_type: str) -> dict[str, Any] | None:
    return next((c for c in account["contacts"] if c["contact_type"] == contact_type), None)


def _contact_doc_name(account_id: str, contact_name: str) -> str:
    return f"{_slug(contact_name)}-{account_id}"


def build_customers(accounts: list[dict[str, Any]]) -> None:
    for account in accounts:
        ts = f"{account['relationship_start_date']} 09:00:00"
        payload = {
            "name": account["account_id"],
            "owner": "Administrator",
            "creation": ts,
            "modified": ts,
            "modified_by": "Administrator",
            "docstatus": 0,
            "customer_name": account["name"],
            "customer_type": "Company",
            "customer_group": account["segment"],
            "territory": account["region"],
            "industry": account["industry"],
            "market_segment": account["segment"],
            "default_currency": "INR",
            "account_manager": account["account_owner"],
            "customer_details": account["notes"],
            "custom_account_health": account["account_health"],
            "custom_relationship_start_date": account["relationship_start_date"],
            "custom_hq_city": account["hq_city"],
            "custom_employee_count": account["employees"],
            "custom_annual_revenue": account["annual_revenue_inr"],
            "custom_account_owner_role": account["account_owner_role"],
        }
        _write("customers", account["account_id"], payload)


def build_contacts(accounts: list[dict[str, Any]]) -> None:
    for account in accounts:
        for contact in account["contacts"]:
            doc_name = _contact_doc_name(account["account_id"], contact["name"])
            name_parts = contact["name"].split(" ", 1)
            first_name, last_name = name_parts[0], (name_parts[1] if len(name_parts) > 1 else "")
            email_ids = []
            if contact.get("email"):
                email_ids.append({"email_id": contact["email"], "is_primary": 1})
            ts = f"{account['relationship_start_date']} 09:00:00"
            payload = {
                "name": doc_name,
                "owner": "Administrator",
                "creation": ts,
                "modified": ts,
                "modified_by": "Administrator",
                "docstatus": 0,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": contact["name"],
                "designation": contact["role"],
                "company_name": account["name"],
                "email_ids": email_ids,
                "links": [{"link_doctype": "Customer", "link_name": account["account_id"]}],
                "custom_contact_type": contact["contact_type"],
                "custom_engagement_level": contact["engagement_level"],
                "custom_notes": contact["notes"],
            }
            _write("contacts", doc_name, payload)


def build_leads(accounts: list[dict[str, Any]]) -> None:
    for account in accounts:
        champion = _contact_by_type(account, "champion") or account["contacts"][0]
        ts = f"{account['relationship_start_date']} 09:00:00"
        payload = {
            "name": f"lead_{account['account_id']}_001",
            "owner": "Administrator",
            "creation": ts,
            "modified": ts,
            "modified_by": "Administrator",
            "docstatus": 0,
            "lead_name": champion["name"],
            "company_name": account["name"],
            "status": "Converted",
            "customer": account["account_id"],
            "industry": account["industry"],
            "territory": account["region"],
            "no_of_employees": _employee_bucket(account["employees"]),
            "annual_revenue": account["annual_revenue_inr"],
        }
        _write("leads", payload["name"], payload)


def build_opportunities(accounts: list[dict[str, Any]], opportunities: list[dict[str, Any]]) -> None:
    accounts_by_id = {a["account_id"]: a for a in accounts}
    for opp in opportunities:
        account = accounts_by_id[opp["account_id"]]
        champion = _contact_by_type(account, "champion") or account["contacts"][0]
        milestones = opp["milestones"]
        band = opp["negotiation_band"]
        items = [
            {
                "item_code": li["sku"],
                "description": li["description"],
                "qty": 1,
                "rate": li["amount_inr"],
                "amount": li["amount_inr"],
            }
            for li in opp["line_items"]
        ]
        risk_factors = [
            {"risk_type": rf["type"], "severity": rf["severity"], "description": rf["description"]}
            for rf in opp["risk_factors"]
        ]
        payload = {
            "name": opp["opportunity_id"],
            "owner": "Administrator",
            "creation": f"{opp['created_date']} 09:00:00",
            "modified": f"{opp['last_activity_date']} 09:00:00",
            "modified_by": "Administrator",
            "docstatus": 0,
            "party_name": account["account_id"],
            "customer": account["account_id"],
            "customer_name": account["name"],
            "title": opp["name"],
            "status": "Open",
            "opportunity_type": "Existing Customer",
            "opportunity_owner": account["account_owner"],
            "sales_stage": opp["stage"],
            "probability": opp["win_probability_pct"],
            "expected_closing": opp["expected_close_date"],
            "transaction_date": opp["created_date"],
            "opportunity_amount": opp["deal_value_inr"],
            "currency": opp["currency"],
            "contact_person": _contact_doc_name(account["account_id"], champion["name"]),
            "industry": account["industry"],
            "territory": account["region"],
            "items": items,
            "notes": [{"note": opp["notes"], "added_on": opp["created_date"]}],
            "custom_last_activity_date": opp["last_activity_date"],
            "custom_technical_evaluation_approved": int(milestones["technical_evaluation_approved"]),
            "custom_technical_evaluation_approved_date": milestones["technical_evaluation_approved_date"],
            "custom_solution_architecture_approved": int(milestones["solution_architecture_approved"]),
            "custom_solution_architecture_approved_date": milestones["solution_architecture_approved_date"],
            "custom_approved_min_discount_pct": band["approved_min_discount_pct"],
            "custom_approved_max_discount_pct": band["approved_max_discount_pct"],
            "custom_hard_ceiling_discount_pct": band["hard_ceiling_discount_pct"],
            "custom_requires_vp_sales_approval_above_pct": band["requires_vp_sales_approval_above_pct"],
            "custom_requires_cro_approval_above_pct": band["requires_cro_approval_above_pct"],
            "opportunity_risk_factors": risk_factors,
        }
        _write("opportunities", opp["opportunity_id"], payload)


def _extract_date_from_stem(stem: str) -> str | None:
    for part in stem.split("_"):
        if len(part) == 10 and part[4] == "-":
            return part
    return None


def build_communications(accounts: list[dict[str, Any]], opportunities: list[dict[str, Any]]) -> None:
    account_ids = [a["account_id"] for a in accounts]
    opp_id_by_account = {o["account_id"]: o["opportunity_id"] for o in opportunities}

    for folder, medium in (("emails", "Email"), ("transcripts", "Phone")):
        for path in sorted((MOCK_CRM_DIR / folder).glob("*.md")):
            stem = path.stem
            account_id = next((a for a in account_ids if stem.startswith(f"{a}_")), None)
            if account_id is None:
                continue
            content = path.read_text()
            communication_date = _extract_date_from_stem(stem)

            if medium == "Email":
                subject_match = _SUBJECT_RE.search(content)
                subject = subject_match.group(1).strip() if subject_match else stem.replace("_", " ")
                sender_match = _FROM_RE.search(content)
                recipients_match = _TO_RE.search(content)
            else:
                type_match = _TYPE_RE.search(content)
                subject = f"Call: {type_match.group(1).strip()}" if type_match else "Call recap"
                sender_match = None
                recipients_match = None

            ts = f"{communication_date} 09:00:00" if communication_date else None
            payload = {
                "name": stem,
                "owner": "Administrator",
                "creation": ts,
                "modified": ts,
                "modified_by": "Administrator",
                "docstatus": 0,
                "subject": subject,
                "communication_medium": medium,
                "communication_type": "Communication",
                "sender": sender_match.group(1).strip() if sender_match else None,
                "recipients": recipients_match.group(1).strip() if recipients_match else None,
                "content": content,
                "communication_date": ts,
                "reference_doctype": "Opportunity",
                "reference_name": opp_id_by_account.get(account_id),
            }
            _write("communications", stem, payload)


def build_items(products: list[dict[str, Any]]) -> None:
    for product in products:
        payload = {
            "name": product["sku"],
            "owner": None,
            "creation": None,
            "modified": None,
            "modified_by": None,
            "docstatus": 0,
            "item_code": product["sku"],
            "item_name": product["name"],
            "item_group": product["category"],
            "standard_rate": product["list_price"],
            "description": product["description"],
            "is_sales_item": 1,
            "disabled": 0,
            "custom_pricing_model": product["pricing_model"],
            "custom_unit": product["unit"],
        }
        _write("items", product["sku"], payload)


def main() -> None:
    accounts = _load_accounts()
    opportunities = _load_opportunities()
    products = _load_catalog()

    build_customers(accounts)
    build_contacts(accounts)
    build_leads(accounts)
    build_opportunities(accounts, opportunities)
    build_communications(accounts, opportunities)
    build_items(products)

    print(f"Wrote Frappe-shaped fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
