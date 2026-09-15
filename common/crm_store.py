"""Read-only structured CRM store built from the Frappe-shaped mock CRM fixtures.

Sources from mock_crm/frappe/ (see scripts/build_frappe_fixtures.py), which
mirrors real Frappe/ERPNext doctype shapes: Customer, Contact, Lead,
Opportunity, Communication, Item. Every table carries Frappe's standard
doctype metadata columns (name as primary key, owner, creation, modified,
modified_by, docstatus).

This module is used only by the conversational RAG chat layer
(crm_tools.py, crm_retrieval.py, conversation_memory.py, ui/app.py). The
old NextBestAction research pipeline (common/context.py et al.) reads the
original mock_crm/accounts + mock_crm/opportunities fixtures directly and
does not depend on this module.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_CRM_DIR = PROJECT_ROOT / "mock_crm" / "frappe"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "crm.sqlite3"

_METADATA_COLUMNS = "name TEXT PRIMARY KEY, owner TEXT, creation TEXT, modified TEXT, modified_by TEXT, docstatus INTEGER"


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _connect_readonly(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the query database in SQLite read-only mode."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 2000")
    return connection


def _load(doctype_dir: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((MOCK_CRM_DIR / doctype_dir).glob("*.json"))]


def build_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """Rebuild the read-only query database from mock_crm/frappe/ source files."""
    with _connect(db_path) as db:
        db.executescript(
            f"""
            DROP TABLE IF EXISTS opportunity_risk_factors;
            DROP TABLE IF EXISTS opportunity_items;
            DROP TABLE IF EXISTS opportunities;
            DROP TABLE IF EXISTS communications;
            DROP TABLE IF EXISTS contact_email_ids;
            DROP TABLE IF EXISTS contact_links;
            DROP TABLE IF EXISTS contacts;
            DROP TABLE IF EXISTS leads;
            DROP TABLE IF EXISTS customers;
            DROP TABLE IF EXISTS items;

            CREATE TABLE customers (
                {_METADATA_COLUMNS},
                customer_name TEXT NOT NULL,
                customer_type TEXT,
                customer_group TEXT,
                territory TEXT,
                industry TEXT,
                market_segment TEXT,
                default_currency TEXT,
                account_manager TEXT,
                customer_details TEXT,
                custom_account_health TEXT,
                custom_relationship_start_date TEXT,
                custom_hq_city TEXT,
                custom_employee_count INTEGER,
                custom_annual_revenue REAL,
                custom_account_owner_role TEXT
            );
            CREATE TABLE leads (
                {_METADATA_COLUMNS},
                lead_name TEXT,
                company_name TEXT,
                status TEXT,
                customer TEXT NOT NULL REFERENCES customers(name),
                industry TEXT,
                territory TEXT,
                no_of_employees TEXT,
                annual_revenue REAL
            );
            CREATE TABLE contacts (
                {_METADATA_COLUMNS},
                first_name TEXT,
                last_name TEXT,
                full_name TEXT NOT NULL,
                designation TEXT,
                company_name TEXT,
                custom_contact_type TEXT,
                custom_engagement_level TEXT,
                custom_notes TEXT
            );
            CREATE TABLE contact_email_ids (
                contact TEXT NOT NULL REFERENCES contacts(name),
                email_id TEXT,
                is_primary INTEGER
            );
            CREATE TABLE contact_links (
                contact TEXT NOT NULL REFERENCES contacts(name),
                link_doctype TEXT,
                link_name TEXT
            );
            CREATE TABLE opportunities (
                {_METADATA_COLUMNS},
                customer TEXT NOT NULL REFERENCES customers(name),
                customer_name TEXT,
                title TEXT,
                status TEXT,
                opportunity_type TEXT,
                opportunity_owner TEXT,
                sales_stage TEXT,
                probability REAL,
                expected_closing TEXT,
                transaction_date TEXT,
                opportunity_amount REAL,
                currency TEXT,
                contact_person TEXT REFERENCES contacts(name),
                industry TEXT,
                territory TEXT,
                custom_last_activity_date TEXT,
                custom_technical_evaluation_approved INTEGER,
                custom_technical_evaluation_approved_date TEXT,
                custom_solution_architecture_approved INTEGER,
                custom_solution_architecture_approved_date TEXT,
                custom_approved_min_discount_pct REAL,
                custom_approved_max_discount_pct REAL,
                custom_hard_ceiling_discount_pct REAL,
                custom_requires_vp_sales_approval_above_pct REAL,
                custom_requires_cro_approval_above_pct REAL,
                notes TEXT
            );
            CREATE TABLE opportunity_items (
                opportunity TEXT NOT NULL REFERENCES opportunities(name),
                item_code TEXT,
                description TEXT,
                qty REAL,
                rate REAL,
                amount REAL
            );
            CREATE TABLE opportunity_risk_factors (
                opportunity TEXT NOT NULL REFERENCES opportunities(name),
                risk_type TEXT,
                severity TEXT,
                description TEXT
            );
            CREATE TABLE communications (
                {_METADATA_COLUMNS},
                subject TEXT,
                communication_medium TEXT,
                communication_type TEXT,
                sender TEXT,
                recipients TEXT,
                content TEXT,
                communication_date TEXT,
                reference_doctype TEXT,
                reference_name TEXT REFERENCES opportunities(name)
            );
            CREATE TABLE items (
                {_METADATA_COLUMNS},
                item_code TEXT NOT NULL,
                item_name TEXT,
                item_group TEXT,
                standard_rate REAL,
                description TEXT,
                is_sales_item INTEGER,
                disabled INTEGER,
                custom_pricing_model TEXT,
                custom_unit TEXT
            );
            """
        )

        for customer in _load("customers"):
            db.execute(
                """INSERT INTO customers
                (name, owner, creation, modified, modified_by, docstatus, customer_name, customer_type,
                 customer_group, territory, industry, market_segment, default_currency, account_manager,
                 customer_details, custom_account_health, custom_relationship_start_date, custom_hq_city,
                 custom_employee_count, custom_annual_revenue, custom_account_owner_role)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    customer["name"], customer["owner"], customer["creation"], customer["modified"],
                    customer["modified_by"], customer["docstatus"], customer["customer_name"],
                    customer["customer_type"], customer["customer_group"], customer["territory"],
                    customer["industry"], customer["market_segment"], customer["default_currency"],
                    customer["account_manager"], customer["customer_details"], customer["custom_account_health"],
                    customer["custom_relationship_start_date"], customer["custom_hq_city"],
                    customer["custom_employee_count"], customer["custom_annual_revenue"],
                    customer["custom_account_owner_role"],
                ),
            )

        for lead in _load("leads"):
            db.execute(
                """INSERT INTO leads
                (name, owner, creation, modified, modified_by, docstatus, lead_name, company_name,
                 status, customer, industry, territory, no_of_employees, annual_revenue)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    lead["name"], lead["owner"], lead["creation"], lead["modified"], lead["modified_by"],
                    lead["docstatus"], lead["lead_name"], lead["company_name"], lead["status"],
                    lead["customer"], lead["industry"], lead["territory"], lead["no_of_employees"],
                    lead["annual_revenue"],
                ),
            )

        for contact in _load("contacts"):
            db.execute(
                """INSERT INTO contacts
                (name, owner, creation, modified, modified_by, docstatus, first_name, last_name, full_name,
                 designation, company_name, custom_contact_type, custom_engagement_level, custom_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    contact["name"], contact["owner"], contact["creation"], contact["modified"],
                    contact["modified_by"], contact["docstatus"], contact["first_name"], contact["last_name"],
                    contact["full_name"], contact["designation"], contact["company_name"],
                    contact["custom_contact_type"], contact["custom_engagement_level"], contact["custom_notes"],
                ),
            )
            db.executemany(
                "INSERT INTO contact_email_ids(contact, email_id, is_primary) VALUES (?, ?, ?)",
                [(contact["name"], e["email_id"], e["is_primary"]) for e in contact["email_ids"]],
            )
            db.executemany(
                "INSERT INTO contact_links(contact, link_doctype, link_name) VALUES (?, ?, ?)",
                [(contact["name"], l["link_doctype"], l["link_name"]) for l in contact["links"]],
            )

        for opp in _load("opportunities"):
            db.execute(
                """INSERT INTO opportunities
                (name, owner, creation, modified, modified_by, docstatus, customer, customer_name, title, status,
                 opportunity_type, opportunity_owner, sales_stage, probability, expected_closing,
                 transaction_date, opportunity_amount, currency, contact_person, industry, territory,
                 custom_last_activity_date, custom_technical_evaluation_approved,
                 custom_technical_evaluation_approved_date, custom_solution_architecture_approved,
                 custom_solution_architecture_approved_date, custom_approved_min_discount_pct,
                 custom_approved_max_discount_pct, custom_hard_ceiling_discount_pct,
                 custom_requires_vp_sales_approval_above_pct, custom_requires_cro_approval_above_pct, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    opp["name"], opp["owner"], opp["creation"], opp["modified"], opp["modified_by"],
                    opp["docstatus"], opp["customer"], opp["customer_name"], opp["title"], opp["status"],
                    opp["opportunity_type"], opp["opportunity_owner"], opp["sales_stage"], opp["probability"],
                    opp["expected_closing"], opp["transaction_date"], opp["opportunity_amount"],
                    opp["currency"], opp["contact_person"], opp["industry"], opp["territory"],
                    opp["custom_last_activity_date"], opp["custom_technical_evaluation_approved"],
                    opp["custom_technical_evaluation_approved_date"], opp["custom_solution_architecture_approved"],
                    opp["custom_solution_architecture_approved_date"], opp["custom_approved_min_discount_pct"],
                    opp["custom_approved_max_discount_pct"], opp["custom_hard_ceiling_discount_pct"],
                    opp["custom_requires_vp_sales_approval_above_pct"],
                    opp["custom_requires_cro_approval_above_pct"],
                    "\n".join(n["note"] for n in opp["notes"]),
                ),
            )
            db.executemany(
                "INSERT INTO opportunity_items(opportunity, item_code, description, qty, rate, amount) VALUES (?, ?, ?, ?, ?, ?)",
                [(opp["name"], i["item_code"], i["description"], i["qty"], i["rate"], i["amount"]) for i in opp["items"]],
            )
            db.executemany(
                "INSERT INTO opportunity_risk_factors(opportunity, risk_type, severity, description) VALUES (?, ?, ?, ?)",
                [(opp["name"], r["risk_type"], r["severity"], r["description"]) for r in opp["opportunity_risk_factors"]],
            )

        for comm in _load("communications"):
            db.execute(
                """INSERT INTO communications
                (name, owner, creation, modified, modified_by, docstatus, subject, communication_medium,
                 communication_type, sender, recipients, content, communication_date, reference_doctype, reference_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    comm["name"], comm["owner"], comm["creation"], comm["modified"], comm["modified_by"],
                    comm["docstatus"], comm["subject"], comm["communication_medium"], comm["communication_type"],
                    comm["sender"], comm["recipients"], comm["content"], comm["communication_date"],
                    comm["reference_doctype"], comm["reference_name"],
                ),
            )

        for item in _load("items"):
            db.execute(
                """INSERT INTO items
                (name, owner, creation, modified, modified_by, docstatus, item_code, item_name, item_group,
                 standard_rate, description, is_sales_item, disabled, custom_pricing_model, custom_unit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item["name"], item["owner"], item["creation"], item["modified"], item["modified_by"],
                    item["docstatus"], item["item_code"], item["item_name"], item["item_group"],
                    item["standard_rate"], item["description"], item["is_sales_item"], item["disabled"],
                    item["custom_pricing_model"], item["custom_unit"],
                ),
            )

        db.commit()
    return db_path


def ensure_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    source_files = []
    for doctype_dir in ("customers", "leads", "contacts", "opportunities", "communications", "items"):
        source_files += list((MOCK_CRM_DIR / doctype_dir).glob("*.json"))
    newest_source = max((path.stat().st_mtime for path in source_files), default=0)
    if not db_path.exists() or db_path.stat().st_mtime < newest_source:
        return build_database(db_path)
    return db_path


def query_rows(sql: str, params: tuple = (), db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    ensure_database(db_path)
    with _connect_readonly(db_path) as db:
        return [dict(row) for row in db.execute(sql, params).fetchall()]


def database_snapshot(db_path: Path = DEFAULT_DB_PATH) -> dict:
    return {
        "accounts": query_rows("SELECT COUNT(*) AS count FROM customers", db_path=db_path)[0]["count"],
        "opportunities": query_rows("SELECT COUNT(*) AS count FROM opportunities", db_path=db_path)[0]["count"],
        "contacts": query_rows("SELECT COUNT(*) AS count FROM contacts", db_path=db_path)[0]["count"],
        "leads": query_rows("SELECT COUNT(*) AS count FROM leads", db_path=db_path)[0]["count"],
        "activities": query_rows("SELECT COUNT(*) AS count FROM communications", db_path=db_path)[0]["count"],
        "items": query_rows("SELECT COUNT(*) AS count FROM items", db_path=db_path)[0]["count"],
        "opportunity_items": query_rows("SELECT COUNT(*) AS count FROM opportunity_items", db_path=db_path)[0]["count"],
        "risk_factors": query_rows("SELECT COUNT(*) AS count FROM opportunity_risk_factors", db_path=db_path)[0]["count"],
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }
