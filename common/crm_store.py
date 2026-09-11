"""Read-only structured CRM store built from the local mock CRM fixtures."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_CRM_DIR = PROJECT_ROOT / "mock_crm"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "crm.sqlite3"


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def build_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """Rebuild the read-only query database from mock_crm/ source files."""
    with _connect(db_path) as db:
        db.executescript(
            """
            DROP TABLE IF EXISTS activities;
            DROP TABLE IF EXISTS contacts;
            DROP TABLE IF EXISTS opportunities;
            DROP TABLE IF EXISTS accounts;
            CREATE TABLE accounts (
                account_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                industry TEXT,
                segment TEXT,
                region TEXT,
                account_health TEXT,
                owner TEXT,
                notes TEXT
            );
            CREATE TABLE opportunities (
                opportunity_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                name TEXT NOT NULL,
                stage TEXT,
                deal_value_inr REAL,
                expected_close_date TEXT,
                win_probability_pct REAL,
                notes TEXT
            );
            CREATE TABLE contacts (
                contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                name TEXT NOT NULL,
                role TEXT,
                contact_type TEXT,
                engagement_level TEXT,
                notes TEXT
            );
            CREATE TABLE activities (
                activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                activity_type TEXT NOT NULL,
                activity_date TEXT,
                title TEXT,
                body TEXT
            );
            """
        )
        for account_file in sorted((MOCK_CRM_DIR / "accounts").glob("*.json")):
            account = json.loads(account_file.read_text())
            db.execute(
                "INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    account["account_id"], account["name"], account.get("industry"),
                    account.get("segment"), account.get("region"), account.get("account_health"),
                    account.get("account_owner"), account.get("notes"),
                ),
            )
            db.executemany(
                "INSERT INTO contacts(account_id, name, role, contact_type, engagement_level, notes) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        account["account_id"], contact["name"], contact.get("role"),
                        contact.get("contact_type"), contact.get("engagement_level"), contact.get("notes"),
                    )
                    for contact in account.get("contacts", [])
                ],
            )

        for opportunity_file in sorted((MOCK_CRM_DIR / "opportunities").glob("*.json")):
            opportunity = json.loads(opportunity_file.read_text())
            db.execute(
                "INSERT INTO opportunities VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    opportunity["opportunity_id"], opportunity["account_id"], opportunity["name"],
                    opportunity.get("stage"), opportunity.get("deal_value_inr"),
                    opportunity.get("expected_close_date"), opportunity.get("win_probability_pct"),
                    opportunity.get("notes"),
                ),
            )

        account_ids = {row["account_id"] for row in db.execute("SELECT account_id FROM accounts")}
        for folder, activity_type in (("emails", "email"), ("transcripts", "transcript")):
            for activity_file in sorted((MOCK_CRM_DIR / folder).glob("*.md")):
                parts = activity_file.stem.split("_")
                account_id = next((candidate for candidate in account_ids if activity_file.stem.startswith(f"{candidate}_")), None)
                if account_id is None:
                    continue
                date_match = next((part for part in parts if len(part) == 10 and part[4] == "-"), None)
                db.execute(
                    "INSERT INTO activities(account_id, activity_type, activity_date, title, body) VALUES (?, ?, ?, ?, ?)",
                    (account_id, activity_type, date_match, activity_file.stem, activity_file.read_text()),
                )
        db.commit()
    return db_path


def ensure_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    source_files = list((MOCK_CRM_DIR / "accounts").glob("*.json"))
    source_files += list((MOCK_CRM_DIR / "opportunities").glob("*.json"))
    source_files += list((MOCK_CRM_DIR / "emails").glob("*.md"))
    source_files += list((MOCK_CRM_DIR / "transcripts").glob("*.md"))
    newest_source = max((path.stat().st_mtime for path in source_files), default=0)
    if not db_path.exists() or db_path.stat().st_mtime < newest_source:
        return build_database(db_path)
    return db_path


def query_rows(sql: str, params: tuple = (), db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    ensure_database(db_path)
    with _connect(db_path) as db:
        return [dict(row) for row in db.execute(sql, params).fetchall()]


def database_snapshot(db_path: Path = DEFAULT_DB_PATH) -> dict:
    return {
        "accounts": query_rows("SELECT COUNT(*) AS count FROM accounts", db_path=db_path)[0]["count"],
        "opportunities": query_rows("SELECT COUNT(*) AS count FROM opportunities", db_path=db_path)[0]["count"],
        "contacts": query_rows("SELECT COUNT(*) AS count FROM contacts", db_path=db_path)[0]["count"],
        "activities": query_rows("SELECT COUNT(*) AS count FROM activities", db_path=db_path)[0]["count"],
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }