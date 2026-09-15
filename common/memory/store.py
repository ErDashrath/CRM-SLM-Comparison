"""SQLite-backed conversational session store.

CRM source data stays in data/crm.sqlite3. Conversation state is deliberately
stored in a separate database so chat retention and CRM refreshes are
independent operations.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from common.memory.budget import estimate_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_DB = PROJECT_ROOT / "data" / "sessions.sqlite3"


class SessionStore:
    def __init__(self, db_path: Path = DEFAULT_SESSION_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    account_scope TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    summary TEXT NOT NULL,
                    covered_until TEXT,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_conversation(self, conversation: dict[str, Any], account_scope: str | None = None) -> None:
        now = datetime.now().isoformat()
        session_id = conversation["id"]
        messages = conversation.get("messages", [])
        with self._connect() as db:
            db.execute(
                """INSERT INTO sessions(session_id, title, account_scope, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET title=excluded.title,
                   account_scope=excluded.account_scope, updated_at=excluded.updated_at""",
                (session_id, conversation.get("title", "New chat"), account_scope, conversation.get("created_at", now), now),
            )
            db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for message in messages:
                payload = {key: value for key, value in message.items() if key not in {"role", "content", "ts"}}
                db.execute(
                    "INSERT INTO messages(session_id, role, content, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, message.get("role", ""), message.get("content", ""), json.dumps(payload, ensure_ascii=False), message.get("ts", now)),
                )
                db.execute(
                    "UPDATE messages SET token_count = ? WHERE message_id = last_insert_rowid()",
                    (estimate_tokens(str(message.get("content", ""))),),
                )

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            sessions = db.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
            return [self._load_row(db, row) for row in sessions]

    def load_conversation(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            return self._load_row(db, row) if row else None

    def _load_row(self, db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        messages = []
        for message in db.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY message_id", (row["session_id"],)):
            payload = json.loads(message["payload_json"] or "{}")
            messages.append({"role": message["role"], "content": message["content"], **payload, "ts": message["created_at"]})
        return {
            "id": row["session_id"],
            "title": row["title"],
            "messages": messages,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "account_scope": row["account_scope"],
        }

    def save_summary(self, session_id: str, summary: str, covered_until: str | None, token_count: int = 0) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO memory_summaries(session_id, summary, covered_until, token_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, summary, covered_until, token_count, datetime.now().isoformat()),
            )

    def usage(self, session_id: str) -> dict[str, int]:
        with self._connect() as db:
            row = db.execute(
                "SELECT COALESCE(SUM(token_count), 0) AS message_tokens FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return {"message_tokens": int(row["message_tokens"])}

    def limit_state(self, session_id: str, max_tokens: int = 100_000) -> dict[str, int | bool | str]:
        used = self.usage(session_id)["message_tokens"]
        warning = used >= int(max_tokens * 0.8)
        reached = used >= max_tokens
        return {
            "session_tokens": used,
            "session_token_limit": max_tokens,
            "session_limit_warning": warning,
            "session_limit_reached": reached,
            "status": "session_limit_reached" if reached else ("session_limit_warning" if warning else "ok"),
        }
