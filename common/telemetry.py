"""Local, privacy-aware telemetry for conversational CRM turns.

The local JSONL sink is the source of truth for the POC.  It is enabled by
default, writes under ``data/telemetry/``, and records stable IDs, counts,
latencies, and status fields.  Prompts and CRM text are omitted unless
``CRM_TELEMETRY_LOG_PROMPTS=true`` is explicitly set.

This module has no LangSmith dependency.  A remote exporter can consume the
same event shape later without coupling the agent loop to a vendor SDK.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "data" / "telemetry" / "events.jsonl"

_LOCK = threading.Lock()
_TEXT_KEYS = {
    "answer", "body", "content", "prompt", "question", "query", "raw",
    "response", "text", "transcript", "email", "email_body",
}


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def text_fingerprint(value: str) -> str:
    """Return a stable, non-reversible identifier for a text value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _redact(value: Any, *, allow_text: bool) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if not allow_text and lowered in _TEXT_KEYS and isinstance(item, str):
                result[f"{key}_hash"] = text_fingerprint(item)
                result[f"{key}_chars"] = len(item)
            else:
                result[str(key)] = _redact(item, allow_text=allow_text)
        return result
    if isinstance(value, list):
        return [_redact(item, allow_text=allow_text) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, allow_text=allow_text) for item in value]
    return value


class TelemetrySink:
    """Append structured events to a local JSONL audit stream."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        enabled: bool | None = None,
        redact: bool | None = None,
        log_prompts: bool | None = None,
    ) -> None:
        self.path = Path(path or os.getenv("CRM_TELEMETRY_PATH", DEFAULT_TELEMETRY_PATH))
        self.enabled = _truthy(os.getenv("CRM_TELEMETRY_ENABLED"), True) if enabled is None else enabled
        self.redact = _truthy(os.getenv("CRM_TELEMETRY_REDACT"), True) if redact is None else redact
        self.log_prompts = _truthy(os.getenv("CRM_TELEMETRY_LOG_PROMPTS"), False) if log_prompts is None else log_prompts

    def emit(self, event: str, **fields: Any) -> str | None:
        """Write one event and return its event ID, or None when disabled."""
        if not self.enabled:
            return None
        event_id = str(uuid.uuid4())
        record = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": _redact(fields, allow_text=self.log_prompts or not self.redact),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        return event_id


_DEFAULT_TELEMETRY_SINK: TelemetrySink | None = None


def default_sink() -> TelemetrySink:
    global _DEFAULT_TELEMETRY_SINK
    if _DEFAULT_TELEMETRY_SINK is None:
        _DEFAULT_TELEMETRY_SINK = TelemetrySink()
    return _DEFAULT_TELEMETRY_SINK


__all__ = ["TelemetrySink", "default_sink", "text_fingerprint"]
