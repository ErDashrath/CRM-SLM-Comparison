"""Compatibility API for the schema-driven CRM query service.

Existing callers may still import ``run_crm_tool``. Query semantics now live
in ``common.crm_query``; this module intentionally contains no SQL router.
"""

from __future__ import annotations

from typing import Any

from common.crm_query import execute_question


def run_crm_tool(
    question: str,
    account_scope: str | None = None,
    *,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return execute_question(question, account_scope=account_scope, plan=plan)
