"""
Direct, RAG-free context assembly over mock_crm/.

SalesIntelligence's rag/retriever.py does top-k semantic retrieval over a
Chroma index; this project deliberately skips that layer (see the plan --
this POC is about how a model was specialized, not about retrieval) and
instead returns an account's entire mock_crm footprint plus global
reference material every time. At this dataset's scale (a handful of files
per account) that's a strict superset of what top-k retrieval would surface
anyway, so nothing is lost by skipping the vector store, and it removes a
whole dependency (chromadb/sentence-transformers) this project doesn't need.

Produces the same hit shape SalesIntelligence's retrieve() does --
{doc_type, account_id, date, text} -- so common/formatting.py's
_format_context/_format_evidence_context (ported from
orchestrator/pipeline.py) work unchanged against either source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_CRM_DIR = PROJECT_ROOT / "mock_crm"

GLOBAL_ACCOUNT_ID = "global"  # matches SalesIntelligence's rag/retriever.py constant

_DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _date_from_filename(path: Path) -> Optional[str]:
    match = _DATE_IN_FILENAME_RE.search(path.stem)
    return match.group(1) if match else None


def _account_hits(account_id: str) -> list[dict]:
    hits: list[dict] = []

    account_file = MOCK_CRM_DIR / "accounts" / f"{account_id}.json"
    if account_file.exists():
        hits.append(
            {
                "doc_type": "account",
                "account_id": account_id,
                "date": None,
                "text": account_file.read_text(),
            }
        )

    # Opportunity filenames don't reliably prefix-match account_id (e.g.
    # "acme_opp.json" for account_id="acme_corp") -- match on the JSON's own
    # account_id field instead, which is authoritative.
    for opp_file in (MOCK_CRM_DIR / "opportunities").glob("*.json"):
        try:
            data = json.loads(opp_file.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("account_id") == account_id:
            hits.append(
                {
                    "doc_type": "opportunity",
                    "account_id": account_id,
                    "date": None,
                    "text": opp_file.read_text(),
                }
            )

    for folder, doc_type in [("emails", "email"), ("transcripts", "transcript")]:
        for f in sorted((MOCK_CRM_DIR / folder).glob(f"{account_id}_*.md")):
            hits.append(
                {
                    "doc_type": doc_type,
                    "account_id": account_id,
                    "date": _date_from_filename(f),
                    "text": f.read_text(),
                }
            )

    return hits


def _global_hits() -> list[dict]:
    hits: list[dict] = []
    for f in sorted((MOCK_CRM_DIR / "playbooks").glob("*.md")):
        hits.append(
            {
                "doc_type": "playbook",
                "account_id": GLOBAL_ACCOUNT_ID,
                "date": None,
                "text": f.read_text(),
            }
        )
    catalog = MOCK_CRM_DIR / "product_catalog.json"
    if catalog.exists():
        hits.append(
            {
                "doc_type": "product_catalog",
                "account_id": GLOBAL_ACCOUNT_ID,
                "date": None,
                "text": catalog.read_text(),
            }
        )
    return hits


def list_account_ids() -> list[str]:
    return sorted(p.stem for p in (MOCK_CRM_DIR / "accounts").glob("*.json"))


def assemble_context(account_id: Optional[str]) -> list[dict]:
    """Every hit relevant to `account_id` (that account's own documents +
    global playbooks/catalog), or every account's documents + global
    material if account_id is None (portfolio-wide)."""
    if account_id is not None:
        return _account_hits(account_id) + _global_hits()

    hits: list[dict] = []
    for acc in list_account_ids():
        hits.extend(_account_hits(acc))
    hits.extend(_global_hits())
    return hits


if __name__ == "__main__":
    for hit in assemble_context("acme_corp"):
        print(f"[{hit['doc_type']} | {hit['account_id']} | {hit['date'] or 'n/a'}]")
