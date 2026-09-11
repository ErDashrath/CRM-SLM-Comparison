"""
One-time (cached) LLM summarization of every unique CRM document (emails,
transcripts, playbooks, product catalog), keyed by the same
"[doc_type | account_id | date]" header string common/formatting.py's
format_context() produces.

Why this exists: common/context_compaction.py originally shortened long
documents by blind character truncation (emails/transcripts) or
keyword-line matching (playbooks) -- fast and free, but genuinely lossy
(cuts mid-sentence, misses relevant lines phrased differently than the
keyword list). Summarization preserves meaning instead of chopping at an
arbitrary boundary.

Cost efficiency: the SAME documents are shared across every query for an
account (13 queries per account all reference the identical raw
emails/transcripts) -- summarizing once and reusing via this cache is both
cheaper and more consistent than summarizing per-training-example. Only
~30 unique prose documents exist across all 4 accounts + global
playbooks/catalog, so this is a small, one-time cost, not something that
scales with dataset size.

Structured account/opportunity JSON is NOT summarized here --
common/context_compaction.py's field-selection approach (keep only
decision-relevant keys, minify) already handles those well; summarization
is for prose documents (emails, transcripts, playbooks, catalog) only.

Usage:
    python -m data_gen.build_context_summaries
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context import assemble_context, list_account_ids  # noqa: E402
from common.llm_backend import get_teacher_backend  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_PATH = PROJECT_ROOT / "data" / "context_summaries.json"

SUMMARY_SYSTEM_PROMPT = (
    "You compress CRM documents (emails, call transcripts, internal playbooks, "
    "product catalogs) into short factual summaries for a downstream sales-risk "
    "model. Preserve every fact relevant to: deal risk, stakeholder "
    "engagement/disengagement, competitors, pricing/discount asks, policy "
    "numbers, timelines, and next steps. Drop pleasantries, greetings, and "
    "restating the obvious. Do not add commentary, opinions, or any fact not "
    "present in the source. 2-4 sentences maximum, plain prose, no markdown."
)


def _header(hit: dict) -> str:
    return f"[{hit['doc_type']} | {hit['account_id']} | {hit.get('date') or 'n/a'}]"


def summarize(backend, doc_type: str, text: str) -> str:
    user_prompt = f"Document type: {doc_type}\n\n{text}\n\nSummarize per the instructions."
    return backend.generate(SUMMARY_SYSTEM_PROMPT, user_prompt, max_tokens=200).strip()


def main() -> None:
    existing: dict[str, str] = {}
    if SUMMARIES_PATH.exists():
        existing = json.loads(SUMMARIES_PATH.read_text())

    # Collect every unique document once -- account-scoped hits repeat the
    # same global playbooks/catalog for every account, dedupe by header.
    seen: dict[str, dict] = {}
    for account_id in list_account_ids():
        for hit in assemble_context(account_id):
            seen[_header(hit)] = hit

    prose_docs = {h: hit for h, hit in seen.items() if hit["doc_type"] not in ("account", "opportunity")}
    to_summarize = {h: hit for h, hit in prose_docs.items() if h not in existing}

    print(
        f"{len(prose_docs)} unique prose documents (emails/transcripts/playbooks/catalog), "
        f"{len(to_summarize)} need summarizing ({len(prose_docs) - len(to_summarize)} already cached)."
    )

    if to_summarize:
        backend = get_teacher_backend()
        for i, (header, hit) in enumerate(to_summarize.items(), start=1):
            print(f"[{i}/{len(to_summarize)}] {header}")
            existing[header] = summarize(backend, hit["doc_type"], hit["text"])
            SUMMARIES_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    print(f"Done. {len(existing)} summaries cached at {SUMMARIES_PATH}")


if __name__ == "__main__":
    main()
