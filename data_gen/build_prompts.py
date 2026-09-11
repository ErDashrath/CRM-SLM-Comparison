"""
Phase 1, no-API-key path: build every teacher prompt up front as data,
instead of calling an external API backend per query.

Reuses generate_teacher_drafts.py's exact query set, context assembly, and
prompt construction (_build_drafting_prompt) -- so the prompts a human
teacher (via sub-agent, see below) sees are byte-identical to what
ClaudeAPIBackend/OpenAIAPIBackend would have been sent. Only the generation
mechanism differs: no ANTHROPIC_API_KEY or OPENAI_API_KEY exists with
budget on this machine (confirmed 2026-09-10), so instead of an API call,
each prompt is handed to a general-purpose sub-agent (via the Agent tool,
orchestrated outside this script) that generates the response directly as
Claude -- functionally the same "stronger model as teacher" the project
calls for, without needing a key at all.

Output: data/kd_prompts.jsonl, one line per query:
    {"index": i, "query": ..., "account_id": ..., "system_prompt": ...,
     "user_prompt": ..., "context": ..., "evidence_context": ...}

data_gen/merge_and_filter.py consumes the sub-agents' raw output alongside
this file to produce the final data/kd_train.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context import assemble_context
from common.formatting import format_context, format_evidence_context
from data_gen.generate_teacher_drafts import (
    _build_drafting_prompt,
    iter_queries,
)
from eval import guardrails

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KD_PROMPTS_PATH = PROJECT_ROOT / "data" / "kd_prompts.jsonl"


def main() -> None:
    pairs = list(iter_queries("all"))
    rows = []
    for i, (query, account_id) in enumerate(pairs):
        hits = assemble_context(account_id)
        context_text = format_context(hits)
        evidence_context_text = format_evidence_context(hits)
        pre_warnings = guardrails.pre_check(evidence_context_text)
        system_prompt, user_prompt = _build_drafting_prompt(query, context_text, pre_warnings)
        rows.append(
            {
                "index": i,
                "query": query,
                "account_id": account_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context": context_text,
                "evidence_context": evidence_context_text,
            }
        )

    with open(KD_PROMPTS_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {KD_PROMPTS_PATH}")


if __name__ == "__main__":
    main()
