"""
Re-score an existing results/eval_results.json against the current
eval/guardrails.py, without re-running any generations or judge calls.

Exists because guardrails.py had 4 real bugs found and fixed 2026-09-10
(dedup, response_lag false-positive, discount_ask_above_policy rubber
stamp, exact-10% boundary) plus a design change (policy-threshold
citations no longer flagged, only the customer's ask) -- the underlying
model responses didn't change, only how they're scored, so re-generating
would waste ~25-30 minutes for no reason.

Usage:
    python -m eval.rescore_results
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context import assemble_context  # noqa: E402
from common.context_compaction import compact_formatted_context  # noqa: E402
from common.formatting import format_evidence_context, parse_next_best_action  # noqa: E402
from eval import guardrails  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "eval_results.json"
SUMMARIES_PATH = PROJECT_ROOT / "data" / "context_summaries.json"
COMPACT_CONTEXT_CHARS = 1800  # matches eval/run_eval.py


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    summaries = json.loads(SUMMARIES_PATH.read_text()) if SUMMARIES_PATH.exists() else {}

    backup_path = RESULTS_PATH.with_name(f"eval_results_v2_pre_guardrail_fix_{datetime.now().strftime('%Y%m%d-%H%M')}.json")
    shutil.copy(RESULTS_PATH, backup_path)
    print(f"Backed up pre-fix results to {backup_path}")

    changed = 0
    for r in results:
        if not r["parsed_ok"]:
            continue  # unparseable responses are unaffected by guardrail logic
        action = parse_next_best_action(r["response_text"])
        hits = assemble_context(r["account_id"])
        evidence_text = compact_formatted_context(
            format_evidence_context(hits), COMPACT_CONTEXT_CHARS, summaries=summaries
        )
        new_violations = guardrails.post_check(action, evidence_text)
        new_pass = len(new_violations) == 0
        if new_pass != r["guardrail_pass"] or new_violations != r["guardrail_violations"]:
            changed += 1
        r["guardrail_pass"] = new_pass
        r["guardrail_violations"] = new_violations

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Re-scored {len(results)} results, {changed} changed. Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
