"""
Phase 1, no-API-key path, step 2: merge sub-agent-generated responses with
data/kd_prompts.jsonl, parse + guardrail-filter, and produce the same
data/kd_train.jsonl / kd_rejected.jsonl / kd_manifest.json shape
generate_teacher_drafts.py would have written via a live API call.

Expects one or more batch output files (JSONL, each line
{"index": i, "raw_response": "<json text the sub-agent produced>"}) under
data/kd_batch_output/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.formatting import parse_next_best_action
from eval import guardrails

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KD_PROMPTS_PATH = PROJECT_ROOT / "data" / "kd_prompts.jsonl"
KD_BATCH_OUTPUT_DIR = PROJECT_ROOT / "data" / "kd_batch_output"
KD_TRAIN_PATH = PROJECT_ROOT / "data" / "kd_train.jsonl"
KD_REJECTED_PATH = PROJECT_ROOT / "data" / "kd_rejected.jsonl"
KD_UNPARSEABLE_PATH = PROJECT_ROOT / "data" / "kd_unparseable.jsonl"
KD_MANIFEST_PATH = PROJECT_ROOT / "data" / "kd_manifest.json"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    prompts = {row["index"]: row for row in _load_jsonl(KD_PROMPTS_PATH)}

    batch_files = sorted(KD_BATCH_OUTPUT_DIR.glob("*.jsonl"))
    if not batch_files:
        print(f"No batch output files found under {KD_BATCH_OUTPUT_DIR}. Nothing to merge.")
        return

    responses: dict[int, str] = {}
    for bf in batch_files:
        for row in _load_jsonl(bf):
            responses[row["index"]] = row["raw_response"]

    missing = sorted(set(prompts) - set(responses))
    if missing:
        print(f"WARNING: {len(missing)} prompt(s) have no response yet: {missing}")

    accepted = rejected = unparseable = 0
    with open(KD_TRAIN_PATH, "w") as f_ok, open(KD_REJECTED_PATH, "w") as f_bad, open(
        KD_UNPARSEABLE_PATH, "w"
    ) as f_unparse:
        for index in sorted(responses):
            prompt_row = prompts[index]
            raw = responses[index]
            action = parse_next_best_action(raw)

            if action is None:
                f_unparse.write(
                    json.dumps({"index": index, "query": prompt_row["query"], "raw": raw}, ensure_ascii=False)
                    + "\n"
                )
                unparseable += 1
                continue

            violations = guardrails.post_check(action, prompt_row["evidence_context"])
            row = {
                "query": prompt_row["query"],
                "account_id": prompt_row["account_id"],
                "context": prompt_row["context"],
                "response": json.loads(action.model_dump_json()),
                "_source": "claude_code_agent_teacher_draft",
            }
            if violations:
                row["_guardrail_violations"] = violations
                f_bad.write(json.dumps(row, ensure_ascii=False) + "\n")
                rejected += 1
            else:
                f_ok.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1

    manifest = {
        "total_queries": len(prompts),
        "responses_received": len(responses),
        "accepted": accepted,
        "rejected": rejected,
        "unparseable": unparseable,
        "teacher_backend": "mixed_batch_outputs (existing sub-agent batches plus Anthropic API continuation)",
        "human_review_minutes": 0,
    }
    KD_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(
        f"{accepted} accepted -> {KD_TRAIN_PATH}, {rejected} guardrail-rejected -> "
        f"{KD_REJECTED_PATH}, {unparseable} unparseable -> {KD_UNPARSEABLE_PATH}. "
        f"Manifest: {KD_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
