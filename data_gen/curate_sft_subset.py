"""
Phase 2 -- SFT track: human review CLI over a subset of the teacher drafts.

This is the step that makes the SFT variant different from KD, not just a
smaller copy of it: a real person reads each candidate, corrects or accepts
it, and only reviewed rows become training data. Running this script IS the
"human review" cost line in the final tradeoffs report -- there is no way
to script around that without collapsing the actual comparison this project
exists to make, so this file deliberately does not auto-approve anything.

Reads data/kd_train.jsonl (Phase 1's guardrail-passed teacher drafts,
produced by data_gen/generate_teacher_drafts.py) and presents a subset of
them one at a time for review. Also offers rows from data/kd_rejected.jsonl
(drafts that failed a guardrail) since a human can often just fix the
specific violation rather than discard the whole example -- useful raw
material, not automatically excluded from consideration.

For each row: [a]ccept as-is, [e]dit rationale/risk_flags/payload inline,
[s]kip, [q]uit (saves progress so far). Timed with a wall-clock stopwatch
per row (paused while you're actually typing an edit doesn't matter here --
the number that goes in the report is "how long did a human spend on this
dataset," and typing time IS review time) so data/sft_manifest.json ends up
with a real, honest human_review_minutes figure -- not a guess.

Usage:
    python -m data_gen.curate_sft_subset --count 55
    python -m data_gen.curate_sft_subset --count 55 --include-rejected
    python -m data_gen.curate_sft_subset --resume   # continue an in-progress session
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.cheatsheet import account_cheatsheet, portfolio_cheatsheet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KD_TRAIN_PATH = PROJECT_ROOT / "data" / "kd_train.jsonl"
KD_REJECTED_PATH = PROJECT_ROOT / "data" / "kd_rejected.jsonl"
SFT_TRAIN_PATH = PROJECT_ROOT / "data" / "sft_train.jsonl"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "sft_manifest.json"
PROGRESS_PATH = PROJECT_ROOT / "data" / ".sft_curation_progress.json"

RANDOM_SEED = 42  # reproducible sample selection


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"reviewed_indices": [], "elapsed_seconds": 0.0, "accepted": 0, "skipped": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def _print_row(row: dict) -> None:
    print("\n" + "=" * 78)
    account_id = row.get("account_id")
    print(f"Account: {account_id or 'portfolio-wide'}")
    print("-" * 78)
    print("GROUND TRUTH (from mock_crm/, not the draft -- check the draft against this):")
    if account_id:
        print(account_cheatsheet(account_id))
    else:
        print(portfolio_cheatsheet())
    print("-" * 78)
    print(f"Query:   {row['query']}")
    print("-" * 78)
    response = row["response"]
    print(f"action_type: {response['action_type']}")
    print(f"target_object: {response['target_object']}")
    print(f"confidence: {response['confidence']}")
    print(f"rationale: {response['rationale']}")
    print(f"risk_flags: {response['risk_flags']}")
    print(f"payload: {json.dumps(response['payload'])}")
    if row.get("_guardrail_violations"):
        print(f"\n!! Originally REJECTED for: {row['_guardrail_violations']}")
    print("=" * 78)


def _edit_response(response: dict) -> dict:
    print("Editing -- press Enter to keep the current value for each field.")
    new_rationale = input(f"rationale [{response['rationale'][:60]}...]: ").strip()
    if new_rationale:
        response["rationale"] = new_rationale
    new_flags = input(f"risk_flags (comma-separated) [{response['risk_flags']}]: ").strip()
    if new_flags:
        response["risk_flags"] = [f.strip() for f in new_flags.split(",") if f.strip()]
    new_confidence = input(f"confidence [{response['confidence']}]: ").strip()
    if new_confidence:
        try:
            response["confidence"] = float(new_confidence)
        except ValueError:
            print("  (not a number, keeping original)")
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=55, help="How many candidates to review this session.")
    parser.add_argument("--include-rejected", action="store_true", help="Also draw from kd_rejected.jsonl.")
    parser.add_argument("--resume", action="store_true", help="Continue a previous in-progress session.")
    args = parser.parse_args()

    pool = _load_jsonl(KD_TRAIN_PATH)
    if args.include_rejected:
        pool += _load_jsonl(KD_REJECTED_PATH)

    if not pool:
        print(
            f"No candidates found in {KD_TRAIN_PATH}"
            + (f" or {KD_REJECTED_PATH}" if args.include_rejected else "")
            + ". Run data_gen/generate_teacher_drafts.py first (Phase 1)."
        )
        return

    random.Random(RANDOM_SEED).shuffle(pool)
    pool = pool[: args.count]

    progress = _load_progress() if args.resume else {
        "reviewed_indices": [],
        "elapsed_seconds": 0.0,
        "accepted": 0,
        "skipped": 0,
    }
    reviewed = set(progress["reviewed_indices"])

    sft_file = open(SFT_TRAIN_PATH, "a" if args.resume else "w")

    print(
        f"Reviewing {len(pool)} candidates ({len(reviewed)} already done). "
        f"[a]ccept  [e]dit  [s]kip  [q]uit-and-save"
    )

    for i, row in enumerate(pool):
        if i in reviewed:
            continue
        _print_row(row)
        start = time.time()
        choice = input("Action [a/e/s/q]: ").strip().lower()

        if choice == "q":
            progress["elapsed_seconds"] += time.time() - start
            _save_progress(progress)
            break

        if choice == "s":
            progress["skipped"] += 1
        elif choice in ("a", "e"):
            response = row["response"]
            if choice == "e":
                response = _edit_response(response)
            out_row = {
                "query": row["query"],
                "account_id": row["account_id"],
                "context": row["context"],
                "response": response,
                "_source": "human_reviewed" if choice == "e" else "human_accepted_as_is",
            }
            sft_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            sft_file.flush()
            progress["accepted"] += 1
        else:
            print("  Unrecognized input, treating as skip.")
            progress["skipped"] += 1

        progress["elapsed_seconds"] += time.time() - start
        reviewed.add(i)
        progress["reviewed_indices"] = sorted(reviewed)
        _save_progress(progress)

    sft_file.close()

    manifest = {
        "total_reviewed": len(reviewed),
        "accepted": progress["accepted"],
        "skipped": progress["skipped"],
        "human_review_minutes": round(progress["elapsed_seconds"] / 60, 1),
    }
    SFT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(
        f"\n{progress['accepted']} accepted -> {SFT_TRAIN_PATH}, "
        f"{progress['skipped']} skipped. "
        f"Total human review time so far: {manifest['human_review_minutes']} min. "
        f"Manifest: {SFT_MANIFEST_PATH}"
    )
    if len(reviewed) < len(pool):
        print(f"{len(pool) - len(reviewed)} candidates remain -- rerun with --resume to continue.")


if __name__ == "__main__":
    main()
