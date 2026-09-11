"""
Phase 2, automated path (no human review): select the SFT training subset
from data/kd_train.jsonl without a manual review pass.

Chosen over "just take the global top-N by confidence" because confidence
turned out to correlate strongly with account (Delta's easy, low-risk
scenario clusters at 0.92-0.95; Acme's hardest, multi-risk scenario clusters
at 0.80-0.85) -- a global top-N would have produced an SFT set that's
mostly Delta and barely any Acme, which defeats the point of a CRM dataset
built specifically to have varied risk complexity per account. Stratifying
per account_id first keeps the SFT set's scenario mix representative,
matching what a real curator would insist on even without reading every row.

This is NOT human review. data/sft_manifest.json records
selection_method="automated_stratified_by_account_top_confidence" and
human_review_minutes=0 explicitly, so the eventual tradeoffs report doesn't
mislabel this as curation cost the project didn't actually pay. If real
human review happens later, run data_gen/curate_sft_subset.py instead and
it will overwrite data/sft_train.jsonl with a genuinely reviewed set.

Usage:
    python -m data_gen.auto_curate_sft --fraction 0.65
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KD_TRAIN_PATH = PROJECT_ROOT / "data" / "kd_train.jsonl"
SFT_TRAIN_PATH = PROJECT_ROOT / "data" / "sft_train.jsonl"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "sft_manifest.json"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.65,
        help="Fraction of each account's accepted drafts to keep, by confidence (default 0.65).",
    )
    args = parser.parse_args()

    rows = _load_jsonl(KD_TRAIN_PATH)
    by_account: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_account[row["account_id"] or "portfolio"].append(row)

    selected: list[dict] = []
    breakdown = {}
    for account, group in sorted(by_account.items()):
        group_sorted = sorted(group, key=lambda r: r["response"]["confidence"], reverse=True)
        keep_n = max(1, math.ceil(len(group_sorted) * args.fraction))
        kept = group_sorted[:keep_n]
        selected.extend(kept)
        breakdown[account] = {"available": len(group_sorted), "selected": keep_n}

    with open(SFT_TRAIN_PATH, "w") as f:
        for row in selected:
            out_row = {**row, "_source": "auto_curated_top_confidence_stratified"}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    manifest = {
        "total_available": len(rows),
        "total_selected": len(selected),
        "fraction_per_account": args.fraction,
        "breakdown_by_account": breakdown,
        "selection_method": "automated_stratified_by_account_top_confidence",
        "human_review_minutes": 0,
    }
    SFT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"Selected {len(selected)}/{len(rows)} rows -> {SFT_TRAIN_PATH}")
    for account, stats in breakdown.items():
        print(f"  {account}: {stats['selected']}/{stats['available']}")
    print(f"Manifest: {SFT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
