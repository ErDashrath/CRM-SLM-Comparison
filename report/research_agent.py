"""
Phase 5 -- the "proper research agent with proper prompt" the project asked
for. Reads results/eval_results.json's aggregate numbers plus known
qualitative facts (training data volume, human review time, licensing,
methodology caveats) and produces a structured, numbers-cited tradeoffs
writeup via the teacher backend (Claude) -- not a hand-authored summary,
and not generic hedging.

Output: results/tradeoffs-<timestamp>.md AND the text is handed to
report/build_excel.py to populate the workbook's "Tradeoffs" sheet.

The prompt is deliberately loaded with the SAME caveats this project
surfaced the hard way while building it (token-budget bug found and fixed,
judge/teacher same-model-family bias, Delta not a true unseen account,
small dataset sizes, base model downsized from 4B to 1.7B) -- instructed
to actually reason about their implications, not just append them as a
disclaimer list.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_backend import get_teacher_backend  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "eval_results.json"
KD_MANIFEST_PATH = PROJECT_ROOT / "data" / "kd_manifest.json"
SFT_MANIFEST_PATH = PROJECT_ROOT / "data" / "sft_manifest.json"

RESEARCH_AGENT_SYSTEM_PROMPT = (
    "You are a research analyst producing a tradeoffs report for a CTO who "
    "will use it to decide how to specialize a small (1.7B) open model for a "
    "CRM sales-assistant task, comparing three approaches: (1) the base model "
    "with no fine-tuning, (2) knowledge distillation from a stronger teacher "
    "model with no human review of the resulting data, and (3) supervised "
    "fine-tuning on a smaller, curated dataset.\n\n"
    "You will be given real quantitative results from running all three "
    "variants against the same 20 held-out evaluation queries, plus known "
    "qualitative facts and methodology caveats. Your job:\n\n"
    "1. State what the numbers actually show -- cite specific figures, don't "
    "round them away into vague language like 'performed better.'\n"
    "2. Reason about WHY, using the qualitative facts given (dataset size, "
    "review process, training approach) -- don't just restate the numbers.\n"
    "3. Take the methodology caveats seriously and reason about how they "
    "might affect the numbers' reliability -- don't just append them as a "
    "disclaimer paragraph at the end.\n"
    "4. Give a direct recommendation. Not 'it depends' -- a real answer, "
    "with the reasoning that leads to it, and what would change your mind.\n\n"
    "Do not hedge for the sake of hedging. Do not pad with generic "
    "boilerplate about fine-tuning tradeoffs -- every claim must trace back "
    "to a specific number or fact given below. Write in plain prose, no "
    "markdown headers inside sections (the section structure is fixed, "
    "provided below)."
)

REPORT_STRUCTURE = """
Produce the report in exactly this structure (markdown headers as shown):

## What the Numbers Show

## Why: Data Volume, Curation, and What Each Approach Actually Optimized For

## Methodology Caveats and Their Likely Effect on These Results

## The Recommendation

## What Would Change This Recommendation
"""


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def compute_aggregate_summary() -> dict:
    results = json.loads(RESULTS_PATH.read_text())
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)

    summary = {}
    for variant, rows in by_variant.items():
        n = len(rows)
        scored = [r for r in rows if r["judge"]["correctness"] is not None]
        summary[variant] = {
            "n_queries": n,
            "parsed_ok": sum(r["parsed_ok"] for r in rows),
            "guardrail_pass": sum(r["guardrail_pass"] for r in rows),
            "avg_correctness": round(sum(r["judge"]["correctness"] for r in scored) / len(scored), 2),
            "avg_completeness": round(sum(r["judge"]["completeness"] for r in scored) / len(scored), 2),
            "avg_risk_surfacing": round(sum(r["judge"]["risk_surfacing"] for r in scored) / len(scored), 2),
            "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / n, 1),
            "avg_tokens_per_second": round(
                sum(r["tokens_per_second"] for r in rows if r["tokens_per_second"]) / n, 1
            ),
        }
    return summary


def build_context_block() -> str:
    summary = compute_aggregate_summary()
    kd_manifest = _load_json(KD_MANIFEST_PATH)
    sft_manifest = _load_json(SFT_MANIFEST_PATH)

    return f"""
=== Aggregate evaluation results (20 held-out queries, none seen during training) ===
{json.dumps(summary, indent=2)}

=== Training data facts ===
Base: no training data, no fine-tuning.
KD (knowledge-distilled): {kd_manifest.get('accepted', '?')} examples, teacher-generated,
  guardrail-filtered only (no human review), 0 human-review minutes. Teacher backend
  varied across the run: some batches via Claude Code sub-agents acting directly as
  teacher (no API key available yet), some via the Anthropic API once a key was added.
SFT (fine-tuned): {sft_manifest.get('total_selected', '?')} examples, selected via
  {sft_manifest.get('selection_method', 'unknown method')} from the same teacher-drafted
  pool KD used -- stratified per account by confidence, top ~65% kept per account
  (NOT a global top-N, which would have skewed toward the easiest account and
  excluded the hardest one). 0 human-review minutes -- the user opted to skip manual
  review, so this is NOT genuinely "curated by a human," despite the name; the only
  difference from KD is automated selection criteria and volume, not human judgment.

=== Base model ===
Qwen3-1.7B-Instruct (Apache 2.0). Originally planned as Qwen3-4B-Instruct-2507, but
switched mid-project: the 4B model OOM'd during training on both the local T2000 (4GB)
and a free-tier Colab T4 (16GB) at every context length tried down to 5120 tokens --
Turing-architecture GPUs (both cards) lack a fused/flash attention kernel, so attention
memory scales quadratically with sequence length regardless of card size. Neither L4
nor A100 (which do have flash attention) are available on this Colab account's tier.

=== Context compaction ===
Real CRM context (account/opportunity records, emails, transcripts, playbooks) averaged
~5,700 prompt tokens, some over 13,000 -- far more than this hardware could train on.
Fixed via LLM-summarized compaction (unique documents summarized once via Claude, cached,
reused across every query that references them) down to a ~1800-character budget, at
which 100% of training examples retained their complete target JSON (no completions lost
to truncation). The SAME compaction is applied at evaluation time, for all 3 variants
including base, specifically to avoid a train/inference mismatch.

=== Methodology caveats ===
1. Judge/teacher same-family bias: the evaluation judge is Claude (same model family
   that generated the KD training data), which could bias KD's judged quality upward
   relative to an independent judge -- yet KD still scored LOWEST of the three variants
   on every judge dimension. This makes the caveat work AGAINST what was actually
   observed, which is itself worth reasoning about.
2. Delta was NOT held out as a true unseen-account generalization test, despite that
   being the original plan -- Phase 1's teacher generation used all 4 mock accounts,
   including Delta, so no account-level generalization is actually being tested here,
   only query-level (the 20 eval queries are worded differently from the 58 training
   queries but touch the same 4 accounts).
3. Small sample sizes throughout: 44 KD examples, 31 SFT examples, only 20 eval queries.
4. A real bug was found and fixed during Phase 4: the first eval run used a too-tight
   token budget (700) that the more verbose kd/sft variants hit far more often than
   base, producing an apparent "fine-tuning made JSON parsing worse" pattern that was
   actually just truncation. Re-run at 1200 tokens; the numbers above are from the
   corrected run.
5. LoRA rank 8, only 22 training steps per adapter (2 epochs at this dataset size) --
   a very light-touch fine-tune by any standard, not a large training investment.
6. A second, deeper bug was found and fixed in the guardrail logic itself (not the
   eval harness) after visual inspection of real UI output: the discount-ceiling
   check (a) never deduplicated repeated mentions of the same number, so one
   underlying issue could produce 2-4 near-identical violation lines; (b) could not
   tell "citing the fixed policy ceiling to justify an escalation" (e.g. "exceeds our
   10% hard ceiling") apart from "restating what the customer asked for" (e.g. "they
   want 12%") -- both used to get flagged identically; (c) the response_lag_or_stall
   risk-surfacing check had a surface_regex that matched the bare word "pending",
   which false-triggered on completely unrelated content (e.g. Delta's real
   "budget approval pending" risk), letting genuine unsurfaced-lag cases pass
   silently; (d) the discount_ask_above_policy risk-surfacing check's surface_regex
   was just the word "discount", so "no discount is warranted" (the literal opposite
   of engaging with the risk) counted as surfacing it; (e) an exact 10.0% discount
   fell through to the weaker "approved band" message instead of "hard ceiling",
   since the original code used a strict `>` where the policy says "at or above 10%"
   should be flagged. All five were fixed, and the EXISTING 60 responses were
   RE-SCORED (not re-generated -- the model outputs did not change) against the
   corrected logic. This raised the overall guardrail pass rate from 18/60 (30%) to
   25/60 (42%), and importantly made the per-variant numbers genuinely different for
   the first time (base 9/20, kd 9/20, sft 7/20) rather than an artificial three-way
   tie at 6/20 each -- the original tie was itself a symptom of these bugs equally
   polluting all three variants' scores, not a real finding about generalization.
""".strip()


def main() -> None:
    context_block = build_context_block()
    backend = get_teacher_backend()

    user_prompt = f"{context_block}\n\n{REPORT_STRUCTURE}\n\nWrite the report now."
    # 2000, then 3000, both confirmed truncated mid-word/mid-sentence once
    # caveat 6 (the guardrail bug writeup) was added to the input context --
    # this report reliably runs long enough to need real headroom. 5000
    # verified sufficient (see module history for the two prior failures).
    report_text = backend.generate(RESEARCH_AGENT_SYSTEM_PROMPT, user_prompt, max_tokens=5000)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = PROJECT_ROOT / "results" / f"tradeoffs-{timestamp}.md"
    out_path.write_text(report_text)
    print(f"Wrote tradeoffs report to {out_path}")

    # Also save the raw context block + report together for build_excel.py
    # to consume without re-deriving the aggregate numbers.
    bundle_path = PROJECT_ROOT / "results" / "tradeoffs_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {"aggregate_summary": compute_aggregate_summary(), "report_text": report_text, "report_path": str(out_path)},
            indent=2,
        )
    )
    print(f"Wrote bundle for report/build_excel.py to {bundle_path}")


if __name__ == "__main__":
    main()
