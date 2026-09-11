"""
Phase 4 -- run every query in data/eval_set.jsonl through every registered
model variant (models/registry.yaml), scoring guardrail pass/fail +
Claude-as-judge scores (correctness/completeness/risk_surfacing) + perf
(latency, tokens/sec, peak VRAM) for each. Writes results/eval_results.json
for report/build_excel.py to consume.

Loads one variant at a time (the 4GB T2000 can't hold two ~1.7B GGUFs
resident simultaneously) -- see models/inference.py's VariantHandle.close()
requirement.

Context compaction at eval time, deliberately consistent with training:
kd/sft were trained on LLM-summarized, compacted context (see
common/context_compaction.py, data_gen/build_context_summaries.py --
1800-char budget, 100% of training examples fit intact). Evaluating them
on the FULL uncompacted context instead would both exceed the inference
n_ctx (measured: 7181 tokens for one query against 5120 available) and
reintroduce exactly the train/inference mismatch flagged during Phase 3 --
kd/sft would be asked to parse a shape of input they were never trained
on. So this harness applies the SAME compact_formatted_context() call, at
the SAME budget, to every variant's input -- including "base", so the
comparison isolates "what was trained in," not "which variant got an
easier prompt."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context import assemble_context  # noqa: E402
from common.context_compaction import compact_formatted_context  # noqa: E402
from common.formatting import (  # noqa: E402
    build_user_prompt,
    format_context,
    format_evidence_context,
    load_system_prompt,
    parse_next_best_action,
)
from common.llm_backend import get_teacher_backend  # noqa: E402
from eval import guardrails  # noqa: E402
from eval.judge import judge_response  # noqa: E402
from eval.perf import timed_generate  # noqa: E402
from models.inference import list_variants, load  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.jsonl"
RESULTS_PATH = PROJECT_ROOT / "results" / "eval_results.json"
SUMMARIES_PATH = PROJECT_ROOT / "data" / "context_summaries.json"
COMPACT_CONTEXT_CHARS = 1800  # matches training/train_lora.py's --compact-context-chars used for kd/sft


def load_eval_queries() -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_variant(variant_name: str, queries: list[dict], judge_backend) -> list[dict]:
    handle = load(variant_name)
    system_prompt = load_system_prompt()
    summaries = json.loads(SUMMARIES_PATH.read_text()) if SUMMARIES_PATH.exists() else {}
    results = []
    try:
        for i, q in enumerate(queries, start=1):
            account_id = q["account_id"]
            hits = assemble_context(account_id)
            context_text = compact_formatted_context(
                format_context(hits), COMPACT_CONTEXT_CHARS, summaries=summaries
            )
            evidence_context_text = compact_formatted_context(
                format_evidence_context(hits), COMPACT_CONTEXT_CHARS, summaries=summaries
            )
            pre_warnings = guardrails.pre_check(evidence_context_text)
            user_prompt = build_user_prompt(q["query"], context_text, pre_warnings)

            # 700 was too tight: Qwen3's <think> block eats budget before the
            # JSON even starts, and it hit the ceiling disproportionately for
            # kd/sft (2/20, 5/20, 8/20 respectively in the first run 2026-09-10)
            # -- a token-budget artifact masquerading as a quality difference,
            # not a real one. 1200 gives real headroom for think + full JSON.
            perf = timed_generate(handle, system_prompt, user_prompt, max_tokens=1200)
            action = parse_next_best_action(perf["text"])

            if action is None:
                guardrail_pass = False
                violations = ["unparseable_response"]
            else:
                violations = guardrails.post_check(action, evidence_context_text)
                guardrail_pass = len(violations) == 0

            judge_scores = judge_response(q["query"], account_id, perf["text"], backend=judge_backend)

            row = {
                "variant": variant_name,
                "query": q["query"],
                "account_id": account_id,
                "response_text": perf["text"],
                "parsed_ok": action is not None,
                "guardrail_pass": guardrail_pass,
                "guardrail_violations": violations,
                "judge": judge_scores,
                "elapsed_seconds": perf["elapsed_seconds"],
                "output_tokens": perf["output_tokens"],
                "tokens_per_second": perf["tokens_per_second"],
                "peak_vram_mb": perf["peak_vram_mb"],
                "retried": perf["retried"],
            }
            results.append(row)
            print(
                f"  [{i}/{len(queries)}] {q['query'][:55]}... "
                f"guardrail={'PASS' if guardrail_pass else 'FAIL'} "
                f"judge={judge_scores.get('correctness')}/{judge_scores.get('completeness')}/{judge_scores.get('risk_surfacing')} "
                f"{perf['tokens_per_second']}tok/s"
            )
    finally:
        handle.close()
    return results


def main() -> None:
    queries = load_eval_queries()
    variants = list_variants()
    print(f"{len(queries)} eval queries x {len(variants)} variants = {len(queries) * len(variants)} generations")

    judge_backend = get_teacher_backend()  # constructed once, reused across every judge call

    all_results: list[dict] = []
    for variant_name in variants:
        print(f"=== Evaluating variant: {variant_name} ===")
        all_results.extend(run_variant(variant_name, queries, judge_backend))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(all_results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
