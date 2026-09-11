"""
Phase 1 -- KD track: generate teacher demonstrations for data/kd_train.jsonl.

For each query in QUERY_TEMPLATES/PORTFOLIO_QUERIES (the same 58-query set
proven out in SalesIntelligence's llm/sft/generate_candidates.py -- 13
queries x 4 accounts + 6 portfolio-wide), assembles account-scoped context
directly from mock_crm/ (common/context.py, no RAG layer), calls the
teacher backend (common/llm_backend.py, TEACHER_BACKEND env var --
currently "openai"/gpt-4o-mini, no ANTHROPIC_API_KEY configured yet),
parses the response as a NextBestAction, and guardrail-filters it
(eval/guardrails.post_check against the evidence-only context).

Everything that passes goes STRAIGHT to data/kd_train.jsonl -- unlike
SalesIntelligence's SFT drafting (which explicitly expects human review
before anything is trusted), KD is supposed to scale without a human
bottleneck. A guardrail-filtered draft that's still wrong in some way a
regex can't catch is a real, expected failure mode of this technique --
that's exactly the tradeoff the final report needs to surface, not paper
over.

Usage:
    python -m data_gen.generate_teacher_drafts --dry-run
    python -m data_gen.generate_teacher_drafts
    python -m data_gen.generate_teacher_drafts --account acme_corp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context import assemble_context
from common.formatting import (
    build_user_prompt,
    format_context,
    format_evidence_context,
    load_system_prompt,
    parse_next_best_action,
)
from common.llm_backend import LLMBackend, get_teacher_backend
from eval import guardrails

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KD_TRAIN_PATH = PROJECT_ROOT / "data" / "kd_train.jsonl"
KD_REJECTED_PATH = PROJECT_ROOT / "data" / "kd_rejected.jsonl"
KD_MANIFEST_PATH = PROJECT_ROOT / "data" / "kd_manifest.json"

ACCOUNT_IDS = ["acme_corp", "globex", "initech", "delta"]

# Same query set as SalesIntelligence's llm/sft/generate_candidates.py --
# proven, grounded in this exact mock CRM data. Reused rather than
# reinvented; see that file for the design rationale in its own comments.
QUERY_TEMPLATES: dict[str, list[str]] = {
    "acme_corp": [
        "Analyse the Acme Corp opportunity and tell me the current deal status, key risks, probability of closure, and what I should do next.",
        "What's the state of the Acme Corp deal and what should I do about it?",
        "Give me a risk assessment for Acme Corp before Friday's forecast call.",
        "Acme Corp's procurement lead is pushing hard on price -- what should I tell them?",
        "Is the Acme Corp discount request something I can approve, and if not, what's the right next step?",
        "Summarize the commercial negotiation status for Acme Corp for my manager.",
        "What are the biggest risks to closing Acme Corp on schedule?",
        "Should I escalate the Acme Corp deal internally, and if so to whom and why?",
        "Acme Corp mentioned a competitor's lower quote -- how should I respond?",
        "Walk me through Acme Corp's deal health: engagement, competition, and pricing.",
        "What's blocking Acme Corp from closing, and what's the highest-leverage next action?",
        "Prepare a next-best-action recommendation for the Acme Corp account.",
        "Is there anything about Acme Corp I should flag before it slips this quarter?",
    ],
    "globex": [
        "Analyse the Globex opportunity and tell me the current deal status, key risks, probability of closure, and what I should do next.",
        "What's the state of the Globex deal and what should I do about it?",
        "Is Globex on track to close on schedule?",
        "Give me a risk assessment for the Globex account.",
        "What should my next action be on the Globex opportunity?",
        "Summarize the Globex deal for a pipeline review.",
        "Are there any open items I need to resolve before Globex signs?",
        "Does Globex need any pricing exception or approval before closing?",
        "Walk me through Globex's deal health: engagement, competition, and pricing.",
        "What's the highest-leverage next action for the Globex account this week?",
        "Prepare a next-best-action recommendation for the Globex account.",
        "Is Globex's economic buyer still engaged?",
        "Anything concerning about Globex ahead of quarter close?",
    ],
    "initech": [
        "Analyse the Initech opportunity and tell me the current deal status, key risks, and what I should do next.",
        "What's the state of the Initech deal and what should I do about it?",
        "Initech has gone quiet -- should we still be chasing this deal?",
        "Give me a risk assessment for the Initech account.",
        "What's the highest-leverage next action for re-engaging Initech?",
        "Summarize the Initech deal for a pipeline review -- is it stalled?",
        "Should I escalate the Initech opportunity internally?",
        "Walk me through Initech's deal health: engagement, competition, and pricing.",
        "Is there a pricing conversation to have with Initech, or is that not the blocker?",
        "Prepare a next-best-action recommendation for the Initech account.",
        "What should I do about Initech's unanswered follow-up emails?",
        "Is Initech's win probability accurate given recent activity?",
        "Anything concerning about Initech ahead of quarter close?",
    ],
    "delta": [
        "Analyse the Delta Freight & Warehousing opportunity and tell me the current deal status, key risks, and what I should do next.",
        "What's the state of the Delta deal and what should I do about it?",
        "Is Delta's pricing within policy, and is there anything to flag?",
        "Give me a risk assessment for the Delta account.",
        "What's the highest-leverage next action for the Delta opportunity?",
        "Summarize the Delta deal for a pipeline review.",
        "Is Delta's budget-committee timeline a real risk to the close date?",
        "Walk me through Delta's deal health: engagement, competition, and pricing.",
        "Does Delta need any escalation before quarter end?",
        "Prepare a next-best-action recommendation for the Delta account.",
        "Is Delta's economic buyer still engaged?",
        "Anything concerning about Delta ahead of quarter close?",
        "Should I follow up with Delta's CFO directly about budget sign-off?",
    ],
}

PORTFOLIO_QUERIES: list[str] = [
    "Which accounts in my portfolio are at risk right now and why?",
    "Give me a portfolio-wide risk scan across all my open opportunities.",
    "Which deals need my attention this week, and what should I do about each?",
    "Rank my open opportunities by risk and tell me why.",
    "Are there any discount requests across my portfolio that exceed policy?",
    "Which accounts have gone quiet and might need re-engagement?",
]


def _kd_drafting_instructions() -> str:
    """KD-specific framing, distinct from SalesIntelligence's SFT-drafting
    instructions: no human will review or edit this before it becomes a
    training example, so the guardrail-compatibility rules matter even more
    here, not less."""
    discount_policy_path = PROJECT_ROOT / "mock_crm" / "playbooks" / "discount_policy.md"
    discount_policy_text = discount_policy_path.read_text() if discount_policy_path.exists() else ""

    return (
        "\n\n=== Training demonstration instructions (not part of a live "
        "product prompt -- this output becomes a KNOWLEDGE-DISTILLATION "
        "training example for a smaller model, with NO human review before "
        "it is used) ===\n"
        "Produce the best possible NextBestAction: thorough, precise, and "
        "grounded only in the retrieved context. Because nothing will "
        "correct this output before it is trained on, get it right the "
        "first time.\n\n"
        "Full discount policy for reference:\n"
        f"{discount_policy_text}\n\n"
        "Guardrail-compatibility rules -- follow these exactly, they are "
        "checked programmatically:\n"
        "  1. NEVER restate a specific discount percentage number in the "
        "same sentence/vicinity as the word 'discount' or 'off' if that "
        "number is above 7% -- describe it qualitatively instead.\n"
        "  2. If the retrieved context shows evidence of economic-buyer "
        "disengagement, a competitor, an above-policy discount ask, or a "
        "stalled/delayed response, you MUST explicitly name that risk in "
        "rationale and/or risk_flags, and if a topic is only generic "
        "playbook material and NOT actually a risk for this specific "
        "account, say so explicitly rather than omitting it.\n"
        "  3. NEVER use action_type='update_crm_field' -- use "
        "'log_risk_note', 'draft_email', 'schedule_meeting', "
        "'escalate_internal', or 'no_action' instead.\n"
        "  4. Ground every claim in the retrieved context only.\n"
    )


def _build_drafting_prompt(query: str, context_text: str, pre_warnings: list[str]) -> tuple[str, str]:
    system_prompt = load_system_prompt() + _kd_drafting_instructions()
    user_prompt = build_user_prompt(query, context_text, pre_warnings)
    return system_prompt, user_prompt


def generate_one(backend: LLMBackend, query: str, account_id: str | None) -> dict:
    """Assemble context, call the teacher, parse + guardrail-check the
    result. Returns {"status": "accepted"|"rejected"|"unparseable", ...}."""
    hits = assemble_context(account_id)
    context_text = format_context(hits)
    evidence_context_text = format_evidence_context(hits)
    pre_warnings = guardrails.pre_check(evidence_context_text)

    system_prompt, user_prompt = _build_drafting_prompt(query, context_text, pre_warnings)

    raw = backend.generate(system_prompt, user_prompt, max_tokens=900)
    action = parse_next_best_action(raw)
    if action is None:
        strict_prompt = (
            user_prompt
            + "\n\nYour previous response was not valid JSON. Return ONLY "
            "a single valid JSON object matching the schema above."
        )
        raw = backend.generate(system_prompt, strict_prompt, max_tokens=900)
        action = parse_next_best_action(raw)

    if action is None:
        return {"status": "unparseable", "query": query, "account_id": account_id, "raw": raw}

    violations = guardrails.post_check(action, evidence_context_text)
    row = {
        "query": query,
        "account_id": account_id,
        "context": context_text,
        "response": json.loads(action.model_dump_json()),
        "_source": "openai_teacher_draft",
    }
    if violations:
        row["_guardrail_violations"] = violations
        return {"status": "rejected", **row}
    return {"status": "accepted", **row}


def iter_queries(account: str):
    accounts = ACCOUNT_IDS if account == "all" else [account]
    for acc in accounts:
        for query in QUERY_TEMPLATES[acc]:
            yield query, acc
    if account == "all":
        for query in PORTFOLIO_QUERIES:
            yield query, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", choices=ACCOUNT_IDS + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    pairs = list(iter_queries(args.account))

    if args.dry_run:
        print(f"[dry-run] {len(pairs)} queries would be drafted:")
        for query, account_id in pairs:
            print(f"  - ({account_id or 'portfolio-wide'}) {query}")
        print("[dry-run] No API calls made, nothing written.")
        return

    backend = get_teacher_backend()

    accepted = rejected = unparseable = 0
    with open(KD_TRAIN_PATH, "w") as f_ok, open(KD_REJECTED_PATH, "w") as f_bad:
        for i, (query, account_id) in enumerate(pairs, start=1):
            print(f"[{i}/{len(pairs)}] ({account_id or 'portfolio-wide'}) {query[:70]}...")
            result = generate_one(backend, query, account_id)
            if result["status"] == "accepted":
                f_ok.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_ok.flush()
                accepted += 1
            elif result["status"] == "rejected":
                f_bad.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_bad.flush()
                rejected += 1
                print(f"  REJECTED (guardrail): {result['_guardrail_violations']}")
            else:
                unparseable += 1
                print("  UNPARSEABLE, skipped.")
            if i < len(pairs):
                time.sleep(args.sleep_seconds)

    manifest = {
        "total_queries": len(pairs),
        "accepted": accepted,
        "rejected": rejected,
        "unparseable": unparseable,
        "teacher_backend": "openai/gpt-4o-mini",
        "human_review_minutes": 0,
    }
    KD_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(
        f"\nDone. {accepted} accepted -> {KD_TRAIN_PATH}, "
        f"{rejected} guardrail-rejected -> {KD_REJECTED_PATH}, "
        f"{unparseable} unparseable/skipped. Manifest: {KD_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
