"""
Prompt assembly + response parsing shared by every variant in this project.

Ported from ~/Magna/SalesIntelligence/orchestrator/pipeline.py so all three
model variants (base/kd/sft) and the teacher-drafting script see an
identical prompt shape -- required for the comparison to isolate "what was
trained in" rather than "how the prompt happened to differ."
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import yaml

from common.schemas import NextBestAction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "common" / "system_prompts" / "v1_base.yaml"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        spec = yaml.safe_load(f)
    return (
        f"{spec['role_instructions']}\n\n"
        f"{spec['domain_rules']}\n\n"
        f"{spec['output_schema']}"
    )


def format_context(hits: list[dict]) -> str:
    """Full context text -- what the model/UI actually sees, including
    global playbook/catalog material (the model does need policy
    knowledge)."""
    if not hits:
        return "(no retrieved context)"
    lines = []
    for hit in hits:
        header = f"[{hit['doc_type']} | {hit['account_id']} | {hit.get('date') or 'n/a'}]"
        lines.append(f"{header}\n{hit['text']}")
    return "\n\n".join(lines)


def format_evidence_context(hits: list[dict]) -> str:
    """Same as format_context but excludes account_id=="global" hits.

    Used ONLY for guardrails.pre_check/post_check's risk-indicator
    detection, never for what the model/UI sees. Reason (found during
    SalesIntelligence's A2 dataset review, see eval/guardrails.py's module
    docstring): RISK_INDICATORS' context_regex checks matched generic
    playbook boilerplate that mentions a risk concept in the abstract, even
    when the specific account has no actual evidence of it. Risk
    *existence* must be judged from account-specific evidence
    (emails/transcripts), not company-wide policy text.
    """
    from common.context import GLOBAL_ACCOUNT_ID

    return format_context([h for h in hits if h.get("account_id") != GLOBAL_ACCOUNT_ID])


def build_user_prompt(query: str, context_text: str, pre_warnings: list[str]) -> str:
    warnings_block = "\n".join(f"- {w}" for w in pre_warnings)
    return (
        "=== Retrieved context ===\n"
        f"{context_text}\n\n"
        "=== Guardrail constraints (you must respect these) ===\n"
        f"{warnings_block}\n\n"
        "=== Question ===\n"
        f"{query}\n\n"
        "Respond with the NextBestAction JSON object now."
    )


def extract_json_object(text: str) -> Optional[dict]:
    """Parse `text` as JSON directly; falling back to extracting the first
    {...} block handles models that wrap JSON in markdown fences or prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_next_best_action(raw_text: str) -> Optional[NextBestAction]:
    obj = extract_json_object(raw_text)
    if obj is None:
        return None
    try:
        return NextBestAction.model_validate(obj)
    except Exception:
        return None
