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

from common.schemas import ActionType, NextBestAction

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


_ACTION_TYPE_VALUES = {e.value for e in ActionType}
_ACTION_TYPE_ALIASES = {
    "update_crm": "update_crm_field",
    "escalate": "escalate_internal",
    "log_risk": "log_risk_note",
    "schedule": "schedule_meeting",
    "email": "draft_email",
    "draft_an_email": "draft_email",
    "discount": "recommend_discount",
    "none": "no_action",
    "no_action_needed": "no_action",
}


def repair_next_best_action_dict(obj: dict) -> dict:
    """Best-effort, deterministic fixes for common small-model JSON
    malformations -- applied BEFORE re-validating, so most failures never
    need a second model call at all. Only coerces types/defaults for
    fields the schema already treats as structurally optional or bounded
    (payload, risk_flags, confidence, action_type's exact spelling) --
    never invents target_object or rationale content, since those can't be
    safely guessed."""
    obj = dict(obj)

    action_type = obj.get("action_type")
    if isinstance(action_type, str):
        normalized = action_type.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in _ACTION_TYPE_VALUES:
            obj["action_type"] = normalized
        else:
            obj["action_type"] = _ACTION_TYPE_ALIASES.get(normalized, "other")

    confidence = obj.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence.strip().rstrip("%"))
            if confidence > 1.0:  # e.g. model wrote "80" meaning 80%
                confidence = confidence / 100.0
        except ValueError:
            confidence = 0.5
        obj["confidence"] = confidence
    if isinstance(obj.get("confidence"), (int, float)):
        obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
    elif obj.get("confidence") is None:
        obj["confidence"] = 0.5  # explicit low-confidence placeholder, never invented as high

    if not isinstance(obj.get("payload"), dict):
        obj["payload"] = {}

    risk_flags = obj.get("risk_flags")
    if not isinstance(risk_flags, list):
        obj["risk_flags"] = [str(risk_flags)] if risk_flags else []
    else:
        obj["risk_flags"] = [str(x) for x in risk_flags]

    return obj


def parse_next_best_action_diagnostic(raw_text: str) -> tuple[Optional[NextBestAction], Optional[str]]:
    """Like parse_next_best_action, but on failure returns (None, <specific
    reason>) instead of a bare None -- distinguishes "no JSON found" from
    "found JSON but it doesn't match the schema even after auto-repair",
    and the repair pass means many small-model malformations (wrong
    action_type spelling, confidence as a string/percentage, payload as a
    string instead of an object) succeed WITHOUT needing another
    generation call at all."""
    obj = extract_json_object(raw_text)
    if obj is None:
        return None, "no JSON object found in the response text"

    try:
        return NextBestAction.model_validate(obj), None
    except Exception:
        pass

    try:
        return NextBestAction.model_validate(repair_next_best_action_dict(obj)), None
    except Exception as second_error:
        return None, f"schema validation failed even after auto-repair: {second_error}"


def parse_next_best_action(raw_text: str) -> Optional[NextBestAction]:
    action, _ = parse_next_best_action_diagnostic(raw_text)
    return action
