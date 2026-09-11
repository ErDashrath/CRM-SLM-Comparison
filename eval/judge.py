"""
Phase 4 -- Claude-as-judge scoring for eval/run_eval.py.

Scores one model response against the FULL, uncompacted account context
(common/context.py::assemble_context() + common/formatting.py::
format_context() -- the same real CRM documents training/eval assemble,
before any compaction) on three axes: correctness, completeness,
risk-surfacing. Returns 1-5 integer scores plus a short rationale.

**Changed 2026-09-11** (was: common/cheatsheet.py's deterministic
structured-fields extraction): the cheatsheet approach, while accurate,
only covered STRUCTURED facts (contacts, deal terms, the risk_factors
array) -- it couldn't judge nuance that only exists in an email or call
transcript's actual wording. The compacted context the models themselves
read (~1800 chars, LLM-summarized) exists ONLY because the 1.7B student
model has a hard VRAM/training-context ceiling this project's hardware
imposes -- the judge (Claude, via the API) has no such ceiling, so there's
no reason to hobble it with the same constraint. Feeding it the FULL
context is a strict superset of whatever any variant actually saw, so this
can only make the judge MORE rigorous and catch more real detail, never
less fair to any variant (all three still get graded against the same
full ground truth).

**Methodology caveat, stated plainly, not buried**: the judge backend
defaults to TEACHER_BACKEND=claude via common/llm_backend.py -- the SAME
model family that generated the KD training data in Phase 1. This can bias
KD's judged quality upward relative to SFT/base (a judge tends to favor
outputs that match its own reasoning style/conventions). This is flagged
here in code and must be flagged again in the final tradeoffs report
(report/research_agent.py) -- do not let this caveat get lost between here
and there.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context import assemble_context  # noqa: E402
from common.formatting import format_context  # noqa: E402
from common.llm_backend import get_teacher_backend  # noqa: E402

JUDGE_SYSTEM_PROMPT = (
    "You are grading a CRM sales-assistant's response against ground-truth facts. "
    "You will be given: (1) the FULL retrieved CRM context for the account(s) "
    "involved -- account/opportunity records, emails, call transcripts, and "
    "company-wide policy/playbook material, exactly as it exists in the real "
    "system (this is MORE than what the model being graded may have seen -- the "
    "model may have read a shortened, summarized version of this due to its own "
    "hardware limits; you are not under that limit, so grade against the full "
    "picture), (2) the user's question, (3) the model's response (a JSON "
    "NextBestAction object). Score the response on three axes, each 1-5:\n\n"
    "correctness (1-5): Are the facts the response states actually true per the "
    "ground truth? A response that states something false or contradicts the "
    "ground truth scores 1-2 regardless of how well-written it is.\n\n"
    "completeness (1-5): Does the response address what the question actually "
    "asked, using the relevant facts available? Missing the obviously relevant "
    "facts scores low even if what IS stated is accurate.\n\n"
    "risk_surfacing (1-5): If the ground truth lists risk factors relevant to "
    "this question, does the response name them? A response that ignores an "
    "applicable named risk factor scores low here specifically, even if the "
    "rest of the response is fine.\n\n"
    "Respond with ONLY a JSON object: "
    '{"correctness": <1-5>, "completeness": <1-5>, "risk_surfacing": <1-5>, '
    '"rationale": "<one or two sentences citing specifics>"}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_response(query: str, account_id: Optional[str], response_text: str, backend=None) -> dict:
    """Score one model response. `response_text` is the raw model output
    (may or may not be clean JSON -- the judge sees it as-is, unparseable
    output is itself a signal worth scoring low on). Returns
    {"correctness", "completeness", "risk_surfacing", "rationale"} or, if
    the judge itself fails to produce parseable output,
    {"correctness": None, ..., "rationale": "<judge failure note>"}."""
    backend = backend or get_teacher_backend()
    ground_truth = format_context(assemble_context(account_id))

    user_prompt = (
        f"=== Full CRM context (ground truth -- may include more than the model saw) ===\n{ground_truth}\n\n"
        f"=== Question ===\n{query}\n\n"
        f"=== Model response ===\n{response_text}\n\n"
        "Score this response now."
    )
    raw = backend.generate(JUDGE_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    scores = _extract_json(raw)
    if scores is None or not all(k in scores for k in ("correctness", "completeness", "risk_surfacing")):
        return {
            "correctness": None,
            "completeness": None,
            "risk_surfacing": None,
            "rationale": f"Judge produced unparseable output: {raw[:200]!r}",
        }
    return scores
