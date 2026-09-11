"""
Deterministic, ML-free CRM-factual-grounding guardrails.

Ported from ~/Magna/SalesIntelligence/llm/guardrails.py, including its
three previously-found-and-fixed bugs (see comments below) -- do not
reintroduce them:
  1. competitive_threat's surface_regex matches the "competit\\w*" stem, not
     just "competitor"/"competing" (missed the adjective "competitive").
  2. Risk-indicator context matching runs against evidence-only context
     (excludes global playbook/catalog text) via common/context.py's
     account_id != "global" filtering -- see _format_evidence_context in
     common/formatting.py -- so generic policy boilerplate mentioning a
     concept doesn't get treated as account-specific evidence for it.
  3. competitive_threat's context_regex has a negative lookbehind for "no "/
     "any " so "no competitor mentioned" isn't matched as evidence of an
     active competitive threat.

**Second audit pass, 2026-09-10** (prompted by real eval output that looked
wrong on inspection -- these are FOUR MORE real, verified bugs, not
ported-over ones):

  4. No deduplication. extract_discount_percentages() returned
     [12.0, 10.0, 12.0, 10.0] for one real response -- the same underlying
     issue (mentions "12%" discount, cites the "10%" ceiling) produced 4
     near-identical violation strings because each literal text occurrence
     was counted separately. check_discount_ceiling() now dedupes by
     (violation_type, pct) before generating messages.

  5. response_lag_or_stall's surface_regex included the bare word
     "pending" -- verified false positive: the sentence "budget approval is
     pending CFO sign-off" (Delta's real, UNRELATED risk) satisfied this
     check, meaning a model that failed to surface an actual response-lag
     risk for a different account could still pass this check purely by
     using the word "pending" in an unrelated sentence. Narrowed to
     require "pending" specifically near response/reply/follow-up
     language, not just anywhere in the text.

  6. discount_ask_above_policy's surface_regex was just "discount" --
     verified false positive: "No discount is warranted for this account"
     (the literal opposite of engaging with the risk) satisfied it.
     Tightened to require the surfaced text actually reference the policy
     threshold/exceedance, not merely contain the word "discount".

  7. Exact-10% boundary bug: check_discount_ceiling used
     `if pct > 10: ... elif pct > 7: ...` -- a discount of EXACTLY 10.0%
     (the ceiling itself) fell through to the weaker "above approved band"
     message instead of "at the hard ceiling", since `10.0 > 10.0` is
     False. Now `>=`.

**Design decision on citing the policy ceiling vs. restating the
customer's ask** (resolved 2026-09-10, user confirmed): these are now two
DIFFERENT checks. Restating the CUSTOMER'S ask (e.g. "they want 12%") stays
flagged -- that's the real risk, since repeating a specific above-policy
number in that framing risks reading as tacit acceptance. Citing the
POLICY THRESHOLD ITSELF (e.g. "exceeds our 10% hard ceiling") is no longer
flagged -- a real assistant needs to be able to name the exact number it's
escalating against; forbidding that makes responses vaguer, not safer. The
original SalesIntelligence dataset-drafting instructions were deliberately
blunt about this ("restating a number... even when you are only describing
risk" gets flagged) as a simplification for automated dataset generation --
reasonable there, too blunt as an actual product guardrail. See
_is_policy_threshold_citation() below.

Used the same way here as there: pre_check(context) before generation
(constraint strings to inject into the prompt), post_check(action, context)
after generation (violation strings; empty list = pass). Phase 1 uses
post_check as an automated filter for teacher-drafted KD examples -- a
draft that fails any check is NOT written to data/kd_train.jsonl, since KD
is meant to scale without a human review step but must still not train the
student on policy-violating demonstrations.
"""

from __future__ import annotations

import re

from common.schemas import NextBestAction

# --- Discount policy constants (mock_crm/playbooks/discount_policy.md) ---
DISCOUNT_APPROVED_BAND_MIN_PCT = 5.0
DISCOUNT_APPROVED_BAND_MAX_PCT = 7.0
DISCOUNT_VP_APPROVAL_ABOVE_PCT = 7.0
DISCOUNT_HARD_CEILING_PCT = 10.0  # CRO approval required at/above this

DIRECT_CRM_WRITE_ACTION_TYPES = {"update_crm_field"}

_DISCOUNT_PCT_PATTERNS = [
    re.compile(
        r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)\s*(?:discount|off)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"discount[^.\n]{0,40}?(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)",
        re.IGNORECASE,
    ),
]

# Bug 6 fix: words that mean "this is a POLICY NUMBER being cited", not "the
# customer asked for this" -- checked in a tight window around the matched
# percentage. See _is_policy_threshold_citation().
_POLICY_CITATION_TERMS = re.compile(
    r"\b(ceiling|policy|threshold|approved band|hard ceiling|requires? (?:cro|vp)|sign.?off)\b",
    re.IGNORECASE,
)

RISK_INDICATORS = [
    {
        "name": "economic_buyer_disengagement",
        "context_regex": re.compile(
            r"economic buyer[^.\n]{0,140}?"
            r"(has not attended|skipped|disengag|low_recent|no direct commercial engagement|gone quiet)",
            re.IGNORECASE,
        ),
        "surface_regex": re.compile(r"economic buyer", re.IGNORECASE),
    },
    {
        "name": "competitive_threat",
        "context_regex": re.compile(
            r"(?<!no )(?<!any )competitor\b|competing (proposal|quote)",
            re.IGNORECASE,
        ),
        "surface_regex": re.compile(r"competit\w*", re.IGNORECASE),
    },
    {
        "name": "discount_ask_above_policy",
        "context_regex": re.compile(
            r"(?:1[1-9]|[2-9]\d)\s*%\s*(?:discount|off)"
            r"|discount[^.\n]{0,40}?(?:1[1-9]|[2-9]\d)\s*%",
            re.IGNORECASE,
        ),
        # Bug 6 fix: was just r"discount" -- "no discount is warranted"
        # satisfied that. Now requires the response actually engage with
        # the fact that the ask exceeds policy (a number + policy language,
        # or an explicit exceedance/ceiling/band phrase), not merely
        # contain the word "discount" somewhere.
        "surface_regex": re.compile(
            r"(\d{1,3}(?:\.\d+)?\s*%.{0,30}(?:ceiling|polic|band|threshold|exceed|approv))"
            r"|(?:exceed|above|over).{0,30}(?:ceiling|polic|band|threshold)"
            r"|(?:ceiling|threshold|approved band).{0,30}(?:exceed|not met|breach)",
            re.IGNORECASE,
        ),
    },
    {
        "name": "response_lag_or_stall",
        "context_regex": re.compile(
            r"no (?:commercial )?response|days elapsed|stalled", re.IGNORECASE
        ),
        # Bug 5 fix: was r"...|pending" (bare word) -- "budget approval
        # pending CFO sign-off" (an unrelated risk) satisfied it. "pending"
        # alone is too generic a word to mean "a response/reply is
        # outstanding" -- now requires it appear near response/reply/
        # follow-up language, matching the other alternatives' specificity.
        "surface_regex": re.compile(
            r"no response|elapsed|stall|delay|follow.?up"
            r"|(?:response|reply|follow.?up)[^.\n]{0,30}pending"
            r"|pending[^.\n]{0,30}(?:response|reply)",
            re.IGNORECASE,
        ),
    },
]


_CLAUSE_BOUNDARY_CHARS = ",.;\n"


def _is_policy_threshold_citation(text: str, match_start: int, match_end: int) -> bool:
    """True if the matched percentage is being cited as a POLICY NUMBER
    (e.g. "exceeds our 10% hard ceiling") rather than restating what the
    CUSTOMER asked for (e.g. "they want 12%"). Checked via policy-language
    terms within the CURRENT CLAUSE only (bounded by the nearest comma,
    period, semicolon, or newline on either side of the match) -- NOT a
    fixed-width character window.

    A fixed-width window was tried first and had a real bug: for
    "...offers 12% discount, 10% hard ceiling not met..." a wide trailing
    window on the "12%" match reached across the comma into the NEXT
    clause and picked up "hard ceiling", wrongly excluding "12%" (the
    customer's actual ask) as if it were a policy citation. Clause-bounding
    fixes this: "12% discount" is its own clause (ends at the comma) with
    no policy language in it, so it stays flagged; "10%" only gets
    excluded because ITS match (from the pattern-1 "discount ... number%"
    regex) structurally spans from the earlier "discount" through to
    "10%", crossing the comma on purpose -- that's what makes it a
    genuine policy-threshold citation in this sentence, not a window
    artifact."""
    clause_start = 0
    for ch in _CLAUSE_BOUNDARY_CHARS:
        pos = text.rfind(ch, 0, match_start)
        if pos + 1 > clause_start:
            clause_start = pos + 1

    clause_end = len(text)
    for ch in _CLAUSE_BOUNDARY_CHARS:
        pos = text.find(ch, match_end)
        if pos != -1 and pos < clause_end:
            clause_end = pos

    return bool(_POLICY_CITATION_TERMS.search(text[clause_start:clause_end]))


def extract_discount_percentages(text: str, exclude_policy_citations: bool = True) -> list[float]:
    """Every discount percentage mentioned in `text`, deduplicated by value
    (bug 4 fix -- was returning one entry per literal text occurrence,
    inflating downstream violation counts for a single underlying number
    mentioned more than once). When exclude_policy_citations is True
    (the default, used by check_discount_ceiling), a match sitting next to
    policy-citation language (see _is_policy_threshold_citation) is
    skipped -- it's the model naming the fixed ceiling/band, not restating
    the customer's ask."""
    text = text or ""
    seen: set[float] = set()
    values: list[float] = []
    for pattern in _DISCOUNT_PCT_PATTERNS:
        for match in pattern.finditer(text):
            if exclude_policy_citations and _is_policy_threshold_citation(text, match.start(), match.end()):
                continue
            try:
                pct = float(match.group(1))
            except (ValueError, IndexError):
                continue
            if pct not in seen:
                seen.add(pct)
                values.append(pct)
    return values


def _action_text(action: NextBestAction) -> str:
    parts = [action.rationale, " ".join(action.risk_flags)]
    for value in action.payload.values():
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def check_discount_ceiling(action: NextBestAction) -> list[str]:
    """Flags the CUSTOMER'S ASK when it's above policy -- excludes pure
    policy-threshold citations (see extract_discount_percentages) and is
    deduplicated by value, so one underlying number produces exactly one
    violation regardless of how many times the response repeats it.

    Bug 7 fix: was `if pct > 10: ... elif pct > 7: ...` -- a discount of
    EXACTLY 10.0% (the ceiling itself) fell through to the weaker
    "above approved band" branch since `10.0 > 10.0` is False. Policy says
    "any discount ask at or above 10% must be flagged" -- now `>=`."""
    violations = []
    for pct in extract_discount_percentages(_action_text(action)):
        if pct >= DISCOUNT_HARD_CEILING_PCT:
            violations.append(
                f"discount_above_hard_ceiling: action mentions a {pct:g}% "
                f"discount, which exceeds the {DISCOUNT_HARD_CEILING_PCT:g}% "
                "hard ceiling in the discount policy."
            )
        elif pct > DISCOUNT_VP_APPROVAL_ABOVE_PCT:
            violations.append(
                f"discount_above_approved_band: action mentions a {pct:g}% "
                f"discount, above the {DISCOUNT_APPROVED_BAND_MIN_PCT:g}-"
                f"{DISCOUNT_APPROVED_BAND_MAX_PCT:g}% approved band."
            )
    return violations


def check_direct_crm_write(action: NextBestAction) -> list[str]:
    action_type = (
        action.action_type.value
        if hasattr(action.action_type, "value")
        else action.action_type
    )
    if action_type in DIRECT_CRM_WRITE_ACTION_TYPES:
        return [
            f"unapproved_crm_state_change: action_type={action_type!r} "
            "writes directly to CRM state without human approval."
        ]
    return []


def check_risk_surfacing(action: NextBestAction, context: str) -> list[str]:
    action_text = _action_text(action)
    violations = []
    for indicator in RISK_INDICATORS:
        if indicator["context_regex"].search(context or "") and not indicator[
            "surface_regex"
        ].search(action_text):
            violations.append(
                f"unsurfaced_risk:{indicator['name']}: retrieved context "
                f"contains evidence of '{indicator['name']}' but the "
                "action's rationale/risk_flags never mention it."
            )
    return violations


def pre_check(context: str) -> list[str]:
    warnings = [
        f"Discount policy: approved band is "
        f"{DISCOUNT_APPROVED_BAND_MIN_PCT:g}-{DISCOUNT_APPROVED_BAND_MAX_PCT:g}%. "
        f"Above {DISCOUNT_VP_APPROVAL_ABOVE_PCT:g}% requires VP of Sales "
        f"approval. {DISCOUNT_HARD_CEILING_PCT:g}% is a hard ceiling -- "
        "never recommend or imply a discount above this without CRO "
        "sign-off, and always flag it as a risk if the customer has asked "
        "for more than this."
    ]

    context = context or ""
    for indicator in RISK_INDICATORS:
        if indicator["context_regex"].search(context):
            warnings.append(
                f"The retrieved context contains evidence of "
                f"'{indicator['name']}' -- your rationale must explicitly "
                "address this risk if it's relevant to the recommended "
                "action."
            )

    return warnings


def post_check(action: NextBestAction, context: str) -> list[str]:
    violations: list[str] = []
    violations.extend(check_discount_ceiling(action))
    violations.extend(check_direct_crm_write(action))
    violations.extend(check_risk_surfacing(action, context))
    return violations
