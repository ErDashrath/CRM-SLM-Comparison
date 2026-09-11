"""
Structured output schema shared by every variant in this project.

Ported verbatim from ~/Magna/SalesIntelligence/orchestrator/schemas.py so
teacher-drafted (KD), curated (SFT), and locally-generated (base) responses
are all validated against the identical shape -- a prerequisite for the
eval harness to compare them apples-to-apples. Duplicated rather than
imported across projects, deliberately (see models/inference.py's module
docstring for why this project stays decoupled from SalesIntelligence's
live repo).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    DRAFT_EMAIL = "draft_email"
    SCHEDULE_MEETING = "schedule_meeting"
    ESCALATE_INTERNAL = "escalate_internal"
    LOG_RISK_NOTE = "log_risk_note"
    UPDATE_CRM_FIELD = "update_crm_field"
    RECOMMEND_DISCOUNT = "recommend_discount"
    NO_ACTION = "no_action"
    OTHER = "other"


class NextBestAction(BaseModel):
    """The structured recommendation every model variant must emit as JSON."""

    action_type: ActionType = Field(
        ..., description="The category of action being recommended."
    )
    target_object: str = Field(
        ...,
        description=(
            "The CRM object this action applies to, e.g. an account_id "
            "('acme_corp') or opportunity_id ('opp_acme_corp_001')."
        ),
    )
    payload: dict = Field(
        default_factory=dict,
        description="Free-form details for the action; shape depends on action_type.",
    )
    rationale: str = Field(
        ...,
        description=(
            "Why this action is recommended, grounded in the retrieved "
            "context. Should name specific evidence, not just assert a "
            "conclusion."
        ),
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Short strings naming risks relevant to this recommendation.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in this recommendation, 0-1.",
    )

    model_config = ConfigDict(use_enum_values=True)
