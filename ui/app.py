"""
ChatGPT-style CRM assistant UI.

Direct conversational chat with the CRM — powered by RAG + tool calling
(common/rag_chat.py) and the selected model variant (base / KD / SFT).

The model comparison research pipeline (guardrails, NextBestAction JSON,
eval harness) is intentionally NOT wired into this UI. It remains in
eval/run_eval.py for batch experiments. This UI is for real use.

Run with:
    /home/dsp-at-magna/Magna/venv-gpu/bin/python -m streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from common.context import list_account_ids
from common.crm_store import database_snapshot
from common.memory.store import SessionStore
from common.rag_chat import rag_chat
from models.inference import list_variants

st.set_page_config(
    page_title="CRM assistant",
    page_icon=":material/chat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_LABELS = {
    "base": "Base",
    "kd": "KD",
    "sft": "SFT",
}

SUGGESTIONS = [
    "Which accounts are at risk right now?",
    "Give me a pipeline overview by stage.",
    "What's the deal status for Acme Corp?",
    "Who are the key contacts at TechNova?",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

SESSION_STORE = SessionStore()

st.session_state.setdefault("conversations", [])
st.session_state.setdefault("active_id", None)
st.session_state.setdefault("selected_models", ["base"])
st.session_state.setdefault("compare_mode", False)
st.session_state.setdefault("account_id", None)
st.session_state.setdefault("score_with_judge", False)
if not st.session_state.get("_sessions_loaded"):
    persisted = SESSION_STORE.list_conversations()
    if persisted:
        st.session_state.conversations = persisted
        st.session_state.active_id = persisted[0]["id"]
    st.session_state._sessions_loaded = True


def _new_conversation() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }


def _active_conv() -> dict:
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c
    # Bootstrap first conversation
    conv = _new_conversation()
    st.session_state.conversations.insert(0, conv)
    st.session_state.active_id = conv["id"]
    SESSION_STORE.save_conversation(conv, st.session_state.get("account_id"))
    return conv


def _auto_title(text: str) -> str:
    clean = text.strip().replace("\n", " ")
    return (clean[:48] + "…") if len(clean) > 48 else clean


def _group_by_date(convs: list[dict]) -> dict[str, list[dict]]:
    now = datetime.now()
    groups: dict[str, list[dict]] = {"Today": [], "Yesterday": [], "Earlier": []}
    for c in convs:
        try:
            delta = (now - datetime.fromisoformat(c["created_at"])).days
        except Exception:
            delta = 999
        if delta == 0:
            groups["Today"].append(c)
        elif delta == 1:
            groups["Yesterday"].append(c)
        else:
            groups["Earlier"].append(c)
    return groups


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

variant_names = list_variants()

with st.sidebar:

    # New chat
    if st.button(
        "New chat",
        icon=":material/add:",
        type="primary",
        width="stretch",
    ):
        conv = _new_conversation()
        st.session_state.conversations.insert(0, conv)
        st.session_state.active_id = conv["id"]
        SESSION_STORE.save_conversation(conv, st.session_state.get("account_id"))
        st.rerun()

    st.space("small")

    # LLM-as-judge toggle — prominent as per wireframe
    st.session_state.score_with_judge = st.toggle(
        "LLM as judge",
        value=st.session_state.score_with_judge,
        help="After each response, runs a Claude judge scoring quality (slower — adds an API call).",
    )

    st.space("small")

    # Chat history
    if st.session_state.conversations:
        grouped = _group_by_date(st.session_state.conversations)
        for label, convs in grouped.items():
            if not convs:
                continue
            st.caption(label)
            for conv in convs:
                is_active = conv["id"] == st.session_state.active_id
                if st.button(
                    conv["title"],
                    key=f"conv_{conv['id']}",
                    type="tertiary",
                    icon=":material/chat:" if is_active else ":material/chat_bubble_outline:",
                    width="stretch",
                ):
                    st.session_state.active_id = conv["id"]
                    st.rerun()

    # Settings — at bottom of sidebar
    st.space("medium")
    with st.expander("Settings", icon=":material/settings:"):
        account_ids = list_account_ids()
        account_options = ["Portfolio-wide"] + account_ids
        choice = st.selectbox(
            "Account scope",
            account_options,
            key="account_scope",
        )
        st.session_state.account_id = (
            None if choice == "Portfolio-wide" else choice
        )
        try:
            snap = database_snapshot()
            st.caption(
                f":material/database: {snap['accounts']} accounts · "
                f"{snap['opportunities']} opportunities"
            )
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Main area — model selector then chat
# ---------------------------------------------------------------------------

conv = _active_conv()

# ── Model selector row ──────────────────────────────────────────────────────
with st.container(horizontal=True):
    compare_on = st.toggle(
        "Compare",
        value=st.session_state.compare_mode,
        key="compare_toggle",
        help="Run all selected models and show responses side by side.",
    )
    st.session_state.compare_mode = compare_on

    if compare_on:
        sel = st.pills(
            "Select models",
            variant_names,
            selection_mode="multi",
            default=st.session_state.selected_models,
            key="model_pills",
            label_visibility="collapsed",
        )
        if sel:
            st.session_state.selected_models = list(sel)
    else:
        current = (
            st.session_state.selected_models[0]
            if st.session_state.selected_models
            else "base"
        )
        sel = st.segmented_control(
            "Model",
            variant_names,
            default=current,
            key="model_seg",
            label_visibility="collapsed",
        )
        st.session_state.selected_models = [sel] if sel else ["base"]

# ---------------------------------------------------------------------------
# Response renderer — clean, no guardrails exposed
# ---------------------------------------------------------------------------

def _render_model_trace(result: dict) -> None:
    """Show the exact local-model inputs, tool plan, and raw outputs on demand."""
    trace = result.get("debug_trace") or {}
    if not trace:
        return
    with st.expander("Model trace", icon=":material/visibility:"):
        st.caption(
            "This is the exact execution for this variant. CRM text is shown here because trace viewing was requested."
        )
        overview = {
            key: trace[key]
            for key in (
                "route", "variant", "account_scope", "original_question",
                "effective_question", "memory", "status", "model_max_tokens",
                "budget_limits",
            )
            if key in trace
        }
        if overview:
            st.json(overview, expanded=False)

        prompt_sections = (
            ("Planner system prompt", "planner_system_prompt"),
            ("Planner user prompt", "planner_user_prompt"),
            ("Final system prompt", "final_system_prompt"),
            ("Final assembled prompt", "final_prompt"),
        )
        for label, key in prompt_sections:
            if trace.get(key) is not None:
                st.markdown(f"**{label}**")
                st.code(str(trace[key]), language="text")

        structured_sections = (
            ("Raw planner output", "planner_raw_output"),
            ("Validated tool calls", "validated_tool_calls"),
            ("Tool result", "tool_result"),
            ("Retrieval and tool trace", "tool_trace"),
            ("Retrieved documents", "retrieved_documents"),
            ("Raw final model output", "final_raw_output"),
            ("Cleaned final answer", "final_cleaned_output"),
        )
        for label, key in structured_sections:
            if trace.get(key) is not None:
                st.markdown(f"**{label}**")
                value = trace[key]
                if isinstance(value, (dict, list)):
                    st.json(value, expanded=False)
                else:
                    st.code(str(value), language="text")


def _render_chat_answer(result: dict, label: str | None = None) -> None:
    """Render a single model's answer inside a chat bubble."""
    if label:
        st.badge(label, color="violet", icon=":material/smart_toy:")

    # Casual / greeting — just show the reply, no metadata
    if result.get("is_casual"):
        st.write(result.get("answer", ""))
        _render_model_trace(result)
        return

    limits = result.get("limits") or {}
    if limits.get("status") == "history_compacted":
        st.info(
            "Earlier conversation turns were summarized/removed to stay within the model context.",
            icon=":material/compress:",
        )
    elif limits.get("status") == "context_limit_reached":
        st.warning(
            "This request reached the model context limit. Narrow the account, date range, or question.",
            icon=":material/warning:",
        )
    elif limits.get("status") == "session_limit_warning":
        st.warning(
            "This session is nearing its configured usage limit.",
            icon=":material/hourglass_top:",
        )
    elif limits.get("status") == "session_limit_reached":
        st.warning(
            "This session limit has been reached. Start a new session to continue.",
            icon=":material/block:",
        )

    if not result.get("model_used") and result.get("answer"):
        st.info(result["answer"], icon=":material/search:")
        return

    answer = result.get("answer", "")
    if answer:
        st.write(answer)
    else:
        st.caption("_(no response)_")

    _render_model_trace(result)

    # Collapsible evidence — kept out of the way
    retrieved = result.get("retrieved", [])
    tool_result = result.get("tool_result", {})
    tool_rows = tool_result.get("rows") if tool_result else None

    if tool_rows:
        with st.expander("Structured data", icon=":material/table:"):
            st.dataframe(tool_rows, width="stretch", hide_index=True)

    tool_trace = result.get("tool_trace", [])
    if tool_trace:
        with st.expander("Retrieval steps", icon=":material/build:"):
            for step in tool_trace:
                tool = step.get("tool", "tool")
                status = step.get("status", "ok")
                args = step.get("arguments", {})
                if step.get("fallback"):
                    st.caption(f"Fallback: **{tool}** · {status}")
                else:
                    st.caption(f"**{tool}** · {status} · `{args}`")

    if retrieved:
        with st.expander(f"Sources ({len(retrieved)})", icon=":material/search:"):
            for r in retrieved:
                st.caption(
                    f":material/article: **{r.get('source_type', '?')} — "
                    f"{r.get('title', '')}** · score {r.get('score', 0):.2f}"
                )

    if result.get("judge"):
        j = result["judge"]
        st.caption(
            f":material/star: Judge — correctness {j.get('correctness')}/5 · "
            f"completeness {j.get('completeness')}/5 · "
            f"risk-surfacing {j.get('risk_surfacing')}/5"
        )

    st.caption(
        f":material/model_training: {VARIANT_LABELS.get(result.get('variant', ''), result.get('variant', ''))}"
    )


# ── Chat messages ────────────────────────────────────────────────────────────

# Suggestion chips — only on empty chat
if not conv["messages"]:
    st.space("large")
    with st.container(horizontal_alignment="center"):
        st.markdown("### What can I help you with?")
        st.space("small")
        picked = st.pills(
            "Suggestions",
            SUGGESTIONS,
            label_visibility="collapsed",
        )
        if picked:
            conv["messages"].append(
                {"role": "user", "content": picked, "ts": datetime.now().isoformat()}
            )
            conv["title"] = _auto_title(picked)
            st.rerun()

# Render conversation history
for msg in conv["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])

    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            results: dict[str, dict] = msg.get("results", {})
            model_names = list(results.keys())

            if len(model_names) == 0:
                st.write(msg.get("content", ""))
            elif len(model_names) == 1:
                _render_chat_answer(results[model_names[0]])
            else:
                cols = st.columns(len(model_names))
                for col, vname in zip(cols, model_names):
                    with col:
                        _render_chat_answer(results[vname], label=VARIANT_LABELS.get(vname, vname))


# ── Chat input ───────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask about accounts, deals, pipeline, contacts…",
    submit_mode="disable",
)

if prompt:
    account_id = st.session_state.account_id
    conv["messages"].append(
        {"role": "user", "content": prompt, "ts": datetime.now().isoformat()}
    )
    SESSION_STORE.save_conversation(conv, account_id)
    if conv["title"] == "New chat":
        conv["title"] = _auto_title(prompt)

    with st.chat_message("user"):
        st.write(prompt)

    models_to_run = st.session_state.selected_models or ["base"]
    score_judge = st.session_state.score_with_judge

    # Build history for context (exclude the message we just added)
    history = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in conv["messages"][:-1]
        if m["role"] in ("user", "assistant") and m.get("content")
    ]

    # Judge backend (lazy)
    judge_backend = None
    if score_judge:
        try:
            from common.llm_backend import get_teacher_backend
            judge_backend = get_teacher_backend()
        except Exception:
            pass

    results: dict[str, dict] = {}

    with st.chat_message("assistant"):
        # Quick-check: if ALL variants would return a casual reply, skip the spinner
        from common.rag_chat import _classify_casual, _CASUAL_REPLIES  # noqa: PLC0415
        casual_key = _classify_casual(prompt)

        if casual_key is not None:
            # Casual message — instant reply, no model loading, no spinner
            casual_result = {
                "answer": _CASUAL_REPLIES[casual_key],
                "tool_result": {},
                "retrieved": [],
                "variant": models_to_run[0],
                "model_used": False,
                "is_casual": True,
            }
            for vname in models_to_run:
                results[vname] = {**casual_result, "variant": vname}
            _render_chat_answer(results[models_to_run[0]])

        else:
            # Real CRM query — show thinking spinner + step timeline
            with st.status(":shimmer[Thinking…]", type="compact") as status:
                for vname in models_to_run:
                    with st.status(
                        f"Running **{VARIANT_LABELS.get(vname, vname)}**",
                        type="step",
                    ):
                        session_limits = SESSION_STORE.limit_state(conv["id"])
                        if session_limits["session_limit_reached"]:
                            result = {
                                "answer": "This session limit has been reached. Start a new session to continue.",
                                "tool_result": {},
                                "retrieved": [],
                                "tool_trace": [],
                                "variant": vname,
                                "model_used": False,
                                "limits": session_limits,
                            }
                        else:
                            result = rag_chat(
                                question=prompt,
                                variant_name=vname,
                                account_id=account_id,
                                conversation_history=history,
                                session_id=conv["id"],
                            )
                            if session_limits["session_limit_warning"]:
                                result["limits"] = {**result.get("limits", {}), **session_limits}
                        if score_judge and judge_backend and result.get("model_used"):
                            try:
                                from eval.judge import judge_response
                                result["judge"] = judge_response(
                                    prompt,
                                    account_id,
                                    result["answer"],
                                    backend=judge_backend,
                                )
                            except Exception:
                                pass
                        results[vname] = result
                        st.write(f"Done — {VARIANT_LABELS.get(vname, vname)}")

                status.update(label="Done", state="complete")

            model_names = list(results.keys())
            if len(model_names) == 1:
                _render_chat_answer(results[model_names[0]])
            else:
                cols = st.columns(len(model_names))
                for col, vname in zip(cols, model_names):
                    with col:
                        _render_chat_answer(
                            results[vname],
                            label=VARIANT_LABELS.get(vname, vname),
                        )

    model_names = list(results.keys())
    # Store in conversation
    conv["messages"].append(
        {
            "role": "assistant",
            "content": results[model_names[0]].get("answer", "") if model_names else "",
            "results": results,
            "ts": datetime.now().isoformat(),
        }
    )
