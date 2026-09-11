"""
Phase 6 -- side-by-side comparison UI for CRM-SLM-Comparison.

Streamlit front-end that reads models/registry.yaml dynamically (via
models/inference.py's list_variants()/load()) -- adding a 4th variant to
the registry requires zero changes to this file.

Each variant runs INDEPENDENTLY: click "Run <variant>" for the current
account/query and only that variant executes. Every result is written into
a persistent session_state store keyed by (account, query, variant) and
stays there across reruns -- running a different variant, or a different
query, never overwrites or clears a result you already have. The results
section below groups everything by (account, query) and shows whichever
variants have actually been run for that combination side by side, so the
comparison grid naturally reflects whatever permutation/combination of
variants x queries you've explored, not a fixed 3-column layout.

Output is rendered as labeled fields (action type, confidence, rationale,
risk flags, payload) -- not a raw JSON/text dump. The model's <think>
block (Qwen3's reasoning trace) is shown separately, collapsed, since it
isn't the actual answer.

Also exposes a "Run Full Eval Suite" button that re-invokes
eval/run_eval.py -> report/research_agent.py -> report/build_excel.py and
links the resulting workbook.

VRAM constraint (same as eval/run_eval.py): the 4GB T2000 holds one ~1.7B
GGUF at a time, so each "Run <variant>" click loads and closes its own
handle -- deliberately NOT st.cache_resource'd the way SalesIntelligence's
hitl_ui/app.py caches its single backend, since caching multiple resident
variant handles here would attempt to hold all of them in VRAM
simultaneously and OOM.

Run with:
    /home/dsp-at-magna/Magna/venv-gpu/bin/python -m streamlit run ui/app.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from common.context import assemble_context, list_account_ids
from common.context_compaction import compact_formatted_context
from common.crm_assistant import answer_crm_question
from common.crm_store import database_snapshot
from common.formatting import build_user_prompt, format_context, format_evidence_context, load_system_prompt, parse_next_best_action
from eval import guardrails
from eval.judge import judge_response
from eval.perf import timed_generate
from models.inference import VariantNotBuiltError, list_variants, load
from data_gen.generate_teacher_drafts import QUERY_TEMPLATES, PORTFOLIO_QUERIES

st.set_page_config(page_title="CRM Small-Model Comparison", layout="wide")

COMPACT_CONTEXT_CHARS = 1800  # matches training/train_lora.py + eval/run_eval.py -- same input shape everywhere
SUMMARIES_PATH = _PROJECT_ROOT / "data" / "context_summaries.json"
RESULTS_DIR = _PROJECT_ROOT / "results"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@st.cache_data
def load_registry_labels() -> dict[str, str]:
    import yaml

    registry = yaml.safe_load((_PROJECT_ROOT / "models" / "registry.yaml").read_text())
    return {name: entry["label"] for name, entry in registry["variants"].items()}


@st.cache_data
def load_summaries() -> dict[str, str]:
    return json.loads(SUMMARIES_PATH.read_text()) if SUMMARIES_PATH.exists() else {}


def default_query(account_id: str | None) -> str:
    if account_id is None:
        return "Which accounts in my portfolio are at risk right now and why?"
    return "Analyse this opportunity and tell me the current deal status, key risks, and what I should do next."


@st.cache_data
def load_example_queries() -> dict:
    """Every training-time query (data_gen/generate_teacher_drafts.py's
    QUERY_TEMPLATES/PORTFOLIO_QUERIES) plus every eval-set query
    (data/eval_set.jsonl), grouped by account_id (None = portfolio-wide),
    deduplicated, order preserved. Powers the "choose an example query"
    picker so you're not stuck typing one from scratch."""
    examples: dict = defaultdict(list)
    for acc, queries in QUERY_TEMPLATES.items():
        examples[acc].extend(queries)
    examples[None].extend(PORTFOLIO_QUERIES)

    eval_set_path = _PROJECT_ROOT / "data" / "eval_set.jsonl"
    if eval_set_path.exists():
        for line in eval_set_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                examples[row["account_id"]].append(row["query"])

    deduped: dict = {}
    for acc, queries in examples.items():
        seen: set = set()
        deduped[acc] = [q for q in queries if not (q in seen or seen.add(q))]
    return deduped


def run_one_variant(variant_name: str, query: str, account_id: str | None, score_with_judge: bool, judge_backend) -> dict:
    summaries = load_summaries()
    hits = assemble_context(account_id)
    context_text = compact_formatted_context(format_context(hits), COMPACT_CONTEXT_CHARS, summaries=summaries)
    evidence_context_text = compact_formatted_context(
        format_evidence_context(hits), COMPACT_CONTEXT_CHARS, summaries=summaries
    )
    pre_warnings = guardrails.pre_check(evidence_context_text)
    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(query, context_text, pre_warnings)

    try:
        handle = load(variant_name)
    except VariantNotBuiltError as e:
        return {"variant": variant_name, "error": str(e)}

    try:
        perf = timed_generate(handle, system_prompt, user_prompt, max_tokens=1200)
    finally:
        handle.close()

    response_text = perf["text"]
    think_match = _THINK_RE.search(response_text)
    think_text = think_match.group(1).strip() if think_match else None

    action = parse_next_best_action(response_text)
    action_dict = action.model_dump() if action is not None else None

    if action is None:
        guardrail_pass, violations = False, ["unparseable_response"]
    else:
        violations = guardrails.post_check(action, evidence_context_text)
        guardrail_pass = len(violations) == 0

    judge_scores = None
    if score_with_judge:
        judge_scores = judge_response(query, account_id, response_text, backend=judge_backend)

    return {
        "variant": variant_name,
        "response_text": response_text,
        "think_text": think_text,
        "action": action_dict,
        "parsed_ok": action is not None,
        "guardrail_pass": guardrail_pass,
        "guardrail_violations": violations,
        "judge": judge_scores,
        "output_tokens": perf["output_tokens"],
        "tokens_per_second": perf["tokens_per_second"],
        "elapsed_seconds": perf["elapsed_seconds"],
        "retried": perf["retried"],
        "parse_error": perf["parse_error"],
    }


GUARDRAIL_EXPLAINER = """
**Guardrails are an automated compliance check on phrasing, not a quality score.**

No model is involved in the check itself -- it's plain pattern-matching against
this company's own written discount policy, the same category of tool as a
spell-checker or a compliance linter running over the AI's draft, not a verdict
on whether the response is good.

Two categories, and nothing else, get flagged:
- **Discount-ceiling** -- the response's text contains a specific number (like
  "12%") that policy says shouldn't be casually written out in a forward-facing
  recommendation, because it risks reading as if the number is being entertained
  rather than flagged as a problem.
- **Unsurfaced risk** -- the account data contained a known risk (a competitor,
  a disengaged decision-maker, a stalled deal) and the response didn't mention it.

**A response can FAIL this check while still being the best response.** That's
not a contradiction -- in this project's own results, the fine-tuned model had
the *highest* independent quality scores (most correct, most complete, best at
catching risks) but the *lowest* guardrail pass rate, because it tends to write
more thoroughly, including citing the exact numbers involved. Guardrail
pass/fail and answer quality are two separate, complementary signals here, not
one combined grade.
"""


def render_response_card(r: dict) -> None:
    if "error" in r:
        st.warning(r["error"])
        return

    if r["guardrail_pass"]:
        st.success("Guardrails: PASS", icon="✅")
    else:
        st.error("Guardrails: FAIL", icon="⚠️")
        for v in r["guardrail_violations"]:
            st.caption(f"⚠️ {v}")
    with st.expander("What does this mean?", expanded=False):
        st.markdown(GUARDRAIL_EXPLAINER)

    if r.get("judge"):
        j = r["judge"]
        st.markdown(
            f"**Judge:** correctness {j.get('correctness')}/5 &middot; "
            f"completeness {j.get('completeness')}/5 &middot; "
            f"risk-surfacing {j.get('risk_surfacing')}/5"
        )
        if j.get("rationale"):
            st.caption(j["rationale"])

    st.caption(f"{r['output_tokens']} tokens &middot; {r['tokens_per_second']} tok/s &middot; {r['elapsed_seconds']}s")
    if r.get("retried"):
        st.caption(
            "🔁 First attempt ran out of budget mid-reasoning with no JSON produced -- "
            "automatically retried once with a \"skip reasoning\" instruction. "
            "Tokens/time above cover both attempts combined."
        )
    st.divider()

    action = r.get("action")
    if action:
        st.markdown(f"**Action:** `{action['action_type']}` &rarr; `{action['target_object']}`")
        st.progress(action["confidence"], text=f"Confidence: {action['confidence']:.0%}")

        st.markdown("**Rationale**")
        st.write(action["rationale"])

        st.markdown("**Risk flags**")
        if action["risk_flags"]:
            for f in action["risk_flags"]:
                st.markdown(f"- {f}")
        else:
            st.caption("None surfaced.")

        if action["payload"]:
            st.markdown("**Payload**")
            for k, v in action["payload"].items():
                st.markdown(f"- **{k}:** {v}")
    else:
        reason = r.get("parse_error") or "unknown reason"
        st.warning(f"Could not parse a structured NextBestAction from this response: {reason}")
        st.text((r.get("response_text") or "")[:600])

    if r.get("think_text"):
        with st.expander("Model's reasoning (<think> block)", expanded=False):
            st.caption(r["think_text"])

    with st.expander("Raw response text", expanded=False):
        st.text(r.get("response_text", ""))


# --- Sidebar -----------------------------------------------------------

with st.sidebar:
    st.markdown("## Registered variants")
    st.caption("From models/registry.yaml -- add a 4th variant there, nothing here changes.")
    labels = load_registry_labels()
    for name, label in labels.items():
        st.markdown(f"- **{name}**: {label}")

    st.divider()
    st.markdown("## Run Full Eval Suite")
    st.caption("Re-runs eval/run_eval.py (20 held-out queries x every variant) then rebuilds the Excel report.")
    if st.button("Run Full Eval Suite", width="stretch"):
        with st.spinner("Running eval/run_eval.py -- this takes a while (many generations + judge calls)..."):
            result = subprocess.run(
                [sys.executable, "-m", "eval.run_eval"], cwd=str(_PROJECT_ROOT), capture_output=True, text=True
            )
        if result.returncode != 0:
            st.error(f"Eval run failed:\n{result.stderr[-2000:]}")
        else:
            with st.spinner("Building Excel report..."):
                result2 = subprocess.run(
                    [sys.executable, "-m", "report.research_agent"], cwd=str(_PROJECT_ROOT), capture_output=True, text=True
                )
                result3 = subprocess.run(
                    [sys.executable, "-m", "report.build_excel"], cwd=str(_PROJECT_ROOT), capture_output=True, text=True
                )
            if result3.returncode == 0:
                st.success("Done. Latest workbook:")
                xlsx_files = sorted(RESULTS_DIR.glob("results-*.xlsx"))
                if xlsx_files:
                    st.code(str(xlsx_files[-1]))
            else:
                st.error(f"Report build failed:\n{(result2.stderr + result3.stderr)[-2000:]}")

# --- Main: inputs + per-variant run buttons ---------------------------------

st.title("CRM Small-Model Comparison")
st.caption(
    "Same base model (Qwen3-1.7B-Instruct), three specializations: no fine-tuning, "
    "knowledge distillation, and supervised fine-tuning. Run each variant independently -- "
    "results stick around so you can build up a comparison across queries and variants."
)

# --- Tool-backed CRM chat -------------------------------------------------

st.subheader("CRM assistant")
st.caption(
    "Ask about structured CRM data first. The assistant uses read-only tools "
    "for counts, pipeline, account summaries, and risk views; it asks for "
    "scope when the current schema cannot answer safely."
)

if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []

for message in st.session_state.assistant_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("rows"):
            st.dataframe(message["rows"], width="stretch", hide_index=True)

assistant_prompt = st.chat_input("Ask about accounts, opportunities, pipeline, risks, or contacts")
if assistant_prompt:
    st.session_state.assistant_messages.append({"role": "user", "content": assistant_prompt})
    with st.chat_message("user"):
        st.write(assistant_prompt)
    with st.chat_message("assistant"):
        with st.status("Searching CRM", type="step"):
            # The assistant chat is intentionally portfolio-scoped. The
            # comparison controls below have their own account selector.
            assistant_result = answer_crm_question(assistant_prompt, account_id=None)
            tool_result = assistant_result["tool_result"]
            snapshot = database_snapshot()
        st.write(assistant_result["answer"])
        if tool_result.get("rows"):
            st.dataframe(tool_result["rows"], width="stretch", hide_index=True)
        if assistant_result["retrieved"]:
            with st.expander("Retrieved CRM evidence", expanded=False):
                for record in assistant_result["retrieved"]:
                    st.markdown(
                        f"**{record['source_type']} · {record.get('title', '')}** "
                        f"({record.get('score', 0):.2f})"
                    )
        if tool_result["status"] == "needs_clarification":
            st.info(
                "This answer came from the CRM schema tool; no model variant was run. "
                "Available structured entities: " + ", ".join(tool_result["available_entities"])
            )
        st.caption(
            f"Tool: {tool_result['tool']} · source: local CRM database · "
            f"{snapshot['accounts']} accounts, {snapshot['opportunities']} opportunities · "
            f"retrieved: {len(assistant_result['retrieved'])} records"
        )
    st.session_state.assistant_messages.append(
        {"role": "assistant", "content": assistant_result["answer"], "rows": tool_result.get("rows", [])}
    )

if "results_store" not in st.session_state:
    st.session_state.results_store: dict[tuple, dict] = {}

account_ids = list_account_ids()
account_options = ["(portfolio-wide -- all accounts)"] + account_ids
account_choice = st.selectbox("Account", account_options)
account_id = None if account_choice.startswith("(portfolio") else account_choice

if "query_text" not in st.session_state or st.session_state.get("_last_account") != account_choice:
    st.session_state.query_text = default_query(account_id)
    st.session_state._last_account = account_choice
    st.session_state.pop("example_query_select", None)  # options list differs per account

example_queries = load_example_queries().get(account_id, [])


def _on_example_change() -> None:
    selected = st.session_state.example_query_select
    if selected != "(write your own below)":
        st.session_state.query_text = selected


if example_queries:
    st.selectbox(
        f"Choose an example query ({len(example_queries)} available for this account)",
        options=["(write your own below)"] + example_queries,
        key="example_query_select",
        on_change=_on_example_change,
    )
else:
    st.caption("No example queries for this account -- write your own below.")

query = st.text_area("Query", key="query_text", height=80)
score_with_judge = st.checkbox("Score with Claude judge (slower, adds an API call per variant)", value=False)

st.markdown("**Run a variant for this account + query:**")
variant_names = list_variants()
run_cols = st.columns(len(variant_names) + 1)

judge_backend_holder: list = [None]  # constructed lazily, shared across buttons in this rerun


def _get_judge_backend():
    if score_with_judge and judge_backend_holder[0] is None:
        from common.llm_backend import get_teacher_backend

        judge_backend_holder[0] = get_teacher_backend()
    return judge_backend_holder[0]


for i, vname in enumerate(variant_names):
    if run_cols[i].button(f"Run {vname}", width="stretch"):
        with st.spinner(f"Running {vname}..."):
            result = run_one_variant(vname, query, account_id, score_with_judge, _get_judge_backend())
        st.session_state.results_store[(account_id, query, vname)] = result
        st.rerun()

if run_cols[-1].button("Run all variants", type="primary", width="stretch"):
    progress = st.progress(0.0, text="Starting...")
    for i, vname in enumerate(variant_names):
        progress.progress(i / len(variant_names), text=f"Running {vname}...")
        result = run_one_variant(vname, query, account_id, score_with_judge, _get_judge_backend())
        st.session_state.results_store[(account_id, query, vname)] = result
    progress.progress(1.0, text="Done.")
    st.rerun()

# --- Results: grouped by (account, query), variants shown side by side -----

st.divider()
st.subheader("Model comparison results")
st.caption(
    "These are saved results from explicitly running base, KD, or SFT. "
    "They are separate from the CRM assistant chat above and do not answer the latest chat question."
)

if st.session_state.results_store:
    if st.button("Clear all results"):
        st.session_state.results_store = {}
        st.rerun()

    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for (acc, q, var), result in st.session_state.results_store.items():
        grouped[(acc, q)][var] = result

    # Most recently touched groups first -- dict preserves insertion order,
    # and later writes to an existing key don't move it, so reverse gives a
    # reasonable "newest first" ordering without extra bookkeeping.
    for (acc, q) in reversed(list(grouped.keys())):
        variant_results = grouped[(acc, q)]
        acc_label = "portfolio-wide" if acc is None else acc
        with st.container(border=True):
            st.markdown(f"**{acc_label}** &mdash; {q}")
            cols = st.columns(len(variant_results))
            for col, (vname, result) in zip(cols, variant_results.items()):
                with col:
                    st.markdown(f"#### {vname}")
                    st.caption(labels.get(vname, vname))
                    render_response_card(result)
else:
    st.info("No results yet -- run a variant above.")
