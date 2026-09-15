from __future__ import annotations

import json

from common.crm_agent import retrieve
from common.agent.loop import run_agent_turn
from common.memory.budget import BudgetPolicy, prepare_prompt_context
from common.memory.store import SessionStore
from common.telemetry import TelemetrySink
from common.crm_retrieval import _documents, hybrid_search
from common.retrieval_reranker import rerank
from common.crm_store import database_snapshot, query_rows
from common.conversation_memory import resolve_question
from common.crm_tools import run_crm_tool
from common.crm_query import execute_question, schema_catalog
from common.rag_chat import _clean_model_answer
from common.tools.registry import CRM_TOOL_REGISTRY


def test_frappe_database_counts_match_source_contract():
    assert database_snapshot() == {
        "accounts": 4,
        "opportunities": 4,
        "contacts": 12,
        "leads": 4,
        "activities": 16,
        "items": 6,
        "opportunity_items": 17,
        "risk_factors": 8,
        "as_of": database_snapshot()["as_of"],
    }


def test_leads_are_answerable_from_the_new_schema():
    result = run_crm_tool("Which leads are associated with Acme Corp?")
    assert result["status"] == "ok"
    assert result["tool"] == "leads_all"
    assert result["rows"]
    assert result["rows"][0]["company_name"] == "Acme Corp"


def test_structured_contact_count_is_exact():
    result = run_crm_tool("How many contacts are there?")
    assert result["tool"] == "count_contacts"
    assert result["rows"] == [{"count": 12}]


def test_structured_email_count_uses_all_communications():
    result = run_crm_tool("How many emails are there in the system?")
    assert result["tool"] == "count_emails"
    assert result["rows"] == [{"count": 8}]
    assert result["answer"] == "There are 8 emails across the portfolio."


def test_account_ownership_is_separate_from_customer_contacts():
    result = run_crm_tool("Who is in charge of Acme Corp?")
    assert result["tool"] == "account_ownership"
    assert result["rows"][0]["account_owner"] == "Sanjay Kulkarni"
    assert result["rows"][0]["customer_contacts"][0]["contact_type"] == "champion"
    assert "owned by Sanjay Kulkarni" in result["answer"]


def test_model_sees_one_unified_crm_tool():
    assert CRM_TOOL_REGISTRY.names() == {"crm"}


def test_semantic_plan_executes_against_allowlisted_schema():
    result = execute_question(
        "Show the Acme stakeholders",
        plan={
            "entity": "contacts",
            "operation": "list",
            "account_id": "acme_corp",
            "requested_fields": ["name", "role", "contact_type"],
            "limit": 50,
            "include_evidence": False,
        },
    )
    assert result["status"] == "ok"
    assert result["plan"]["entity"] == "contacts"
    assert len(result["rows"]) == 3
    assert {row["contact_type"] for row in result["rows"]} == {"champion", "economic_buyer", "procurement"}


def test_invalid_semantic_field_is_rejected_without_sql_execution():
    result = execute_question(
        "Show CRM passwords",
        plan={"entity": "contacts", "operation": "list", "requested_fields": ["password"]},
    )
    assert result["status"] == "needs_clarification"
    assert result["rows"] == []
    assert "Unsupported CRM fields" in result["answer"]
    extra = execute_question("Show contacts", plan={"entity": "contacts", "operation": "list", "unexpected": True})
    assert extra["status"] == "needs_clarification"


def test_empty_question_is_a_typed_clarification():
    result = execute_question("")
    assert result["status"] == "needs_clarification"
    assert "question is required" in result["answer"]


def test_application_account_scope_cannot_be_widened_by_model_plan():
    result = execute_question(
        "Show contacts",
        account_scope="acme_corp",
        plan={
            "entity": "contacts",
            "operation": "list",
            "account_id": "globex",
            "requested_fields": ["name", "account_id"],
            "limit": 50,
        },
    )
    assert result["status"] == "ok"
    assert {row["account_id"] for row in result["rows"]} == {"acme_corp"}


def test_schema_catalog_is_the_single_planner_contract():
    catalog = schema_catalog()
    assert set(catalog) == {"accounts", "contacts", "leads", "opportunities", "communications"}
    assert "account_owner" in catalog["accounts"]["fields"]
    assert "account_id" in catalog["contacts"]["filters"]


def test_at_risk_account_question_returns_accounts_not_portfolio_count():
    result = run_crm_tool("Which accounts are at risk right now?")
    assert result["plan"]["entity"] == "accounts"
    assert result["plan"]["operation"] == "search"
    assert result["plan"]["filters"] == {"account_health": "at_risk"}
    assert [row["account"] for row in result["rows"]] == [
        "Acme Corp",
        "Delta Freight & Warehousing",
        "Initech Solutions",
    ]
    assert all("opportunities" in row and "risk_factors" in row for row in result["rows"])


def test_risk_records_are_recovered_if_small_model_claims_evidence_is_missing():
    result = run_crm_tool("Which accounts are at risk right now?")
    answer = _clean_model_answer("The evidence is insufficient to list them.", {"tool": "crm", **result})
    assert "Acme Corp" in answer
    assert "Delta Freight & Warehousing" in answer
    assert "Initech Solutions" in answer


def test_contact_coverage_uses_contact_names_not_shared_account_name():
    result = run_crm_tool("Which contacts are at Acme Corp?")
    answer = _clean_model_answer("Acme Corp has contacts.", {"tool": "crm", **result})
    for name in ("Rajesh Mehta", "Ananya Sharma", "Vikram Nair"):
        assert name in answer


def test_email_count_does_not_use_top_k_retrieval():
    tool, hits, _trace = retrieve("How many emails are there in the system?")
    assert tool["tool"] == "crm"
    assert tool["structured_tool"] == "count_emails"
    assert tool["rows"] == [{"count": 8}]
    assert hits == []


def test_exact_count_answer_cannot_be_overridden_by_model_text():
    answer = _clean_model_answer(
        "The retrieved documents suggest there are 3 emails.",
        {"tool": "crm", "structured_tool": "count_emails", "answer": "There are 8 emails across the portfolio."},
    )
    assert answer == "There are 8 emails across the portfolio."


def test_followup_memory_resolves_a_contact_list_request():
    history = [
        {"role": "user", "content": "How many contacts are there?"},
        {"role": "assistant", "content": "There are 12 contacts."},
    ]
    resolved, memory = resolve_question("Which are they?", history)
    assert memory.subject == "contacts"
    assert resolved.startswith("list contacts")
    tool, hits, trace = retrieve("Which are they?", conversation_history=history)
    assert tool["tool"] == "crm"
    assert tool["structured_tool"] == "contacts_all"
    assert len(tool["rows"]) == 12
    assert hits == []
    assert trace[0]["tool"] == "conversation_memory"


def test_retrieval_has_stable_document_ids_and_account_scope():
    hits = hybrid_search("What is the Acme discount request?", account_id="acme_corp", limit=5)
    assert hits
    assert all(hit["document_id"] for hit in hits)
    assert all(hit["account_id"] == "acme_corp" for hit in hits)
    assert any("Formal pricing request" in hit["title"] for hit in hits)


def test_retrieval_document_ids_are_unique_for_same_subjects():
    documents = _documents()
    ids = [document["document_id"] for document in documents]
    assert len(ids) == len(set(ids))


def test_retrieval_does_not_cross_account_scope():
    hits = hybrid_search("contract status", account_id="globex", limit=10)
    assert all(hit["account_id"] == "globex" for hit in hits)


def test_single_crm_tool_returns_structured_data_and_evidence():
    tool, hits, trace = retrieve(
        "What did the Acme procurement email say?",
        planner_handle=type(
            "CrmPlanner",
            (),
            {"generate": lambda self, system, prompt, max_tokens=220: '{"tool_calls":[{"name":"crm","arguments":{"question":"What did the Acme procurement email say?"}}]}'},
        )(),
    )
    assert tool["status"] in {"ok", "needs_clarification"}
    assert tool["tool"] == "crm"
    assert tool["structured_tool"]
    assert any(step["tool"] == "crm" for step in trace)
    assert hits


def test_reranker_preserves_candidates_and_exposes_components():
    candidates = hybrid_search("What is the Acme discount request?", account_id="acme_corp", limit=8)
    reranked = rerank("What is the Acme discount request?", candidates, account_id="acme_corp", limit=5)
    assert reranked
    assert all("rerank_score" in hit for hit in reranked)
    assert all(set(hit["rerank_components"]) == {"lexical", "entity", "source_type", "recency", "scope"} for hit in reranked)
    assert any("Formal pricing request" in hit["title"] for hit in reranked)


def test_opportunity_line_items_and_risk_factors_are_loaded():
    items = query_rows("SELECT COUNT(*) AS count FROM opportunity_items")[0]["count"]
    risks = query_rows("SELECT COUNT(*) AS count FROM opportunity_risk_factors")[0]["count"]
    assert items == 17
    assert risks == 8


def test_per_model_agent_loop_uses_same_handle_for_tool_decision_and_answer(tmp_path):
    class FakeHandle:
        def __init__(self):
            self.calls = []

        def generate(self, system_prompt, user_prompt, max_tokens=600):
            self.calls.append((system_prompt, user_prompt))
            if len(self.calls) == 1:
                return '{"tool_calls":[{"name":"crm","arguments":{"question":"How many contacts are there?"}}]}'
            return "There are 12 contacts across the portfolio."

    handle = FakeHandle()
    telemetry_path = tmp_path / "events.jsonl"
    turn = run_agent_turn(
        handle,
        question="How many contacts are there?",
        variant_name="base",
        account_id=None,
        conversation_history=[],
        system_prompt="system",
        build_prompt=lambda question, history, tool, hits: f"{question} {tool}",
        clean_answer=lambda raw, tool: raw.strip(),
        session_id="session-test",
        telemetry=TelemetrySink(telemetry_path),
    )
    assert len(handle.calls) == 2
    assert turn.model_used is True
    assert turn.tool_result["tool"] == "crm"
    assert turn.tool_result["structured_tool"] == "count_contacts"
    assert turn.answer.startswith("There are 12")
    assert "planner_user_prompt" in turn.debug_trace
    assert "final_prompt" in turn.debug_trace
    assert turn.debug_trace["validated_tool_calls"][0]["name"] == "crm"
    events = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    names = {event["event"] for event in events}
    assert {"crm.turn.started", "model.tool_decision", "tool.execute", "budget.checked", "model.final_answer", "crm.turn.completed"} <= names
    started = next(event for event in events if event["event"] == "crm.turn.started")
    assert "question" not in started["payload"]
    assert "question_hash" in started["payload"]


def test_prompt_budget_reports_history_compaction():
    history = [{"role": role, "content": "long CRM context " * 500} for role in ("user", "assistant", "user", "assistant")]
    kept, limits = prepare_prompt_context(
        "What is the answer?",
        history,
        lambda q, h, tool, hits: q + " " + " ".join(message["content"] for message in h),
        {},
        [],
        BudgetPolicy(context_window=300, reserved_output=50),
    )
    assert len(kept) < len(history)
    assert limits["history_turns_dropped"] > 0
    assert limits["status"] in {"history_compacted", "context_limit_reached"}
    assert limits["history_summary_created"] is True
    assert any("Earlier conversation summary" in message["content"] for message in kept)


def test_session_store_round_trips_conversation(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    conversation = {
        "id": "session-test",
        "title": "Contacts",
        "created_at": "2026-09-15T10:00:00",
        "messages": [
            {"role": "user", "content": "How many contacts?", "ts": "2026-09-15T10:00:01"},
            {"role": "assistant", "content": "There are 12.", "results": {"base": {"answer": "There are 12."}}, "ts": "2026-09-15T10:00:02"},
        ],
    }
    store.save_conversation(conversation, "acme_corp")
    loaded = store.load_conversation("session-test")
    assert loaded is not None
    assert loaded["title"] == "Contacts"
    assert loaded["account_scope"] == "acme_corp"
    assert loaded["messages"][1]["results"]["base"]["answer"] == "There are 12."
    state = store.limit_state("session-test", max_tokens=1)
    assert state["session_limit_reached"] is True
    assert state["status"] == "session_limit_reached"
