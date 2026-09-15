
# Modular RAG Agent Architecture Proposal

## Executive recommendation

Restructure the current Streamlit assistant into a per-model agent runtime with six independent boundaries:

1. **Agent loop** — the selected local model understands the user, chooses a tool or answers, receives tool results, and answers naturally.
2. **Retrieval pipeline** — candidate retrieval, reranking, evidence selection, and citations are separate components.
3. **Tool layer** — typed, read-only CRM tools with validated arguments and bounded results.
4. **Session memory** — durable conversation state, summaries, token budgets, and session limits.
5. **Observability** — LangSmith traces and local audit events for every turn.
6. **Evaluation** — retrieval and answer evaluation with MRR, delta-MRR, nDCG, tool accuracy, and groundedness.

The local Base, KD, and SFT models remain the decision-makers. There is no external Claude runtime orchestrator. Claude remains a teacher/evaluator only.

## Current architecture assessment

| Area | Current state | Architectural concern |
|---|---|---|
| UI | Streamlit state stores conversations | State is lost when the process/session is reset |
| Agent runtime | 'common/rag_chat.py' | Chat, routing, retrieval, synthesis, and output repair are coupled |
| Tool planning | 'common/crm_agent.py' | Planner output, fallback routing, memory resolution, and retrieval are mixed |
| Retrieval | Lexical BM25-style search | No independent reranker or retrieval benchmark |
| Structured data | Schema-driven service in 'common/crm_query.py' | Semantic plan and allow-listed fields are now explicit |
| Memory | 'common/conversation_memory.py' | Heuristic state, no durable store or token budget |
| Context limits | No preflight budget manager | Prompt can fail or silently lose useful history |
| Telemetry | None | Cannot trace tool choice, evidence, latency, or model failures |
| Evaluation | NextBestAction evaluation | Does not measure conversational tool use or retrieval quality |
| Model contract | Current adapters target NextBestAction JSON | Existing adapters are not trained for natural tool-use conversations |

The Ripwire architecture scan found no dependency cycles, but identified 'common/rag_chat.py', 'common/crm_agent.py', 'common/context.py', and 'common/crm_store.py' as high-leverage areas. They should be treated as integration boundaries rather than expanded further.

## Scope decision for this POC

The current corpus is four accounts, four opportunities, twelve contacts, and sixteen activities. That scale does not justify building a production-sized retrieval platform before measuring whether the current lexical ranking is wrong.

The implementation order is therefore deliberately narrow:

| Priority | Build now | Defer until the metric justifies it |
|---|---|---|
| P0 | Stable document IDs, golden retrieval queries, baseline Recall@k/MRR, typed tools, per-model agent loop, token budgets | — |
| P1 | Deterministic reranking and delta-MRR comparison | Cross-encoder reranker if the deterministic reranker improves the benchmark |
| P2 | Durable sessions, local event logging, and the first evaluation hooks | LangSmith, semantic embeddings, production CRM connector |

The reranker is not a success criterion by itself. If baseline retrieval has near-perfect Recall@k and MRR on the golden set, keep the simpler retriever and record that result. This avoids ceremony and protects the POC from infrastructure work that does not improve answer quality.

### First measured retrieval decision

The first six-query golden benchmark was run on 2026-09-15. The current lexical retriever remains active because the initial deterministic reranker reduced ranking quality:

| Metric | Baseline | Deterministic reranker | Change |
|---|---:|---:|---:|
| Recall@5 | 1.0000 | 1.0000 | 0.0000 |
| MRR | 0.8333 | 0.8056 | -0.0278 |
| nDCG@5 | 0.8357 | 0.7731 | -0.0626 |

The reranker implementation remains available in 'common/retrieval_reranker.py' for further experiments, but it is not wired into the production chat path. The result is recorded in 'results/retrieval_comparison.json'. No embedding or cross-encoder work is justified by this benchmark yet.

### Phase 1 implementation status

The typed contract and per-model loop boundary is now implemented:

| Component | Location | Status |
|---|---|---|
| Agent contracts | 'common/agent/contracts.py' | Implemented |
| Per-model agent loop | 'common/agent/loop.py' | Implemented and wired through 'rag_chat.py' |
| Tool registry | 'common/tools/registry.py' | One model-visible read-only 'crm' tool; structured lookup and evidence retrieval are internal |
| CRM query contract | 'common/crm_query.py' | Implemented: schema catalog, validated semantic plans, parameterized read-only executor |
| Compatibility API | 'common/crm_tools.py' | Thin adapter only; no SQL or question-specific routing |
| UI compatibility | 'common/agent/AgentTurn.as_ui_result()' | Implemented |
| Regression coverage | 'tests/test_crm_regression.py' | 21 tests passing |

The loop receives one selected local model handle, performs that model's tool decision, executes validated read-only tools, and asks the same model for the final answer. The Streamlit adapter still preserves the one-GGUF-at-a-time VRAM policy.

The public tool boundary is intentionally one tool: 'crm(question, plan)'. The model may provide a semantic plan containing an entity, operation, filters, and requested fields. The application validates that plan against the schema catalog and executes parameterized read-only SQL. It returns one evidence package containing the authoritative structured result plus bounded retrieved evidence when needed. The compatibility path can compile an old free-form question, but new model calls should use the plan contract.

### Phase 3 implementation status

The first session and context-budget slice is now implemented:

| Component | Location | Status |
|---|---|---|
| Durable session store | 'common/memory/store.py' | Separate 'data/sessions.sqlite3' database |
| Session reload | 'ui/app.py' | Conversations reload when Streamlit starts |
| Context budget | 'common/memory/budget.py' | Estimates prompt size, summarizes older history, then trims only if needed |
| Context warnings | 'ui/app.py' | Shows history compaction and hard-limit messages |
| Session limits | 'common/memory/store.py' | Tracks message tokens and blocks sessions at the configured limit |
| Regression coverage | 'tests/test_crm_regression.py' | 21 tests passing, including memory and limit coverage |

The current token counter uses a conservative four-characters-per-token estimate because the repository does not have a local Hugging Face tokenizer cache. The budget interface accepts an exact tokenizer later without changing the session or agent contracts.

### Phase 4 implementation status

The first local telemetry slice is now implemented:

| Component | Location | Status |
|---|---|---|
| Local event sink | 'common/telemetry.py' | JSONL under 'data/telemetry/events.jsonl', enabled by default |
| Turn tracing | 'common/agent/loop.py' | Emits turn start/completion, tool decision, evidence, budget, and final-answer events |
| Tool/retrieval tracing | 'common/crm_agent.py' | Emits validated tool execution and lexical retrieval events |
| Exact model trace | 'common/agent/contracts.py', 'ui/app.py' | Collapsed UI panel shows the exact planner/final prompts, tool calls, evidence, and raw outputs for each variant |
| Privacy controls | 'common/telemetry.py' | Text is hashed and length-only by default; raw prompt logging requires explicit configuration |
| Retrieval workbook | 'report/build_retrieval_report.mjs' | Creates Summary, Per-query, and Definitions sheets from the benchmark JSON |
| Regression coverage | 'tests/test_crm_regression.py' | Verifies event names, default redaction, and partial planner fallback |

The local event stream is the POC source of truth. It contains stable session/turn IDs, model variant, tool names, document IDs, counts, status, and latency, so one turn can be reconstructed without a remote service. LangSmith remains an optional exporter for a later phase and is not required for local development.

The first workbook is generated at 'outputs/phase4-retrieval-20260915/retrieval-evaluation.xlsx'. It reports the baseline and deterministic-reranker metrics side by side, shows per-query rank movement, and defines Recall, MRR, nDCG, and the benchmark scope. The workbook records the current decision to keep the lexical baseline because reranking reduced MRR and nDCG on the six-query set.

### Future CRM target is Frappe/ERPNext, not a generic connector

A real Frappe/ERPNext site (`magnaerp.local`, with the standard Frappe and ERPNext apps) already runs on this machine. `mock_crm/`'s flat schema (`accounts`, `opportunities`, `contacts`, `activities`) does not match it: ERPNext models `Lead` and `Opportunity` and `Customer` as distinct doctypes with a conversion lifecycle, plus `Communication`, `Quotation`, `Sales Order`, and per-doctype custom fields. It also has a real authorization model — roles, user permissions, and per-document sharing — that a bolt-on `account_id` scope filter cannot express.

This does not change the POC plan: mock_crm stays the training/evaluation corpus, and no Frappe integration work starts now. It means that wherever this document says "production CRM connector," the concrete target is the Frappe REST/RPC API against this site, and the tool layer's authorization design (when that phase starts) should call the Frappe API as the acting user and let Frappe enforce permissions, rather than inventing a generic scope model to retrofit later.

## Roadmap authority

'CRM_SLM_RAG_AGENT_ARCHITECTURE.md' is the authoritative implementation roadmap. 'CONVERSATIONAL_AGENT_PLAN.md' defines the training data and model behavior requirements and is subordinate to this roadmap for sequencing. The tool contract and per-model agent loop are designed once and then reused by both documents; they must not be implemented twice.

## Target architecture

~~~mermaid
flowchart TB
    UI[Streamlit UI]
    SESSION[Session service]
    LOOP[Per-model agent loop]
    MEMORY[Memory manager]
    BUDGET[Context budget manager]
    TOOLS[Typed CRM tool registry]
    RETRIEVE[Retrieval pipeline]
    CANDIDATES[Candidate retriever]
    RERANK[Reranker]
    EVIDENCE[Evidence pack and citations]
    MODEL[Selected local Base/KD/SFT model]
    TRACE[LangSmith and local telemetry]
    EVAL[Offline eval suite]

    UI --> SESSION
    SESSION --> LOOP
    LOOP --> MEMORY
    LOOP --> BUDGET
    LOOP --> MODEL
    MODEL -->|validated tool call| TOOLS
    MODEL -->|search request| RETRIEVE
    TOOLS --> EVIDENCE
    RETRIEVE --> CANDIDATES --> RERANK --> EVIDENCE
    EVIDENCE --> LOOP
    LOOP --> UI
    SESSION --> TRACE
    LOOP --> TRACE
    RETRIEVE --> TRACE
    TOOLS --> TRACE
    TRACE --> EVAL
~~~

The important boundary is 'agent_loop.py'. It runs once per selected model, so Base, KD, and SFT each independently understand the user and perform tool use.

## Proposed module structure

~~~text
common/
├── agent/
│   ├── contracts.py          # ToolCall, ToolResult, AgentAnswer, AgentTurn
│   ├── agent_loop.py         # One local model's tool-use loop
│   ├── tool_call_parser.py   # Grammar/schema validation, no business logic
│   ├── prompts.py            # System and tool-use prompt templates
│   └── errors.py             # Typed user-safe errors
├── tools/
│   ├── registry.py           # Tool metadata and registration
│   ├── crm_read.py           # Accounts, contacts, opportunities, activities
│   ├── crm_analytics.py      # Counts, pipeline, risk and forecast
│   └── validators.py         # Scope, limits, argument validation
├── retrieval/
│   ├── models.py             # Document, Candidate, Evidence, Citation
│   ├── corpus.py             # Build/index CRM documents
│   ├── lexical.py            # BM25/SQLite FTS candidate retrieval
│   ├── semantic.py           # Optional embedding retriever
│   ├── reranker.py           # Cross-encoder or deterministic reranker
│   ├── pipeline.py           # Retrieve -> rerank -> evidence pack
│   └── metrics.py             # Recall, MRR, delta-MRR, nDCG
├── memory/
│   ├── models.py             # Session, Message, Summary, MemoryState
│   ├── store.py              # SQLite-backed session persistence
│   ├── summarizer.py         # Model-based or extractive summaries
│   ├── resolver.py           # Follow-up/entity resolution
│   └── budget.py             # Token accounting and context decisions
├── observability/
│   ├── events.py              # Stable event names and payloads
│   ├── langsmith.py          # Optional LangSmith adapter
│   └── redaction.py          # CRM-sensitive field redaction
└── config.py                 # Environment and feature configuration

ui/
├── app.py                    # Shell, session selection, model comparison
└── components/
    ├── answer.py              # Natural answer and citations
    ├── tool_trace.py         # Optional technical trace
    ├── limits.py              # Context/session warning cards
    └── comparison.py         # Variant comparison view

eval/
├── retrieval_eval.py          # Offline retrieval and reranking metrics
├── agent_eval.py              # Tool and answer behavior
├── datasets.py                # Golden queries and relevance labels
└── reports.py                 # JSON and Excel summaries
~~~

The existing modules should be migrated behind these interfaces. Do not delete the old NextBestAction path while the new path is being validated.

## Stable contracts

### Agent contracts

~~~python
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str

@dataclass
class ToolResult:
    call_id: str
    name: str
    rows: list[dict]
    text: str
    citations: list[str]
    truncated: bool

@dataclass
class AgentTurn:
    session_id: str
    user_text: str
    resolved_text: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    answer: str
    citations: list[str]
    usage: "TokenUsage"
    limits: "LimitState"
    trace_id: str | None
~~~

The UI should consume 'AgentTurn'. It should not know how the model chose a tool, how retrieval ranked documents, or how the database was queried.

### Retrieval contracts

~~~python
@dataclass
class RetrievalQuery:
    text: str
    account_id: str | None
    filters: dict[str, str]
    top_k_candidates: int = 30
    top_k_evidence: int = 8

@dataclass
class Candidate:
    document_id: str
    source_type: str
    account_id: str | None
    text: str
    retrieval_score: float
    metadata: dict[str, Any]

@dataclass
class EvidencePack:
    query: RetrievalQuery
    candidates: list[Candidate]
    citations: list[str]
    token_count: int
    truncated: bool
    retrieval_trace: dict[str, Any]
~~~

This lets the project replace lexical retrieval with embeddings or a vector database without changing the agent loop.

## Retrieval design

### Retrieval stages

~~~mermaid
flowchart LR
    Q[Resolved user question]
    FILTER[Scope and metadata filters]
    LEX[Lexical candidate retrieval]
    SEM[Optional semantic retrieval]
    UNION[Candidate union]
    DEDUP[Document deduplication]
    RERANK[Cross-encoder or deterministic reranker]
    BUDGET[Token-aware evidence selection]
    PACK[Evidence pack with citations]
    Q --> FILTER
    FILTER --> LEX
    FILTER --> SEM
    LEX --> UNION
    SEM --> UNION
    UNION --> DEDUP --> RERANK --> BUDGET --> PACK
~~~

Recommended first implementation:

| Stage | First implementation | Later option |
|---|---|---|
| Corpus | SQLite rows plus activity documents | Production CRM connector |
| Candidate retrieval | BM25-style local scorer or SQLite FTS5 | Dense embeddings |
| Candidate count | 30 | 50–100 for larger CRM |
| Deduplication | Stable document ID | Parent/child evidence grouping |
| Reranking | Weighted deterministic score plus entity match | Small cross-encoder on CPU |
| Final evidence | Top 8 within token budget | Query-adaptive evidence count |
| Citations | Source type, title, date, account | CRM URL and record ID |

Any embedding model is CPU-only in this POC. The 4 GB GPU is reserved for the selected local GGUF model. Semantic retrieval is deferred unless the baseline benchmark shows a meaningful lexical-retrieval gap.

### Reranking score

Use a transparent score before introducing a heavier model:

~~~text
final_score =
    0.45 * lexical_score
  + 0.20 * entity_match
  + 0.15 * source_type_match
  + 0.10 * recency_score
  + 0.10 * account_scope_match
~~~

Every component should be logged separately. This makes bad rankings diagnosable.

These weights are initial hypotheses, not ground truth. Tune them against the golden retrieval set by grid search or a small coordinate search, and keep the baseline weights in the report so the measured gain is reproducible.

For a later cross-encoder, keep the same 'Reranker' interface. The cross-encoder receives the question and each candidate and returns a relevance score. It can run on CPU to preserve the local GPU for the selected LLM.

## MRR, delta-MRR, and related metrics

'MRR' is mean reciprocal rank. For each query, find the rank of the first relevant document:

~~~text
RR(query) = 1 / rank_of_first_relevant_document
MRR = mean(RR(query))
~~~

Report:

| Metric | Meaning |
|---|---|
| Recall@k | Whether any relevant document appears in the first k results |
| Precision@k | Fraction of the first k results that are relevant |
| MRR@k | How early the first relevant document appears |
| nDCG@k | Ranking quality when documents have graded relevance |
| delta-MRR | Reranked MRR minus baseline MRR |
| Tool accuracy | Correct tool selected and arguments valid |
| Evidence coverage | Required ground-truth facts present in evidence |
| Answer groundedness | Claims supported by evidence |

'DMRR' is not a single universally defined standard metric. To avoid ambiguity, use 'delta-MRR' in the report and document the formula:

~~~text
delta-MRR = MRR(reranked) - MRR(candidate_retriever)
~~~

If the team specifically wants a metric called DMRR, define it in the workbook before using it. Never present an undefined DMRR number to management.

### Golden retrieval dataset

Create 'data/retrieval_eval.jsonl':

~~~json
{
  "query_id": "retrieval_001",
  "question": "Which accounts are at risk?",
  "account_id": null,
  "relevant_documents": [
    {"document_id": "account:delta", "relevance": 2},
    {"document_id": "opportunity:initech_001", "relevance": 2}
  ],
  "required_entities": ["Delta Freight & Warehousing", "Initech Solutions"]
}
~~~

Use graded relevance:

- 0 = irrelevant
- 1 = related but insufficient
- 2 = directly supports the answer

Hold out query templates, not only rows. Otherwise the metric measures memorization instead of retrieval quality.

## Tool architecture

Expose one model-visible tool and keep internal routing behind that boundary:

~~~mermaid
flowchart LR
    CALL[Validated ToolCall]
    REG[Tool registry]
    VALIDATE[Argument and scope validation]
    CRM[crm(question, plan)]
    PLAN[Semantic QueryPlan validator]
    EXEC[Allow-listed parameterized executor]
    STRUCTURED[Structured CRM result]
    EVIDENCE[Bounded evidence retrieval]
    LIMIT[Row, field, and token limits]
    RESULT[Unified ToolResult with citations]
    CALL --> REG --> VALIDATE --> CRM
    CRM --> PLAN --> EXEC --> STRUCTURED --> LIMIT --> RESULT
    CRM --> EVIDENCE --> LIMIT
~~~

Every tool must declare:

| Field | Example |
|---|---|
| Name | 'crm' |
| Arguments | 'question: string; plan: entity, operation, filters, fields' |
| Scope | Account or portfolio |
| Max rows | Bounded internally by query type |
| Read/write | Read-only |
| Result | Structured data plus relevant evidence |
| Citation format | Stable CRM/document IDs |
| Failure behavior | Typed clarification or tool error |

The model should not be allowed to write arbitrary SQL. It only selects a registered tool and typed arguments.

## Session-based memory

### Data model

Use a dedicated session database or dedicated tables in a separate application SQLite file. Keep it separate from CRM source data.

~~~mermaid
erDiagram
    SESSIONS ||--o{ MESSAGES : contains
    SESSIONS ||--o{ MEMORY_SUMMARIES : has
    SESSIONS ||--o{ TURN_EVENTS : records

    SESSIONS {
      text session_id PK
      text user_id
      text account_scope
      text model_variant
      text status
      text created_at
      text updated_at
      integer input_tokens
      integer output_tokens
    }

    MESSAGES {
      integer message_id PK
      text session_id FK
      text role
      text content
      text tool_name
      text created_at
      integer token_count
    }

    MEMORY_SUMMARIES {
      integer summary_id PK
      text session_id FK
      text summary
      text covered_until
      integer token_count
      text created_at
    }

    TURN_EVENTS {
      integer event_id PK
      text session_id FK
      text trace_id
      text event_type
      text payload_json
      text created_at
    }
~~~

### Memory policy

| State | Action |
|---|---|
| Recent turns fit budget | Send recent user/assistant/tool turns |
| History is getting large | Summarize older turns and retain recent turns verbatim |
| Account/entity facts are stable | Store them as structured memory, not only prose |
| User starts a new chat | Create a new session ID |
| Streamlit process restarts | Reload session from SQLite |
| User requests deletion | Delete messages, summaries, and telemetry references |
| Session idle timeout | Mark session closed after configured TTL |

The session ID must be independent of the model variant. This lets the user compare Base, KD, and SFT against the same conversation state.

## Context and session limits

### Token budget

The budget manager should count tokens using the exact tokenizer used by the selected model.

~~~mermaid
flowchart TB
    TURN[New turn]
    COUNT[Count system + memory + tools + evidence + question]
    SOFT{Above soft limit?}
    SUM[Summarize older history]
    TRIM[Reduce evidence and tool rows]
    HARD{Above hard limit?}
    STOP[Return user-facing limit message]
    GENERATE[Generate with reserved output budget]

    TURN --> COUNT --> SOFT
    SOFT -->|no| GENERATE
    SOFT -->|yes| SUM --> TRIM --> COUNT
    GENERATE --> HARD
    HARD -->|no| GENERATE
    HARD -->|yes| STOP
~~~

Recommended initial settings for the current 5120-token model context:

| Budget item | Suggested value |
|---|---:|
| Total model context | 5120 |
| System and tool definitions | 700 |
| Memory | 700 |
| Retrieved evidence | 1800 |
| Current question | 300 |
| Reserved answer | 1200 |
| Safety margin | 420 |

The exact values should be measured per model and stored in configuration.

### User-facing messages

The UI should distinguish these cases:

| Condition | User message |
|---|---|
| Context approaching limit | “This conversation is getting long. I’ll summarize earlier messages to continue.” |
| Evidence was reduced | “I used the most relevant CRM records to stay within the model context.” |
| Hard context limit | “This request is too large for the current model context. Please narrow the account, date range, or question.” |
| Session token budget near limit | “This session is nearing its usage limit.” |
| Session token budget reached | “This session limit has been reached. Start a new session or increase the configured budget.” |
| Tool result truncated | “The CRM returned more records than can be shown. Narrow the scope to see all records.” |
| Model unavailable | “The selected model is unavailable. Choose another registered variant.” |

The UI should never silently drop history, evidence, or records.

## LangSmith telemetry

LangSmith should be optional and disabled by default unless configured.

### Trace tree

~~~mermaid
flowchart TB
    ROOT[crm.turn]
    MEM[memory.resolve]
    PLAN[model.tool_decision]
    TOOL[tool.execute]
    RET[retrieval.run]
    RERANK[retrieval.rerank]
    PACK[evidence.pack]
    ANSWER[model.final_answer]
    LIMIT[budget.check]
    ROOT --> MEM
    ROOT --> PLAN
    PLAN --> TOOL
    PLAN --> RET
    RET --> RERANK --> PACK
    TOOL --> PACK
    PACK --> LIMIT --> ANSWER
~~~

### Trace fields

| Span | Required fields |
|---|---|
| 'crm.turn' | session_id, turn_id, variant, account_scope, trace_id |
| 'memory.resolve' | history tokens, summary tokens, resolved subject |
| 'model.tool_decision' | model, latency, input/output tokens, parse status |
| 'tool.execute' | tool name, sanitized arguments, row count, duration |
| 'retrieval.run' | query, candidate count, filters, duration |
| 'retrieval.rerank' | candidate scores, selected IDs, duration |
| 'evidence.pack' | selected citations, token count, truncated flag |
| 'model.final_answer' | model, latency, output tokens, stop reason |
| 'budget.check' | used, limit, action taken |

### Privacy rules

- Never log API keys.
- Redact email bodies and personal contact details by default.
- Log stable CRM IDs and document types where possible.
- Make raw prompt logging configurable.
- Keep LangSmith project names environment-specific.
- Add a local JSONL audit sink so the system remains diagnosable when LangSmith is disabled.

Suggested environment variables:

~~~text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=crm-slm-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
CRM_TELEMETRY_REDACT=true
CRM_TELEMETRY_LOG_PROMPTS=false
~~~

## Evaluation plan

### Retrieval evaluation

Run the same golden queries through:

1. Current lexical candidate retriever.
2. Candidate retriever plus deterministic reranker.
3. Candidate retriever plus cross-encoder reranker, when enabled.

Report:

| Category | Metrics |
|---|---|
| Retrieval | Recall@5, Recall@10, MRR@5, nDCG@5 |
| Reranking gain | delta-MRR, delta-nDCG, rank movement |
| Evidence | required-entity coverage, citation precision |
| Cost | latency, candidate count, evidence tokens |

### Agent evaluation

| Category | Metrics |
|---|---|
| Understanding | intent accuracy, follow-up resolution |
| Tool use | tool selection accuracy, argument accuracy, unnecessary-call rate |
| Answer | correctness, completeness, groundedness, naturalness |
| Limits | context warning accuracy, truncation disclosure |
| Runtime | latency, tokens/sec, VRAM, failures |
| Session | memory retention across turns, reload persistence |

Keep the old NextBestAction evaluation separate. Do not compare its guardrail pass rate directly with natural-language agent quality.

## Revised implementation plan

### Phase 0 — Freeze and baseline

- Leave the existing NextBestAction pipeline unchanged.
- Record current Streamlit behavior and model outputs.
- Create stable document IDs for every account, opportunity, contact, and activity.
- Create the golden retrieval set with graded relevance labels.

**Gate:** baseline Recall@5, Recall@10, MRR@5, and nDCG@5 are available.

### Phase 1 — Typed contracts and per-model agent loop

- Add typed tool, evidence, memory, budget, and turn contracts.
- Split tool registration, semantic planning, and SQL execution.
- Keep the CRM schema catalog as the single planner contract.
- Execute only allow-listed parameterized queries; never expose raw SQL to a model.
- Build one per-model agent loop that selects a tool or answers naturally.
- Validate tool calls with a schema/grammar boundary.
- Keep an adapter so the current UI can call the new loop.

**Gate:** Base, KD, and SFT independently handle direct questions, casual turns, and follow-ups using the same tool contract.

### Phase 2 — Retrieval quality before retrieval infrastructure

- Put the current lexical retriever behind the retrieval interface.
- Measure baseline retrieval metrics on the golden set.
- Add evidence packs, stable citations, and token-aware selection.
- Add a transparent deterministic reranker only if the baseline has ranking errors.

**Gate:** compare candidate retrieval against reranking. If reranking does not improve MRR, nDCG, or required-entity coverage, stop here and keep the simpler design.

### Phase 3 — Memory and limits

- Add a separate session SQLite database.
- Persist sessions, messages, summaries, and turn events.
- Add tokenizer-based context budgeting.
- Summarize older history before hard truncation.
- Display context reduction, hard context, tool-result truncation, and session-budget messages.

**Gate:** a session survives a Streamlit restart and never silently loses history or evidence.

### Phase 4 — Local telemetry and evaluation

- Emit local JSONL events first.
- Trace retrieval, reranking, tools, memory, budgets, and model calls.
- Add tool-selection, argument, groundedness, completeness, and latency metrics.
- Build the Excel report for baseline versus reranked retrieval.

**Gate:** one complete turn can be reconstructed without LangSmith.

### Phase 5 — Optional production additions

- Add optional CPU-only semantic retrieval if the benchmark justifies it.
- Add optional CPU-only cross-encoder reranking if deterministic reranking is insufficient.
- Add LangSmith with redaction and environment-specific projects.
- Add export/delete controls and production CRM connectors.

**Gate:** each addition must show measurable benefit or a clear operational requirement.

## Success criteria

The redesigned system is ready for the POC when:

- each local model independently understands casual and CRM queries
- each model can select and call typed tools
- retrieval has measured baseline MRR and Recall@k
- reranking is used only if it improves the golden-set metrics; otherwise the report records that the simpler baseline won
- every answer has evidence citations
- structured tool output is not silently omitted
- the POC either persists sessions across Streamlit restarts or clearly labels persistence as deferred; the production gate requires persistence
- context reduction is announced to the user
- context and session limits produce clear messages
- local telemetry can reconstruct a turn; LangSmith is an optional production adapter
- Base, KD, and SFT are evaluated with the same evidence and tool contracts
- the old NextBestAction pipeline remains reproducible and isolated

## Principal architect decision

Do not keep extending 'common/rag_chat.py' with more routing rules. Make the agent loop, retrieval pipeline, tools, memory, budget manager, telemetry, and evaluation separate modules with typed contracts. This gives the project a stable foundation for changing models, rerankers, UI, CRM connectors, and memory policy independently.
