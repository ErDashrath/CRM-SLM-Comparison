# Conversational CRM Agent Training Plan

> **Roadmap authority:** [CRM_SLM_RAG_AGENT_ARCHITECTURE.md](CRM_SLM_RAG_AGENT_ARCHITECTURE.md) is the authoritative implementation roadmap. This document defines the conversational training data and model behavior. It does not define a second tool registry, retrieval pipeline, memory system, or implementation sequence.

## Goal

Train local CRM models that remain natural conversational assistants after training. Each model must independently:

1. Understand casual messages and CRM questions.
2. Use conversation history for follow-up questions.
3. Decide whether a CRM tool is needed.
4. Produce a validated tool call when needed.
5. Read the tool result.
6. Answer naturally without exposing an internal JSON schema.

Claude is used only as the teacher and dataset generator. The local Base, KD, and SFT variants make the runtime decisions.

## Current problem

The existing KD and SFT datasets teach the model to emit a NextBestAction object:

- action_type
- target_object
- payload
- rationale
- risk_flags
- confidence

That format is useful for the old evaluation pipeline, but it is the wrong training target for a conversational CRM agent. The old pipeline will remain unchanged for comparison and research.

## Target runtime flow

```
User message
  -> local model
  -> final answer OR validated CRM tool call
  -> local CRM tool executes
  -> tool result is appended to conversation
  -> same local model writes a natural answer
```

The model receives the tool definitions and recent conversation history. No keyword router, greeting regex, Claude runtime router, or hardcoded intent patches will be used.

## Tool contract

Expose one read-only model tool with a stable schema:

- `crm(question, plan)`

The plan is semantic rather than SQL:

```json
{
  "entity": "contacts",
  "operation": "list",
  "account_id": "acme_corp",
  "requested_fields": ["name", "role", "contact_type"]
}
```

The application validates the plan against the schema catalog, executes the
allow-listed read-only query, and when useful adds bounded email, transcript,
account, or opportunity evidence to the same result. The model never chooses
between separate query and search tools and never writes SQL.

Every tool must have:

- JSON name
- typed arguments
- argument validation
- bounded result size
- deterministic execution
- human-readable result
- no write operation in the first POC

The model's internal tool call is separate from its final answer.

## Training format

Use conversational message examples, not NextBestAction targets.

Example tool-use record:

```json
{
  "messages": [
    {"role": "user", "content": "How many contacts are there?"},
    {
      "role": "assistant",
      "tool_call": {
        "name": "crm",
        "arguments": {"question": "How many contacts are there?"}
      }
    },
    {
      "role": "tool",
      "name": "crm",
      "content": "[CRM rows]"
    },
    {
      "role": "assistant",
      "content": "There are 12 contacts across the portfolio."
    }
  ]
}
```

Also include:

- greetings: `hello`, `hola`, spelling variations, thanks
- direct CRM answers that require no tool
- account-specific questions
- portfolio questions
- follow-ups such as “which are they?”
- pronouns such as “what about that deal?”
- clarification cases
- insufficient-evidence cases
- risk, contact, opportunity, pipeline, email, transcript, and policy questions
- answers that enumerate every returned record
- answers that distinguish facts from recommendations

## Dataset design

Generate two new datasets and keep the old files untouched:

- `data/chat_tool_kd_train.jsonl`
- `data/chat_tool_sft_train.jsonl`

Recommended first dataset:

- 500–800 total conversations
- 60% CRM tool-use conversations
- 20% follow-up and multi-turn conversations
- 10% casual/general conversation
- 10% clarification and insufficient-evidence cases
- all four accounts represented evenly
- portfolio-wide questions included
- tool argument combinations varied

KD data comes from Claude teacher demonstrations.

SFT data is a curated subset with:

- diverse intents
- correct tool choice
- valid arguments
- complete answers
- no fabricated facts
- no leaked hidden test questions

## Claude teacher generation

Use the existing Claude API key only during dataset creation.

Claude should receive:

- tool definitions
- CRM context
- conversation scenario
- output schema for the training record

Claude must generate:

1. User turn
2. Optional prior conversation
3. Tool call
4. Deterministic tool result
5. Natural final answer

The generated record must be validated by executing the tool call against the local CRM before it enters the training set.

Reject records when:

- tool name is invalid
- arguments do not validate
- answer contains facts absent from the tool result/context
- required records are omitted
- the answer exposes internal training JSON
- the conversation is repetitive or contradictory

## Local training

Keep Qwen3-1.7B-Q4 as the current hardware-compatible model.

Training approach:

- LoRA or QLoRA
- compact CRM context
- full final answer retained
- tool-call examples represented with a stable special format
- assistant loss applied to tool-call and final-answer turns
- tool-result turns excluded from loss
- train for 2–3 epochs initially
- save adapters separately:
  - `adapters/chat_kd`
  - `adapters/chat_sft`
- merge and quantize each adapter separately
- register them as new variants instead of overwriting the old variants

The old `adapters/kd` and `adapters/sft` remain available for the NextBestAction experiment.

## Runtime tool calling

The local model should produce one of two outputs:

1. A final natural-language answer.
2. A structured tool call.

Use constrained decoding or a llama.cpp grammar for the tool-call turn. Do not rely on regex repair as the normal path.

Runtime loop:

- Load one selected variant.
- Send system prompt, recent conversation, and tool definitions.
- Generate either final answer or tool call.
- Validate the tool call.
- Execute the tool.
- Append the tool result.
- Generate the final natural-language response.
- Allow at most two tool rounds.
- If the tool call is invalid, ask the model once for a corrected call.
- If still invalid, explain that the CRM request could not be understood.

## Conversation memory

Store structured session state:

- recent user and assistant messages
- active account scope
- last detected subject
- last tool result
- unresolved clarification
- conversation summary when history becomes long

The model receives the actual recent conversation. The application may maintain a compact summary for context length, but it must not replace the conversation with keyword rules.

## Streamlit changes

The UI should show:

- natural final answer
- optional “CRM data used” section
- optional tool trace for debugging
- selected model variant
- conversation history
- comparison across Base, KD, and SFT

The UI should not show NextBestAction fields for this agent mode.

Add two modes:

- **Conversational Agent** — new tool-use models
- **NextBestAction Research** — existing JSON/guardrail evaluation

## Evaluation

Create a held-out test set that is never used for training.

Measure separately:

### Understanding

- intent accuracy
- casual vs CRM classification
- follow-up resolution
- account/entity resolution

### Tool use

- correct tool selection
- valid arguments
- correct account scope
- unnecessary tool-call rate
- tool-call recovery rate

### Answer quality

- factual correctness
- completeness
- groundedness
- record enumeration coverage
- naturalness
- clarification quality

### System behavior

- average latency
- tokens generated
- VRAM usage
- tool rounds
- parse failures
- hallucination rate

Compare Base, Chat-KD, and Chat-SFT using the same questions, tools, and evidence.

## Training work sequence

The architecture roadmap runs first through its baseline and contract gates. The model-training work then follows this sequence:

1. Freeze the current NextBestAction pipeline.
2. Reuse the single shared tool registry and schemas from the architecture roadmap.
3. Build the Claude teacher generator against those schemas.
4. Generate and validate conversational KD records.
5. Curate the conversational SFT subset.
6. Tokenize and check sequence lengths using the exact local training recipe.
7. Train Chat-KD and Chat-SFT adapters locally.
8. Merge, quantize, and register new variants.
9. Run the per-model agent loop against the retrieval baseline and any approved reranker.
10. Run the held-out agent and retrieval evaluation.
11. Export the tool-use, retrieval, and answer-quality Excel report.

Retrieval reranking, durable sessions, LangSmith, and production connector work are governed by the architecture roadmap. They are not prerequisites for generating the first conversational training set.

## Success criteria

The first POC is successful when every new model can:

- answer “hola” naturally
- answer normal CRM questions
- call the correct tool without a hardcoded keyword router
- handle “which are they?” using session history
- include all relevant returned records
- refuse or clarify when evidence is insufficient
- produce natural-language answers rather than NextBestAction JSON
- run within the available local VRAM
