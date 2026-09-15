# Setup Guide

## Prerequisites

- Windows, macOS, or Linux with Python 3.11+ and Git.
- An NVIDIA GPU is recommended for local inference/training. CPU-only
  inference is possible with a compatible llama.cpp build; longer-context
  training can use Google Colab or another GPU provider.
- An Anthropic API key for Claude teacher generation, Claude-as-judge scoring,
      and the research report. OpenAI can be used as the configured fallback.

## Installation

```bash
cd CRM-SLM-Comparison
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows:     Copy-Item .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`. Keep `.env` local; it is ignored by Git.
Use `TEACHER_BACKEND=claude` for the intended teacher/judge/report workflow,
or `TEACHER_BACKEND=openai` for the fallback backend. On Windows PowerShell,
set the variable with `$env:TEACHER_BACKEND="claude"`; on macOS/Linux, prefix
commands with `TEACHER_BACKEND=claude`.

## Quick Verification

```bash
python -m py_compile \
      common/*.py data_gen/*.py eval/*.py models/*.py report/*.py training/*.py ui/app.py
python models/inference.py
```

The inference smoke test requires the base GGUF referenced by
`models/registry.yaml`. Generated adapters, merged GGUF files, model caches,
and evaluation reports are intentionally kept out of Git.

## Main Workflows

```bash
# Run the local 1.7B QLoRA mechanics smoke test.
python training/train_lora.py \
      --dataset data/kd_train.jsonl \
      --adapter-out adapters/kd-local \
      --max-seq-length 1024

# Evaluate all variants and generate the report workbook.
python -m eval.run_eval
python -m report.research_agent
python -m report.build_excel

# Launch the Streamlit comparison UI.
python -m streamlit run ui/app.py --server.port 8502
```

Local training at short sequence lengths validates the training mechanics.
For complete CRM contexts, use the Colab workflow in `training/` or enable
the compact-context pipeline before training. Do not commit API keys, model
weights, adapter outputs, or generated reports.

## CRM assistant architecture

The Streamlit app also includes a normal chat assistant backed by a structured,
read-only CRM layer:

1. `common/crm_store.py` normalizes the mock CRM fixtures into a local SQLite
      database containing accounts, opportunities, contacts, and activities.
2. `common/crm_query.py` validates semantic query plans and executes allow-listed,
      read-only SQL against the CRM schema.
3. `common/crm_tools.py` exposes the compatibility API and one unified `crm` tool.
4. `common/crm_retrieval.py` performs local hybrid retrieval over CRM records
      using BM25-style lexical scoring plus entity and document-type reranking.
5. `common/rag_chat.py` passes the structured tool result and retrieved
      evidence to the local model for grounded synthesis.

The assistant never invents an entity that is absent from the schema. Leads
are represented in the local CRM schema and can be queried directly. A production deployment can replace the SQLite
source with the company's CRM/ERP connector while keeping the same tool and
retrieval contracts.

# CRM Small-Model Comparison POC

A quantitative, modular comparison of three ways to specialize the same ~4B
open model for CRM tasks: doing nothing (a quantized base model), knowledge
distillation from a stronger teacher (Claude), and supervised fine-tuning on
curated CRM data. This is a standalone project, separate from the related
SalesIntelligence project
(read-only reuse of that project's mock CRM data, guardrails, and training
approach — see below — but never writes back to it).

## The three variants

All three share the exact same base model, so the comparison isolates *how
it was specialized*, not *which base model*:

| Variant | Training data | Volume | Human review |
|---|---|---|---|
| **base** | none | — | — |
| **kd** | Claude-generated CRM demonstrations, guardrail-filtered only | ~120-150 | none |
| **sft** | curated subset of the same draft pool, corrected/verified | ~50-60 | full |

**Terminology note:** this is "distillation via teacher-generated
demonstrations" (Alpaca/Vicuna-style), not logit-level knowledge
distillation — there's no logprob access to Claude at that granularity
through the API. The final tradeoffs report is explicit about this so the
methodology holds up when it reaches a non-technical stakeholder.

## Reused from SalesIntelligence (read-only)

- **Base model**: the Apache 2.0 Qwen GGUF configured in
      `models/registry.yaml`; model weights are downloaded locally and are not
      committed to this repository.
- **`mock_crm/`**: copied once at setup (4 accounts: Acme Corp, Globex,
  Initech, Delta — Delta is the held-out unseen-generalization test).
- **Guardrails pattern**: `eval/guardrails.py` ports the deterministic
  discount-ceiling / economic-buyer / competitor-threat checks from
  `llm/guardrails.py`, including its three already-fixed bugs (the
  `competitive` regex miss, the global-playbook false-positive, and
  negation blindness).
- **Training stack**: plain `peft` + `bitsandbytes` + `trl` (unpatched
  `SFTTrainer`) + `liger-kernel` — **not** Unsloth (reproducible
  Unsloth 2026.9.2 + trl 0.24.0 + Python 3.14 bug, plus a standing
  preference against Unsloth regardless).
- **Training compute**: local GPU or CPU smoke tests; longer-context training
      can use Google Colab or another compatible GPU provider.

## Layout

```
models/        registry.yaml (the modularity lever) + inference.py (registry -> loaded model)
data_gen/      teacher draft generation (KD) + curation CLI (SFT)
data/          kd_train.jsonl, sft_train.jsonl, eval_set.jsonl
training/      train_lora.py (parameterized), merge_and_quantize.sh, colab_train.ipynb
eval/          guardrails.py, judge.py, perf.py, run_eval.py
report/        build_excel.py, research_agent.py
ui/            app.py (Streamlit, reads models/registry.yaml dynamically)
results/       results-<timestamp>.xlsx, tradeoffs-<timestamp>.md (generated, gitignored)
```

Full build plan and rationale are maintained separately from this repository.

## Status

- [x] Phase 0 — scaffold, model registry, inference wrapper
- [x] Phase 1 — teacher data generation (KD track). `data/kd_train.jsonl`: 44 accepted,
      `data/kd_rejected.jsonl`: 14 guardrail-rejected, 0 unparseable, out of 58 queries.
      Generated via a mix of Claude Code sub-agents (no API key needed) and the Anthropic
      API (once `ANTHROPIC_API_KEY` was added) — see `data/kd_manifest.json`.
- [x] Phase 2 — SFT subset. **Automated, not human-reviewed** (the user opted
      to skip manual review) — `data_gen/auto_curate_sft.py` stratifies by
      account and keeps the top 65% by confidence per account (not a global
      top-N, which would have skewed almost entirely toward Delta's easy,
      high-confidence scenarios and excluded Acme's harder multi-risk ones).
      31/44 selected, `human_review_minutes: 0` — labeled honestly in
      `data/sft_manifest.json`, not claimed as curation the project didn't
      pay for. `data_gen/curate_sft_subset.py` (the real human-review CLI,
      with a per-account ground-truth cheat sheet via `common/cheatsheet.py`)
      still exists and can be run later to replace this with a genuinely
      reviewed set.
- [x] Phase 3 — training + GGUF merge. Both adapters trained locally on a
      constrained GPU with Qwen3-1.7B (switched from 4B — see below) at
      `max_seq_length=2575` over LLM-summarized, compacted CRM context
      (`--compact-context-chars 1800`, 100% of examples fully intact, no
      truncated completions — see `common/context_compaction.py` and
      `data_gen/build_context_summaries.py`). Merged against the
      full-precision base and quantized to Q4_K_M via a locally-built
      `llama-quantize` from a local llama.cpp build. All 3 variants
      (`base`/`kd`/`sft`) verified loading and
      generating through `models/inference.py`'s shared registry path.
      **Base model note:** switched from Qwen3-4B to Qwen3-1.7B
      2026-09-10 — the 4B model OOM'd during training on the available
      constrained GPU environments at the longer sequence lengths tested.
      Qwen3-1.7B is the same family/license (Apache 2.0)/training recipe,
      under half the params — `models/registry.yaml`'s `base` entry now
      points at a locally-downloaded `models/Qwen3-1.7B-Q4_K_M.gguf`.
- [x] Phase 4 — evaluation harness. `data/eval_set.jsonl`: 20 genuinely new
      queries (no overlap with the 58 training queries; Delta is NOT a true
      unseen-account test since Phase 1 trained on all 4 accounts — a
      deviation from the original plan, noted). Ran all 3 variants x 20
      queries = 60 generations, each guardrail-checked, Claude-judged
      (correctness/completeness/risk_surfacing 1-5), and perf-measured
      (tokens/sec, VRAM). Results: `results/eval_results.json`. **Real bug
      found and fixed 2026-09-10**: first run used `max_tokens=700`, which
      the more verbose kd/sft variants hit far more often than base (2 vs
      5 vs 8 truncations) — this alone explained an apparent "fine-tuning
      made JSON parsing worse" pattern that was actually just a token-budget
      artifact. Re-ran at `max_tokens=1200`; buggy run kept at
      `results/eval_results_v1_buggy_700tok.json` for reference, not used
      in the report. **Corrected finding**: sft beats both base and kd on
      every judge dimension (3.25/2.80/3.75 vs base's 3.15/2.65/3.60 vs
      kd's 2.80/2.35/2.95) and has the best parse rate (19/20) — curation
      beat raw volume here. All 3 variants tie on guardrail pass rate
      (6/20), suggesting guardrail compliance didn't transfer to novel
      questions regardless of training approach.
- [x] Phase 5 — Excel report + research agent. `report/research_agent.py`
      produces a numbers-cited tradeoffs writeup. One factual error caught
      and fixed by hand in the first draft: "10 extra milliseconds" for
      the SFT-vs-base speed gap — actually ~3 seconds per full response.
      **Second, deeper pass, prompted by manually reading real UI output**:
      found and fixed 5 real bugs in `eval/guardrails.py` itself (not the
      eval harness) — (1) no deduplication, one issue could produce 2-4
      near-identical violation lines; (2) couldn't distinguish citing the
      fixed policy ceiling ("exceeds our 10% hard ceiling") from restating
      the customer's ask ("they want 12%") — resolved per user direction:
      only the customer's ask is flagged now; (3) `response_lag_or_stall`'s
      surface check matched the bare word "pending", false-triggering on
      unrelated content (Delta's real "budget approval pending") and
      silently hiding genuine unsurfaced-lag misses; (4)
      `discount_ask_above_policy`'s surface check was just the word
      "discount" — "no discount is warranted" counted as surfacing the
      risk; (5) an exact 10.0% discount fell through to the weaker
      "approved band" message due to a strict `>` where policy says
      "at or above 10%". All 60 existing responses were **re-scored**
      (`eval/rescore_results.py` — not re-generated, model outputs didn't
      change) against the fixed logic: overall guardrail pass rate rose
      from 18/60 (30%) to 25/60 (42%), and the per-variant breakdown
      became genuinely differentiated for the first time — base 9/20,
      kd 9/20, sft 7/20 — instead of an artificial 6/20 three-way tie that
      was itself a symptom of the bugs equally polluting all three
      variants. Report and Excel rebuilt on the corrected data
      (`results/results-20260910-2232.xlsx`); final recommendation shifted
      to a more nuanced "ship SFT, but the guardrail gap is real and
      within noise at N=20 — validate in production" rather than the
      earlier unqualified SFT recommendation.
- [x] Dataset Browser page — `ui/pages/1_📁_Dataset_Browser.py` (Streamlit's
      standard multi-page convention — appears in the sidebar nav
      automatically alongside the comparison tool). Per account: profile +
      contacts table, opportunity metrics (stage/value/win-probability/
      close date), color-coded risk factors, line items, and every email
      and call transcript rendered in full and individually selectable, plus
      a shared "Company-Wide Reference" tab for the 3 playbooks + product
      catalog. Reuses `common/context.py::assemble_context()` — the same
      function training and eval read through — so it always reflects
      exactly what the models actually see, nothing re-derived. Verified:
      data logic checked against all 4 accounts (including Globex's
      zero-risk-factor edge case) before launch; server started with no
      errors.
- [x] Phase 6 — Streamlit UI. `ui/app.py` reads `models/registry.yaml`
      dynamically (adding a 4th variant needs zero UI code changes), lets
      you pick an account + query, runs it through every registered
      variant sequentially (VRAM constraint — one GGUF resident at a
      time), shows guardrail pass/fail + optional judge scores + perf
      side by side, and has a "Run Full Eval Suite" button that re-invokes
      the eval → research agent → Excel pipeline. Server verified running
      (`streamlit run ui/app.py`, HTTP 200) and the core comparison
      function verified working end-to-end with a real generation +
      guardrail check — **not** visually verified in a browser (no browser
      tool available this session); run it yourself at
      `http://localhost:8502` to confirm the UI itself renders correctly.
      Since Phase 6 shipped: added an example-query picker, a persistent
      per-(account,query,variant) results store so runs don't overwrite
      each other, a guardrail-explainer tooltip, and a Dataset Browser
      page (`ui/pages/`) for reading the full mock CRM in a presentable
      form. **Parse-robustness, two rounds** (found via real use, not
      hypothetical): round 1 added a retry-on-unparseable path (base
      sometimes burns its whole token budget on `<think>` reasoning before
      reaching JSON); round 2 (same day) found that retry alone wasn't
      enough — kd/sft also hit unparseable responses, and
      `parse_next_best_action` was swallowing every validation failure
      into a blind `None` with no visibility into *why*. Fixed properly:
      `common/formatting.py` now has a deterministic repair pass for
      common schema mismatches (bad `action_type` spelling, confidence as
      a percentage string, payload as a string instead of an object,
      risk_flags as a bare string) that succeeds WITHOUT needing a second
      model call at all — tested against 9 synthetic malformation cases,
      7 auto-repair cleanly; the 2 genuinely unrepairable ones (no JSON at
      all, missing a truly required field) now surface the *specific*
      pydantic validation error instead of a generic "could not parse",
      both to the retry prompt (so the model is told exactly what was
      wrong) and to the UI.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Configure `ANTHROPIC_API_KEY` in the local `.env` when using Claude. The
initial dataset was generated through the teacher pipeline, with the exact
backend recorded in the project manifests. Judge (Phase 4) and research
agent (Phase 5) default to `TEACHER_BACKEND=claude` via
`common/llm_backend.py`; set `TEACHER_BACKEND=openai` for the fallback.
