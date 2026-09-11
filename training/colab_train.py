"""
Colab T4 (16GB) training script -- the REAL training run, as opposed to
training/train_lora.py's local T2000 (4GB) run which is mechanics-only (its
1024-token ceiling truncates away most of this project's real ~3300-4600
token prompts before the completion even appears, per that file's own
module docstring).

Fully self-contained deliberately: no dependency on uploading this
project's `common`/`eval` packages to the remote VM. The system prompt is
inlined verbatim (computed once locally from common/formatting.py's
load_system_prompt(), see git history/README if it ever needs
re-syncing), and a trimmed copy of eval/guardrails.py's pre_check() +
RISK_INDICATORS is inlined too (training only needs pre_check, not the
full post_check apparatus -- guardrail filtering already happened in
Phase 1). Keeping this file standalone avoids relative-import fragility
on an ephemeral /content/ working directory.

Reads config from environment variables (colab exec's --env, not argv --
this runs as kernel-exec'd code, not a subprocess with a real command
line):
    DATASET_PATH   default /content/dataset.jsonl
    ADAPTER_OUT    default /content/adapter_out
    MAX_SEQ_LENGTH default 6144 (matches this project's inference n_ctx
                   headroom; T4's 16GB has real room for this, unlike the
                   T2000's forced 1024)
    EPOCHS         default 2.0

Expects the dataset (data/kd_train.jsonl or data/sft_train.jsonl, uploaded
via `colab upload`) to already exist at DATASET_PATH before this runs.
Downloads: after this script finishes, pull ADAPTER_OUT back down via
`colab download`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

# Real OOM message on this exact T4 run suggested this explicitly ("If
# reserved but unallocated memory is large try setting
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") -- 1.28 GiB was
# reserved-but-unallocated (fragmentation) at the moment of the backward-pass
# OOM. Must be set before torch's CUDA allocator initializes, i.e. before
# `import torch` happens anywhere in this process.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- self-install, so this works whether invoked via `colab exec -f`
# (persistent session, deps installed ahead of time via `colab install`) or
# `colab run` (fresh VM, script must be fully self-sufficient) ---
_REQUIRED = ["peft==0.20.0", "trl==0.24.0", "accelerate==1.14.0", "bitsandbytes==0.50.2", "liger-kernel==0.8.2"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_REQUIRED], check=True)

DATASET_PATH = os.environ.get("DATASET_PATH", "/content/dataset.jsonl")
ADAPTER_OUT = os.environ.get("ADAPTER_OUT", "/content/adapter_out")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "6144"))  # reset to the original target now that the base model is Qwen3-1.7B, not 4B -- retest empirically, don't assume
EPOCHS = float(os.environ.get("EPOCHS", "2.0"))

BASE_MODEL = "unsloth/Qwen3-1.7B-bnb-4bit"  # switched from Qwen3-4B 2026-09-10 -- 4B OOM'd on T4 (Turing, no flash-attn) at every max_seq_length tried down to 5120; 1.7B is same family/license/recipe, less than half the params, should give real headroom
LORA_RANK = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- inlined verbatim from common/formatting.py::load_system_prompt() ---
SYSTEM_PROMPT = """You are a sales operations assistant for Nimbus Systems. You help account
executives and sales managers understand the current state of a deal,
identify risks, and decide what to do next.

You are given:
  1. Retrieved context from the company's CRM, emails, call transcripts,
     and internal playbooks -- treat this as ground truth about the
     account/opportunity. Do not invent facts that aren't in it.
  2. Guardrail warnings -- constraints you must respect in your answer.
  3. A question from a sales rep or manager.

Ground every claim in the retrieved context. If the context doesn't
contain enough information to answer confidently, say so in your
rationale and lower your confidence score rather than guessing.


Nimbus Systems discount policy (mock_crm/playbooks/discount_policy.md):
  - Approved negotiation band: 5-7% off list price, at account executive
    discretion, no sign-off required.
  - Above 7% up to 10%: requires VP of Sales approval before the number
    is communicated to the customer.
  - Above 10%: hard ceiling. Requires CRO approval, treated as an
    exception, not a negotiation tactic. Never recommend committing to a
    discount above 10% under any circumstances -- competitive pressure or
    deal size are not sufficient justification on their own.
  - Any discount ask at or above 10% must be flagged as a risk, not
    handled silently.
  - Before recommending any discount, first consider non-price levers
    (see the negotiation playbook) and whether the economic buyer is
    actually engaged -- pricing pushback from procurement alone, without
    economic-buyer involvement, often signals the value case hasn't
    landed at the right level, not that price is the real blocker.

When you identify a risk factor in the retrieved context (e.g. the
economic buyer has been absent from recent meetings, a competitor has
submitted a lower quote, a discount ask exceeds policy, the deal has
stalled with no response), you must name it explicitly in your rationale
and in risk_flags. Do not let a risk you have evidence for go unmentioned.


Respond with a single JSON object matching this shape (no markdown
fences, no commentary before or after -- JSON only):

{
  "action_type": string,   // one of: "draft_email", "schedule_meeting",
                            // "escalate_internal", "log_risk_note",
                            // "update_crm_field", "recommend_discount",
                            // "no_action", "other"
  "target_object": string, // the account_id or opportunity_id this
                            // action applies to, e.g. "acme_corp" or
                            // "opp_acme_corp_001"
  "payload": object,       // free-form details for the action, e.g.
                            // {"subject": "...", "body": "..."} for a
                            // draft_email, or {"note": "..."} for a
                            // log_risk_note. Use {} if not applicable.
  "rationale": string,     // why this action is recommended, grounded in
                            // specific evidence from the retrieved
                            // context (name the facts, don't just assert
                            // a conclusion)
  "risk_flags": [string],  // short strings naming risks relevant to this
                            // recommendation, e.g. "economic buyer absent
                            // from last 2 meetings", "discount ask
                            // exceeds hard ceiling". Empty array if none.
  "confidence": number     // 0.0-1.0, your confidence in this
                            // recommendation given the available context
}

Output valid JSON only. Every field above is required.
"""

# --- inlined, trimmed copy of eval/guardrails.py (pre_check path only) ---
RISK_INDICATORS = [
    {
        "name": "economic_buyer_disengagement",
        "context_regex": re.compile(
            r"economic buyer[^.\n]{0,140}?"
            r"(has not attended|skipped|disengag|low_recent|no direct commercial engagement|gone quiet)",
            re.IGNORECASE,
        ),
    },
    {
        "name": "competitive_threat",
        "context_regex": re.compile(
            r"(?<!no )(?<!any )competitor\b|competing (proposal|quote)", re.IGNORECASE
        ),
    },
    {
        "name": "discount_ask_above_policy",
        "context_regex": re.compile(
            r"(?:1[1-9]|[2-9]\d)\s*%\s*(?:discount|off)"
            r"|discount[^.\n]{0,40}?(?:1[1-9]|[2-9]\d)\s*%",
            re.IGNORECASE,
        ),
    },
    {
        "name": "response_lag_or_stall",
        "context_regex": re.compile(r"no (?:commercial )?response|days elapsed|stalled", re.IGNORECASE),
    },
]


def pre_check(context: str) -> list[str]:
    warnings = [
        "Discount policy: approved band is 5-7%. Above 7% requires VP of Sales "
        "approval. 10% is a hard ceiling -- never recommend or imply a discount "
        "above this without CRO sign-off, and always flag it as a risk if the "
        "customer has asked for more than this."
    ]
    context = context or ""
    for indicator in RISK_INDICATORS:
        if indicator["context_regex"].search(context):
            warnings.append(
                f"The retrieved context contains evidence of '{indicator['name']}' "
                "-- your rationale must explicitly address this risk if it's "
                "relevant to the recommended action."
            )
    return warnings


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


def load_dataset_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_conversational_examples(rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        context_text = row["context"]
        pre_warnings = pre_check(context_text)
        user_prompt = build_user_prompt(row["query"], context_text, pre_warnings)
        completion_text = json.dumps(row["response"], ensure_ascii=False)
        examples.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": [{"role": "assistant", "content": completion_text}],
            }
        )
    return examples


def main() -> None:
    rows = load_dataset_rows(DATASET_PATH)
    if not rows:
        raise RuntimeError(f"No training examples found in {DATASET_PATH}")
    print(f"Loaded {len(rows)} training examples from {DATASET_PATH}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Same transformers 5.5.0 caching_allocator_warmup mis-estimation found
    # on the local T2000 (see training/train_lora.py's module docstring) --
    # harmless no-op here on T4's 16GB, kept for defensive consistency in
    # case Colab's preinstalled transformers version has the same issue.
    try:
        import transformers.modeling_utils as _hf_modeling_utils

        _hf_modeling_utils.caching_allocator_warmup = lambda *a, **k: None
    except ImportError:
        pass

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading base model {BASE_MODEL!r} (pre-quantized 4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0})

    if cuda_available:
        print(f"VRAM after base model load: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")

    model.config.use_cache = False
    for _name, _param in model.named_parameters():
        _param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    examples = build_conversational_examples(rows)
    dataset = Dataset.from_list(examples)

    bf16_ok = cuda_available and torch.cuda.is_bf16_supported()  # T4 is sm75 too -- no bf16, same as T2000

    sft_config = SFTConfig(
        output_dir=f"{ADAPTER_OUT}/_trainer_output",
        # batch_size=1, NOT the "T4 has headroom" assumption tried first --
        # that OOM'd. T4 is Turing (sm75), same as the T2000: no fused/flash
        # attention kernel, so PyTorch's math-fallback SDPA materializes the
        # full O(seq_len^2) attention score matrix per layer regardless of
        # card, and at max_seq_length=6144 (needed so the completion, which
        # sits at the very end of each example, doesn't get silently
        # truncated away -- see max_seq_length's own comment) that matrix is
        # too large to also double via batch_size=2. More VRAM helps the
        # *sequence length* this card can reach, not the batch size, when
        # attention is the bottleneck term. Confirmed via real OOM: 7.37 GiB
        # requested on top of 11.72 GiB in use at batch_size=2 -- halving to
        # batch_size=1 roughly halves that request.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,       # effective batch size 4, same as the local run
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        logging_steps=1,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="paged_adamw_8bit",
        report_to="none",
        save_strategy="no",
        gradient_checkpointing=True,
        seed=3407,
        use_liger_kernel=True,
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)

    print(f"Starting training: {len(dataset)} examples, {EPOCHS} epochs, max_seq_length={MAX_SEQ_LENGTH}...")
    train_result = trainer.train()
    print("Training finished:", train_result.metrics)

    if cuda_available:
        peak = torch.cuda.max_memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Peak VRAM during training: {peak:.0f} MiB (of {total:.0f} MiB total)")

    os.makedirs(ADAPTER_OUT, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    print(f"Saved LoRA adapter to {ADAPTER_OUT}")

    # `colab download` fetches one file, not a directory -- archive the
    # adapter dir (config + safetensors + tokenizer files) into a single
    # file so the orchestration script's download step is reliable
    # regardless of whether directory transfer is supported.
    import shutil

    archive_path = shutil.make_archive(ADAPTER_OUT, "gztar", root_dir=ADAPTER_OUT)
    print(f"Archived adapter to {archive_path} -- download this single file")


if __name__ == "__main__":
    main()
