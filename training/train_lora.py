"""
Phase 3 -- QLoRA fine-tuning script, sized for the dev machine's Quadro
T2000 (4GB VRAM). Trains one LoRA adapter per invocation; run twice (once
per dataset) to produce the "kd" and "sft" adapters.

Adapted from ~/Magna/SalesIntelligence/llm/sft/train_lora.py -- reuses that
script's hard-won, empirically-verified sizing decisions almost verbatim
(same hardware, same base model family, same software stack) rather than
re-deriving them:

  - Base model: the PRE-QUANTIZED 4-bit repo
    "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit" (note: despite the repo name,
    this is a plain HF-transformers-loadable bnb-4bit checkpoint, NOT
    Unsloth's own loader format -- no "unsloth-" infix). This is a
    SEPARATE download from models/registry.yaml's inference GGUF -- training
    needs the HF/bnb checkpoint, inference needs the GGUF.
  - peft LoRA (rank 8, alpha 16, all 7 attention/MLP projections) + trl's
    UNPATCHED SFTTrainer + liger-kernel fused loss -- deliberately NOT
    Unsloth (reproducible Unsloth 2026.9.2 + trl 0.24.0 + Python 3.14 bug
    that injects a broken eos/pad sentinel; also a standing preference
    against Unsloth regardless of that specific bug).
  - fp16 (T2000 is Turing/sm75, no bf16 hardware support), paged_adamw_8bit,
    batch_size=1 + grad_accum=4, hand-rolled prepare_model_for_kbit_training
    equivalent (peft's own helper upcasts embed_tokens/lm_head to fp32 and
    OOMs on this card at Qwen3's ~152k-token vocab).

*** IMPORTANT CAVEAT, carried over unchanged: max_seq_length=1024 is the
verified-working ceiling on THIS card with THIS software stack -- but this
project's real prompts (system prompt + full mock_crm context) run well
past that, same as SalesIntelligence's. A LOCAL run at this max_seq_length
is a training-MECHANICS smoke test only (proves forward/backward/optimizer
step/adapter save work without OOM) -- it does NOT prove the model was
trained against a complete, untruncated target completion. The real
training run needs more VRAM than this card has -- see
training/colab_train.py for the Colab T4 path with a realistic
max_seq_length.

Dataset row shape (data/kd_train.jsonl / data/sft_train.jsonl, produced by
data_gen/generate_teacher_drafts.py + data_gen/auto_curate_sft.py or
data_gen/curate_sft_subset.py): {"query", "account_id", "context",
"response", "_source"} -- "context" is the full assembled context text
(common/formatting.py::format_context output), "response" is the full
NextBestAction dict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.context_compaction import compact_formatted_context  # noqa: E402
from common.formatting import build_user_prompt, load_system_prompt  # noqa: E402
from eval import guardrails  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"

DEFAULT_BASE_MODEL = "unsloth/Qwen3-1.7B-bnb-4bit"  # switched from Qwen3-4B 2026-09-10, see training/colab_train.py's BASE_MODEL comment
DEFAULT_MAX_SEQ_LENGTH = 1024  # see module docstring -- the verified LOCAL ceiling, not a quality claim
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_dataset_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON -- {exc}") from exc
    return rows


def build_conversational_examples(rows: list[dict]) -> list[dict]:
    """Convert each dataset row into trl's conversational prompt/completion
    format, reusing common/formatting.py's real prompt-assembly helpers so
    training sees exactly the input shape live inference will."""
    system_prompt = load_system_prompt()
    examples = []
    for row in rows:
        context_text = row["context"]
        pre_warnings = guardrails.pre_check(context_text)
        user_prompt = build_user_prompt(row["query"], context_text, pre_warnings)
        completion_text = json.dumps(row["response"], ensure_ascii=False)
        examples.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": [{"role": "assistant", "content": completion_text}],
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="e.g. data/kd_train.jsonl or data/sft_train.jsonl")
    parser.add_argument("--adapter-out", required=True, help="e.g. adapters/kd or adapters/sft")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--compact-context-chars",
        type=int,
        default=0,
        help="Compact flattened CRM context to this character budget before training; 0 keeps full context.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = load_dataset_rows(dataset_path)
    if not rows:
        raise RuntimeError(f"No training examples found in {dataset_path}")
    print(f"Loaded {len(rows)} training examples from {dataset_path}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Python 3.14 / dill compatibility workaround, same as SalesIntelligence's
    # train_lora.py -- `datasets`' fingerprinting dill-pickles for a cache
    # key, dill 0.4.0 breaks on Python 3.14's changed Pickler signature.
    # Fingerprints are cache-only here (never relied on for correctness), so
    # falling back to a repr()-based hash on failure is safe.
    import datasets.fingerprint as _hf_fingerprint

    @classmethod
    def _dill_or_repr_hash(cls, value):
        try:
            from datasets.utils._dill import dumps as _dill_dumps
            return cls.hash_bytes(_dill_dumps(value))
        except Exception:
            return cls.hash_bytes(repr(value).encode("utf-8", errors="replace"))

    _hf_fingerprint.Hasher.hash = _dill_or_repr_hash

    # transformers 5.5.0 regression, found on this exact card 2026-09-10 (not
    # present when SalesIntelligence's sibling script was last verified
    # 2026-09-04 -- `transformers` is an unpinned transitive dependency here,
    # so a point release landed the change in between): `from_pretrained`
    # now calls `caching_allocator_warmup()` to pre-allocate one big CUDA
    # block "to avoid having to Malloc afterwards" (a load-time SPEED
    # optimization per that function's own docstring, not required for
    # correctness). Its byte-count estimate for this bnb-4bit checkpoint
    # badly overshoots -- observed asking for another 2.42 GiB on top of the
    # 3.11 GiB already used loading the actual (quantized, ~2.5GB) weights,
    # OOM'ing immediately on this 3.6GB-usable card before training even
    # starts. Disabling it just means normal incremental (on-demand)
    # allocation during loading instead of one upfront block -- slower, but
    # correctness-neutral, and actually safer on a VRAM-constrained card
    # than an overestimated upfront reservation.
    import transformers.modeling_utils as _hf_modeling_utils

    _hf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading base model {args.base_model!r} (pre-quantized 4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map={"": 0})

    if cuda_available:
        print(
            f"VRAM after base model load: "
            f"{torch.cuda.memory_allocated() / 1024**2:.0f} MiB allocated"
        )

    model.config.use_cache = False

    # NOT peft's prepare_model_for_kbit_training() -- it upcasts every
    # remaining fp16/bf16 param (including this model's large, un-quantized
    # embed_tokens/lm_head) to fp32, which OOMs on this card. QLoRA works
    # fine without that upcast -- see SalesIntelligence's train_lora.py for
    # the full measured explanation.
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
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.compact_context_chars:
        summaries_path = PROJECT_ROOT / "data" / "context_summaries.json"
        summaries = json.loads(summaries_path.read_text()) if summaries_path.exists() else {}
        compacted_rows = [
            {
                **row,
                "context": compact_formatted_context(
                    row["context"], args.compact_context_chars, summaries=summaries
                ),
            }
            for row in rows
        ]
        examples = build_conversational_examples(compacted_rows)
        print(
            f"Compacted CRM contexts to <= {args.compact_context_chars} characters "
            f"({len(summaries)} documents summarized via {summaries_path.name if summaries else 'none -- run data_gen/build_context_summaries.py first'}; "
            f"structured records minified; verify final token lengths before training)."
        )
    else:
        examples = build_conversational_examples(rows)
    dataset = Dataset.from_list(examples)

    bf16_ok = cuda_available and torch.cuda.is_bf16_supported()
    adapter_path = Path(args.adapter_out)
    output_dir = str(adapter_path / "_trainer_output")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        max_length=args.max_seq_length,
        packing=False,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="paged_adamw_8bit",
        report_to="none",
        save_strategy="no",
        gradient_checkpointing=True,
        seed=3407,
        use_liger_kernel=True,  # required on this card -- see module docstring
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)

    print(
        f"Starting training: {len(dataset)} examples, {args.epochs} epochs, "
        f"effective batch size {args.batch_size * args.grad_accum}..."
    )
    train_result = trainer.train()
    print("Training finished:", train_result.metrics)

    if cuda_available:
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(
            f"Peak VRAM during training: {peak_allocated:.0f} MiB (of "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MiB total)"
        )

    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Saved LoRA adapter to {adapter_path}")


if __name__ == "__main__":
    main()
