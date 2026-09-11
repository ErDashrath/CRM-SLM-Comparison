"""
Phase 3 -- merge a trained LoRA adapter into the FULL-PRECISION base model,
producing a merged HF checkpoint ready for convert_hf_to_gguf.py.

Why the full-precision base here, not the bnb-4bit checkpoint training
used (unsloth/Qwen3-1.7B-bnb-4bit): a LoRA adapter is a separate additive
low-rank delta (base_weight + A@B), stored in its own fp16/bf16 tensors
regardless of what precision the base was frozen at during training.
Merging only makes numeric sense against the base's real weights --
merging into an already-4-bit-quantized approximation of them would bake
that quantization error into the merged result before we even get to the
GGUF quantization step. So: merge against "Qwen/Qwen3-1.7B" (bf16, official
repo, same weights the bnb-4bit checkpoint was quantized from), then
convert_hf_to_gguf.py -> F16 GGUF -> llama-quantize -> Q4_K_M, per
training/merge_and_quantize.sh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

FULL_PRECISION_BASE = "Qwen/Qwen3-1.7B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="e.g. adapters/kd")
    parser.add_argument("--merged-out", required=True, help="e.g. adapters/kd/merged_hf")
    parser.add_argument("--base-model-path", default=None, help="Local path to a pre-downloaded full-precision base, if not using the hub id directly")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base_model_path or FULL_PRECISION_BASE
    print(f"Loading full-precision base {base!r}...")
    tokenizer = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

    print(f"Attaching LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    out_dir = Path(args.merged_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved merged model to {out_dir}")


if __name__ == "__main__":
    main()
