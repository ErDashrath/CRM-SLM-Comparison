#!/usr/bin/env bash
# Phase 3, final step: merge each trained LoRA adapter (adapters/kd,
# adapters/sft) into the full-precision base, convert to GGUF, quantize to
# Q4_K_M -- so all three variants (base/kd/sft) run through the identical
# llama.cpp backend for a fair latency/VRAM comparison in Phase 4, not a
# peft/transformers path for two variants and llama.cpp for the third.
#
# Output paths match models/registry.yaml's kd/sft gguf_path entries
# exactly: adapters/<variant>/merged-Q4_K_M.gguf
#
# Requires (already set up 2026-09-10):
#   - llama.cpp cloned + llama-quantize built (CPU-only build is enough,
#     quantization doesn't need CUDA) at ~/Magna/llama.cpp/build/bin/
#   - convert_hf_to_gguf.py's requirements installed in venv-gpu
#   - the full-precision base (Qwen/Qwen3-1.7B) downloaded -- pass
#     --base-model-path if using a local copy instead of hitting the Hub

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/home/dsp-at-magna/Magna/venv-gpu/bin/python
LLAMA_CPP=/home/dsp-at-magna/Magna/llama.cpp
BASE_MODEL_PATH="${1:-}"  # optional: local path to full-precision base

for variant in kd sft; do
  echo "=== [$variant] Merging LoRA adapter into full-precision base ==="
  if [ -n "$BASE_MODEL_PATH" ]; then
    $PYTHON training/merge_lora.py --adapter "adapters/$variant" --merged-out "adapters/$variant/merged_hf" --base-model-path "$BASE_MODEL_PATH"
  else
    $PYTHON training/merge_lora.py --adapter "adapters/$variant" --merged-out "adapters/$variant/merged_hf"
  fi

  echo "=== [$variant] Converting merged HF checkpoint to GGUF (F16) ==="
  $PYTHON "$LLAMA_CPP/convert_hf_to_gguf.py" "adapters/$variant/merged_hf" \
    --outfile "adapters/$variant/merged-F16.gguf" --outtype f16

  echo "=== [$variant] Quantizing to Q4_K_M ==="
  "$LLAMA_CPP/build/bin/llama-quantize" \
    "adapters/$variant/merged-F16.gguf" "adapters/$variant/merged-Q4_K_M.gguf" Q4_K_M

  echo "=== [$variant] Cleaning up intermediate files (merged_hf, F16 gguf) ==="
  rm -rf "adapters/$variant/merged_hf" "adapters/$variant/merged-F16.gguf"

  echo "=== [$variant] done -> adapters/$variant/merged-Q4_K_M.gguf ==="
done

echo "=== All variants merged and quantized. Update models/registry.yaml if paths changed (they shouldn't have). ==="
