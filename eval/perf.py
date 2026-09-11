"""
Phase 4 -- performance measurement (latency, tokens/sec, peak VRAM) for one
generate() call, timed around models/inference.py's VariantHandle.

VRAM sampling uses pynvml directly (not nvidia-smi subprocess parsing --
faster, no shelling out per call) if a GPU is present; returns None for
vram_mb on a machine with no NVIDIA GPU rather than raising, so this stays
usable in CI/no-GPU contexts for everything except the actual number.

Retry-on-unparseable (added 2026-09-11, found via real UI use): the "base"
(untrained) variant occasionally burns its ENTIRE token budget inside a
Qwen3 <think> block before ever reaching the JSON answer, especially on
harder portfolio-wide queries -- confirmed via a real case where
output_tokens landed at exactly the max_tokens ceiling with zero JSON
produced. kd/sft rarely do this (their training completions never included
a <think> block, so they've mostly learned to skip it), which is also why
this only ever showed up for base. This mirrors a retry pattern already
established elsewhere in this codebase family
(SalesIntelligence's llm/sft/generate_candidates.py and
orchestrator/pipeline.py both retry once with a stricter instruction on
parse failure) -- it was just never wired into this project's live
generation path until now.
"""

from __future__ import annotations

import time
from typing import Optional

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def _gpu_memory_used_mb() -> Optional[float]:
    if not _NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2
    except Exception:
        return None


_RETRY_SUFFIX = (
    "\n\nYour previous response did not contain a parseable JSON object -- it "
    "likely ran out of space while reasoning before ever writing the answer. "
    "This time, skip extended step-by-step reasoning entirely and respond "
    "with ONLY the JSON object: no <think> block, no markdown fences, no "
    "commentary before or after it."
)


def timed_generate(
    handle, system_prompt: str, user_prompt: str, max_tokens: int = 700, retry_on_unparseable: bool = True
) -> dict:
    """Run one generation, returning {"text", "elapsed_seconds",
    "output_tokens", "tokens_per_second", "peak_vram_mb", "retried"}.
    Uses the loaded model's own tokenizer (handle._llm.tokenize) to count
    output tokens exactly, rather than estimating from character count.

    When retry_on_unparseable is True (the default) and the first attempt
    doesn't parse as a NextBestAction, retries ONCE with an instruction to
    skip reasoning and go straight to JSON -- see module docstring. Timing
    and token counts reflect BOTH attempts combined (the real cost of
    getting a usable answer), not just the second one -- "retried" in the
    result tells the caller this happened, so it's visible, not hidden."""
    from common.formatting import parse_next_best_action  # local import: avoids a hard dependency for pure perf-timing callers

    vram_before = _gpu_memory_used_mb()
    start = time.time()
    text = handle.generate(system_prompt, user_prompt, max_tokens=max_tokens)

    retried = False
    if retry_on_unparseable and parse_next_best_action(text) is None:
        retried = True
        text = handle.generate(system_prompt, user_prompt + _RETRY_SUFFIX, max_tokens=max_tokens)

    elapsed = time.time() - start
    vram_after = _gpu_memory_used_mb()

    output_tokens = len(handle._llm.tokenize(text.encode("utf-8"), add_bos=False))
    tokens_per_second = output_tokens / elapsed if elapsed > 0 else None

    peak_vram_mb = None
    if vram_before is not None and vram_after is not None:
        peak_vram_mb = max(vram_before, vram_after)

    return {
        "text": text,
        "elapsed_seconds": round(elapsed, 3),
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else None,
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
        "retried": retried,
    }
