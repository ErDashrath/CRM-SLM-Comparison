"""
Phase 4 -- performance measurement (latency, tokens/sec, peak VRAM) for one
generate() call, timed around models/inference.py's VariantHandle.

VRAM sampling uses pynvml directly (not nvidia-smi subprocess parsing --
faster, no shelling out per call) if a GPU is present; returns None for
vram_mb on a machine with no NVIDIA GPU rather than raising, so this stays
usable in CI/no-GPU contexts for everything except the actual number.

Retry-on-unparseable, v1 (2026-09-11, found via real UI use): the "base"
(untrained) variant occasionally burns its ENTIRE token budget inside a
Qwen3 <think> block before ever reaching the JSON answer, especially on
harder portfolio-wide queries. Mirrors a retry pattern already established
elsewhere in this codebase family (SalesIntelligence's
llm/sft/generate_candidates.py and orchestrator/pipeline.py both retry
once with a stricter instruction on parse failure).

Retry-on-unparseable, v2 (2026-09-11, same day -- v1 wasn't enough): kd
and sft ALSO hit unparseable responses in real use, and v1's generic retry
prompt sometimes failed AGAIN, because it never told the model what was
actually wrong. Root cause: common/formatting.py::parse_next_best_action
used to swallow every validation failure into a bare None -- no visibility
into whether the problem was "no JSON at all", a JSON syntax error, or
valid JSON that just didn't match the schema (wrong action_type spelling,
confidence as a string, payload as a string instead of an object). Fixed
in two layers:
  1. common/formatting.py now has a deterministic REPAIR pass
     (repair_next_best_action_dict) for the schema-mismatch cases above --
     applied automatically inside parse_next_best_action, so most of those
     failures now succeed on the FIRST attempt with no retry needed at all.
  2. If a response still doesn't parse even after repair, this module now
     retries with the SPECIFIC diagnosed reason embedded in the prompt
     (via parse_next_best_action_diagnostic), not a generic "try again" --
     the model is told exactly what was wrong, not left to guess.
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


def _retry_prompt(user_prompt: str, reason: str) -> str:
    return (
        f"{user_prompt}\n\n"
        f"Your previous response could not be used: {reason}\n"
        "Respond again, this time with ONLY a single valid JSON object matching "
        "the NextBestAction schema above -- no <think> block, no markdown "
        "fences, no commentary before or after it. Every field is required: "
        "action_type, target_object, payload, rationale, risk_flags, confidence."
    )


def timed_generate(
    handle, system_prompt: str, user_prompt: str, max_tokens: int = 700, retry_on_unparseable: bool = True
) -> dict:
    """Run one generation, returning {"text", "elapsed_seconds",
    "output_tokens", "tokens_per_second", "peak_vram_mb", "retried",
    "parse_error"}. Uses the loaded model's own tokenizer
    (handle._llm.tokenize) to count output tokens exactly, rather than
    estimating from character count.

    When retry_on_unparseable is True (the default): the first attempt is
    diagnosed via common/formatting.py::parse_next_best_action_diagnostic,
    which auto-repairs common schema mismatches before ever giving up (see
    module docstring) -- most malformed-but-salvageable responses succeed
    here with NO retry needed. Only if that still fails does this retry
    ONCE, with the SPECIFIC diagnosed reason embedded in the prompt (not a
    generic "try again"). Timing and token counts reflect BOTH attempts
    combined when a retry happens -- the real cost of getting a usable
    answer, not just the second call. "retried" tells the caller this
    happened; "parse_error" carries the final diagnosed reason if it's
    STILL unparseable after the retry, so a UI can show something more
    useful than a blank "could not parse"."""
    from common.formatting import parse_next_best_action_diagnostic  # local import: avoids a hard dependency for pure perf-timing callers

    vram_before = _gpu_memory_used_mb()
    start = time.time()
    text = handle.generate(system_prompt, user_prompt, max_tokens=max_tokens)
    _, parse_error = parse_next_best_action_diagnostic(text)

    retried = False
    if retry_on_unparseable and parse_error is not None:
        retried = True
        text = handle.generate(system_prompt, _retry_prompt(user_prompt, parse_error), max_tokens=max_tokens)
        _, parse_error = parse_next_best_action_diagnostic(text)

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
        "parse_error": parse_error,
    }
