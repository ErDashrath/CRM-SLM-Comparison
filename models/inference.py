"""
Registry-driven local inference wrapper for the CRM-SLM-Comparison POC.

Every component in this project (eval harness, UI) resolves a model variant
by name ("base" / "kd" / "sft") through load() below, which reads
models/registry.yaml -- never a hardcoded GGUF path. This is what makes the
project modular: adding, removing, or repointing a variant is a one-line
change to registry.yaml, not a code change.

Mirrors the VRAM-safe construction pattern already proven out in
~/Magna/SalesIntelligence/llm/provider.py::LocalLlamaCppBackend (same n_ctx
rationale, same quantize_kv/flash_attn coupling) -- see that file's docstring
for the full empirical justification on this exact machine. Duplicated
rather than imported across projects, deliberately, so this project stays
fully decoupled from SalesIntelligence's live repo.

The 4GB T2000 can hold exactly one of these ~4B GGUFs resident at a time --
callers comparing multiple variants MUST call .close() on one VariantHandle
before load()-ing the next.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.yaml"

GGML_TYPE_Q8_0 = 8  # ggml type enum value, see LocalLlamaCppBackend docstring in SalesIntelligence


class VariantNotBuiltError(RuntimeError):
    """Raised when a registry entry's gguf_path doesn't exist yet -- e.g.
    load("kd") before Phase 3's training + merge_and_quantize.sh has run."""


class VariantHandle:
    """One loaded model variant. generate() matches the (system_prompt,
    user_prompt, max_tokens) -> str signature used throughout this
    project's eval harness and UI, so callers never need to know which
    variant they're holding."""

    def __init__(self, name: str, label: str, llm):
        self.name = name
        self.label = label
        self._llm = llm

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512
    ) -> str:
        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return result["choices"][0]["message"]["content"]

    def generate_stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512
    ) -> Iterator[str]:
        """Yield final-answer text fragments from the resident local model."""
        stream = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content

    def close(self) -> None:
        """Release VRAM before loading a different variant -- required on
        this 4GB card, see module docstring."""
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()
        self._llm = None


def _load_registry(registry_path: Optional[Path] = None) -> dict:
    path = registry_path or DEFAULT_REGISTRY_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def list_variants(registry_path: Optional[Path] = None) -> list[str]:
    """Every variant name currently defined in registry.yaml, in file order."""
    return list(_load_registry(registry_path)["variants"].keys())


def load(variant_name: str, registry_path: Optional[Path] = None) -> VariantHandle:
    """Load one variant by name, per registry.yaml. Raises
    VariantNotBuiltError with a clear message if the variant's GGUF doesn't
    exist yet (expected for "kd"/"sft" before Phase 3 has run)."""
    from llama_cpp import Llama  # lazy import: keeps yaml-only callers (e.g. report scripts) light

    registry = _load_registry(registry_path)
    if variant_name not in registry["variants"]:
        raise KeyError(
            f"Unknown variant {variant_name!r}. Defined variants: "
            f"{list(registry['variants'].keys())}"
        )

    entry = registry["variants"][variant_name]
    gguf_path = Path(entry["gguf_path"])
    if not gguf_path.is_absolute():
        gguf_path = PROJECT_ROOT / gguf_path

    if not gguf_path.exists():
        raise VariantNotBuiltError(
            f"Variant {variant_name!r} points at {gguf_path}, which doesn't "
            "exist yet. If this is 'kd' or 'sft', run Phase 3's training + "
            "training/merge_and_quantize.sh first -- these variants have no "
            "GGUF until an adapter has been trained and merged."
        )

    cfg = registry["inference"]
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=cfg["n_gpu_layers"],
        n_ctx=cfg["n_ctx"],
        verbose=False,
        flash_attn=cfg["quantize_kv"],
        type_k=GGML_TYPE_Q8_0 if cfg["quantize_kv"] else None,
        type_v=GGML_TYPE_Q8_0 if cfg["quantize_kv"] else None,
        offload_kqv=cfg["offload_kqv"],
    )
    return VariantHandle(name=variant_name, label=entry["label"], llm=llm)


if __name__ == "__main__":
    # Phase 0 smoke test: load "base" and confirm a real completion comes back.
    handle = load("base")
    try:
        reply = handle.generate(
            system_prompt="You are a concise assistant.",
            user_prompt="In one sentence, what is a CRM?",
            max_tokens=64,
        )
        print(f"[{handle.name}] {reply}")
    finally:
        handle.close()
