"""Out-of-tree model registrations supplied by vSpec."""

from __future__ import annotations

QWEN2_EAGLE_ARCHITECTURES = (
    "Qwen2ForCausalLMEagle",
    "EagleQwen2ForCausalLMEagle",
    "EagleQwen2ForCausalLM",
)


def register_models() -> tuple[str, ...]:
    """Register models removed from the current vLLM-HUST core tree."""
    from vllm import ModelRegistry

    registered: list[str] = []
    supported = ModelRegistry.get_supported_archs()
    model_path = "vllm_hust_vspec.models.qwen2_eagle:Qwen2ForCausalLMEagle"
    for architecture in QWEN2_EAGLE_ARCHITECTURES:
        if architecture in supported:
            continue
        ModelRegistry.register_model(architecture, model_path)
        registered.append(architecture)
    return tuple(registered)


__all__ = ["QWEN2_EAGLE_ARCHITECTURES", "register_models"]
