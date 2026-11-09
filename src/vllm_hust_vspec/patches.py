"""Dispatch vSpec patches to the selected speculative backend."""

from __future__ import annotations

from .config import PluginSettings


def apply_patches(settings: PluginSettings) -> bool:
    from .compatibility import require_host_compatibility

    require_host_compatibility(
        settings.method,
        adaptive=settings.adaptive_speculation,
    )
    from .backends.kv_cache import apply_multi_layer_kv_cache_patch

    applied = apply_multi_layer_kv_cache_patch()
    if settings.adaptive_speculation:
        from .adaptive.runtime import apply_adaptive_patches

        applied = apply_adaptive_patches(settings)
        import os
        if os.environ.get('HUST_VSPEC_EVEN_ROUND_ENABLE')=='1':
            from .adaptive.even_runtime import install as install_even_round
            install_even_round(settings)
    if settings.method == "draft_model":
        from .backends.draft import apply_draft_patches

        return apply_draft_patches(settings) or applied
    if settings.method in {"eagle", "eagle3"}:
        from .backends.eagle import apply_eagle_patches

        return apply_eagle_patches(settings) or applied
    if settings.method == "dflash":
        # DFlash is provided by vLLM-Ascend; vSpec contributes the adaptive
        # control and graph-width hooks installed above.
        return applied
    if settings.method == "mtp":
        from .backends.mtp import apply_mtp_patches
        from .adaptive.quiescent import apply_quiescent_growth

        mtp_applied = apply_mtp_patches(settings)
        from .config import q35_moe_nz_layout_enabled

        if q35_moe_nz_layout_enabled():
            from .backends.q35_moe_nz_layout import install_moe_nz_layout

            install_moe_nz_layout()
        from .config import q35_inplace_head_nz_enabled

        if q35_inplace_head_nz_enabled():
            from .backends.q35_inplace_head_nz import install_inplace_head_nz

            install_inplace_head_nz()
        from .config import q35_native_conv_tile_enabled

        if q35_native_conv_tile_enabled():
            from .backends.q35_native_conv_tile import install_native_conv_tile

            install_native_conv_tile()
        from .config import q35_tail_prefix_shrink_enabled

        if q35_tail_prefix_shrink_enabled():
            from .backends.q35_prefix_shrink_rpc import install_prefix_shrink_rpc

            install_prefix_shrink_rpc()
        from .config import q35_narrow_tail_twin_enabled

        if q35_narrow_tail_twin_enabled():
            from .backends.q35_narrow_tail_twin import install_narrow_tail_twin

            install_narrow_tail_twin()
        from .config import q35_two_bank_embedding_enabled

        if q35_two_bank_embedding_enabled():
            from .backends.q35_two_bank_embedding import install_two_bank_embedding

            install_two_bank_embedding()
        from .config import q35_hybrid_tail_enabled

        if q35_hybrid_tail_enabled():
            from .backends.q35_hybrid_tail import install_hybrid_tail

            install_hybrid_tail()
        from .config import q35_owner_mid_twin_enabled

        if q35_owner_mid_twin_enabled():
            from .backends.q35_owner_mid_twin import install_owner_mid_twin

            install_owner_mid_twin()
        from .config import q35_fused_proposal_vote_enabled

        if q35_fused_proposal_vote_enabled():
            from .backends.q35_fused_proposal_vote import install_fused_proposal_vote

            install_fused_proposal_vote()
        phase_applied = apply_quiescent_growth(settings)
        from .adaptive.prefix_bridge_runtime import install as install_verified_prefix_bridge
        install_verified_prefix_bridge(settings)
        return phase_applied or mtp_applied or applied
    raise ValueError(f"unsupported vSpec method: {settings.method}")
