"""Private gamma8-only kernels; original common compiler, budgets and fences."""
from .preflight import verify_components

verify_components()

from .vspec_even_static_worker import ReadOnlyEvenStaticWorker
from .campaign_fast_observer import ReadOnlyEvenFastWorker as ActivityObserver
from ..experimental_hooks.gamma8_input_domain import install as install_gamma8_inputs
from ..experimental_hooks.gamma8_rgdr_hook import install as install_gamma8_rgdr
from ..experimental_hooks.gamma6_tail_hook import install as install_gamma6_tail
from ..experimental_hooks.gamma6_mask_hook import install as install_gamma6_mask

install_gamma8_inputs()
install_gamma8_rgdr()
install_gamma6_tail()
install_gamma6_mask()

class CandidateWorker(ReadOnlyEvenStaticWorker):
    def get_private_even_fast_state(self):
        return ActivityObserver.get_private_even_fast_state(self)

    def issue18_campaign_state(self):
        import os
        from vllm_ascend.compilation.acl_graph import get_graph_params
        state=self.get_private_even_fast_state();c=self.vllm_config
        owners=self.model_runner._q35_cached_conv_manager.layers
        assert len(owners)==30
        state['native_edge_convolution']=[dict(layer=name,**owner._q35_edge_conv_helper.metrics())
            for name,owner in owners.items()]
        for row in state['native_edge_convolution']:
            row['original_gamma4_and6_kernel_dispatch_unchanged']=False
            row['gamma4_unchanged_gamma6_separate_declared_hook']=True
        from ..experimental_hooks import native_stride_conv
        from pathlib import Path
        import hashlib
        source=Path(native_stride_conv.__file__).resolve()
        state['native_edge_convolution_source']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        helpers=self.model_runner._q35_rgdr_reuse_manager.layers
        assert len(helpers)==30
        state['native_gamma8_RGDR']=[dict(layer=name,**helper._private_gamma8_rgdr.metrics())
            for name,helper in helpers.items()]
        for row in state['native_gamma8_RGDR']:
            row['original_gamma4_and6_apply_fallback_unchanged']=False
            row['gamma4_unchanged_gamma6_separate_declared_hook']=True
        from ..experimental_hooks.gamma8_input_domain import snapshot
        state['gamma8_input_domain']=snapshot(self)
        from ..experimental_hooks.gamma6_tail_hook import snapshot as snapshot_gamma6_tail
        state['gamma6_tail_domain']=snapshot_gamma6_tail(self)
        from ..experimental_hooks.gamma6_mask_hook import snapshot as snapshot_gamma6_mask
        state['gamma6_post_top8_mask']=snapshot_gamma6_mask(self)
        owner=getattr(self.model_runner,'_private_prefix_clip',None)
        state['verified_prefix_bridge']=dict(enabled=False) if owner is None else owner.snapshot()
        state.update(arm=os.environ.get('HUST_ISSUE18_B16_ARM','dynamic'),compiler=self.get_vspec_readonly_compiler_state(),
            static_runtime=self.get_private_even_static_state(),
            graph_parameter_keys=sorted(get_graph_params().attn_params),
            config=dict(memory_utilization=c.cache_config.gpu_memory_utilization,
                explicit_KV_bytes=getattr(c.cache_config,'kv_cache_memory_bytes',None)),
            KV_cache_config=str(getattr(self.model_runner,'kv_cache_config',None)),
            metadata_RPC_outside_generation_timer=True)
        return state
