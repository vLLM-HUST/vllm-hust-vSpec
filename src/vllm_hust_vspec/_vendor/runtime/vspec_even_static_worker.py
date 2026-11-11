"""Private capture-safe static setup and named post-timer metadata RPC."""
import os
from pathlib import Path
import torch_npu
from .vspec_even_fast_worker import ReadOnlyEvenFastWorker
from .vspec_readonly_static_worker import ReadOnlyStaticKernelWorkerObserver
from npugraph_ex.configs import compiler_config
from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel
from .capture_safe_static import install as install_capture_guard
from .hot_static_options import install, HOT, CALLS
from .static_install_metadata import snapshot
from .router_weight_cache import install as install_router_weight_cache
from ..experimental_hooks.native_stride_conv import install_native_conv_tile
from .install_native_fia_host_batch_max8 import install as install_native_fia_host_batch

install_router_weight_cache()
install_native_conv_tile()
install_native_fia_host_batch()

assert os.environ['HUST_VSPEC_CAPTURE_SAFE_STATIC']=='1'
assert os.environ.get('HUST_VSPEC_HOT_STATIC_SHAPES','70')==','.join(str(v) for v in HOT)
_ORIGINAL,_STATISTICS=install_capture_guard(acl_graph,torch_npu.npu.is_current_stream_capturing)
install(compiler_config)

class ReadOnlyEvenStaticWorker(ReadOnlyEvenFastWorker,ReadOnlyStaticKernelWorkerObserver):
    def get_private_replay_tail_state(self):
        from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers
        from .replay_tail_snapshot import snapshot as tail_snapshot
        return dict(pid=os.getpid(),rows=tail_snapshot(list(_acl_graph_wrappers)))

    def get_private_even_static_state(self):
        import torch.distributed as dist
        root=Path(os.environ['HUST_VSPEC_PRIVATE_COMPILER_ROOT']).resolve()
        assert Path.cwd().resolve()==root and dist.get_world_size()==2
        assert _STATISTICS['static_compilation_deferred_inside_capture']>0 and CALLS
        metadata=snapshot(static_kernel,root,dist.get_rank(),os.getpid())
        for row in CALLS:
            assert row['selected_options']['options']['_vllm_aclnn_static_kernel_sym_range']==list(HOT)
        return dict(compiler=self.get_vspec_readonly_compiler_state(),static=metadata,
            capture_safe_statistics=dict(_STATISTICS),compiler_option_calls=list(CALLS),
            predetermined_hot_shapes=list(HOT),FX_math_and_all_capture_keys_unmodified=True)
