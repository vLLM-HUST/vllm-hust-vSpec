"""Standalone native MTP4 worker extension for matched evaluation.

The native child imports this file from its installed asset directory on
PYTHONPATH. Keep every import here top-level so loading this extension never
imports ``vllm_hust_vspec`` or registers the dynamic plugin.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

if os.environ.get("HUST_VSPEC_CAPTURE_SAFE_STATIC") != "1":
    raise RuntimeError("Native evaluation requires capture-safe static compilation")
if os.environ.get("HUST_VSPEC_HOT_STATIC_SHAPES") != "14,42,70,98,126":
    raise RuntimeError("Native and dynamic evaluation require the same hot static shapes")

import torch_npu  # noqa: E402
from capture_safe_static import install as install_capture_guard  # noqa: E402
from evaluation_worker_common import EvaluationMetadataMixin  # noqa: E402
from hot_static_options import CALLS, HOT  # noqa: E402
from hot_static_options import install as install_hot_options
from npugraph_ex._acl_concrete_graph import acl_graph, static_kernel  # noqa: E402
from npugraph_ex.configs import compiler_config  # noqa: E402
from static_install_metadata import snapshot  # noqa: E402
from vspec_readonly_static_worker import ReadOnlyStaticKernelWorkerObserver  # noqa: E402

_ORIGINAL_COMPILE, _STATISTICS = install_capture_guard(
    acl_graph, torch_npu.npu.is_current_stream_capturing
)
_ORIGINAL_OPTIONS = install_hot_options(compiler_config)


def _require_native_mtp4(worker):
    import torch.distributed as dist

    root = Path(os.environ["HUST_VSPEC_PRIVATE_COMPILER_ROOT"]).resolve(strict=True)
    if Path.cwd().resolve() != root or dist.get_world_size() != 2:
        raise RuntimeError("Native static evaluation requires two workers in the private run")
    if any(
        name == "vllm_hust_vspec" or name.startswith("vllm_hust_vspec.") for name in sys.modules
    ):
        raise RuntimeError("Native worker imported the vSpec plugin")
    if worker.vllm_config.speculative_config.num_speculative_tokens != 4:
        raise RuntimeError("Native evaluation must use MTP4")
    return root, dist.get_rank()


class NativeEvaluationWorker(EvaluationMetadataMixin, ReadOnlyStaticKernelWorkerObserver):
    def get_eval_compiler_state(self):
        return self.get_vspec_readonly_compiler_state()

    def get_eval_static_state(self):
        root, rank = _require_native_mtp4(self)
        if _STATISTICS["static_compilation_deferred_inside_capture"] <= 0 or not CALLS:
            raise RuntimeError("Native capture-safe static compilation was not exercised")
        for row in CALLS:
            available = row["original_options"]["options"]["_vllm_aclnn_static_kernel_sym_range"]
            selected = row["selected_options"]["options"]["_vllm_aclnn_static_kernel_sym_range"]
            if selected != [shape for shape in HOT if shape in available]:
                raise RuntimeError("Native static shape selection differs from shared rule")
        return {
            "compiler": self.get_eval_compiler_state(),
            "static_install": snapshot(static_kernel, root, rank, os.getpid()),
            "capture_safe_statistics": dict(_STATISTICS),
            "compiler_option_calls": deepcopy(CALLS),
            "fixed_gamma": 4,
            "vspec_plugin_absent": True,
            "common_hot_shape_rule": list(HOT),
            "FX_math_and_all_capture_keys_unmodified": True,
        }

    def get_eval_operator_proof(self):
        from vllm_ascend.compilation.acl_graph import get_graph_params

        static = self.get_eval_static_state()
        config = self.vllm_config
        return {
            "arm": "native",
            "software_paths": self.get_vspec_readonly_software_paths(),
            "compiler": static["compiler"],
            "static_runtime": static,
            "graph_parameter_keys": sorted(get_graph_params().attn_params),
            "config": {
                "memory_utilization": config.cache_config.gpu_memory_utilization,
                "explicit_KV_bytes": config.cache_config.kv_cache_memory_bytes,
            },
            "metadata_RPC_outside_generation_timer": True,
        }
