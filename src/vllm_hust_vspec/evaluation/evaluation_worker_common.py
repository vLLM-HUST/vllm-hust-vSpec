"""Read-only worker RPC shared by native and dynamic evaluation extensions.

This standalone module deliberately has no vllm_hust_vspec import: the native
service may load it from this installed directory without loading the plugin.
"""

import os
import sys


class EvaluationMetadataMixin:
    def get_eval_metadata(self):
        config = self.vllm_config
        runner = self.model_runner
        arm = os.environ.get("VSPEC_EVAL_ARM")
        if arm not in {"native", "dynamic"}:
            raise RuntimeError("VSPEC_EVAL_ARM must identify the evaluation arm")
        speculative = config.speculative_config
        capacity = None if speculative is None else speculative.num_speculative_tokens
        native_capacity = int(os.environ.get("VSPEC_EVAL_NATIVE_CAPACITY", "4"))
        if native_capacity not in (4, 6, 8):
            raise RuntimeError("VSPEC_EVAL_NATIVE_CAPACITY must be 4, 6, or 8")
        loaded = any(
            name == "vllm_hust_vspec" or name.startswith("vllm_hust_vspec.") for name in sys.modules
        )
        if arm == "native" and (loaded or capacity != native_capacity):
            raise RuntimeError(f"Native evaluation loaded vSpec or is not MTP{native_capacity}")
        if arm == "dynamic" and (not loaded or capacity != 8):
            raise RuntimeError("Dynamic evaluation did not load vSpec capacity 8")
        cache = config.cache_config
        if arm == "dynamic" and (
            cache.mamba_cache_mode != "align" or not runner.use_async_spec_decode or runner.use_dcp
        ):
            raise RuntimeError("Dynamic live-fence preconditions are absent")
        compiler = self.get_eval_compiler_state()
        static = self.get_eval_static_state()
        result = {
            "pid": os.getpid(),
            "arm": arm,
            "dtype": str(config.model_config.dtype),
            "quantization": config.model_config.quantization,
            "max_model_len": config.model_config.max_model_len,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "configured_speculative_capacity": capacity,
            "vspec_module_loaded": loaded,
            "explicit_kv_bytes": cache.kv_cache_memory_bytes,
            "prefix_cache": cache.enable_prefix_caching,
            "mamba_cache_mode": cache.mamba_cache_mode,
            "runner_async_spec": runner.use_async_spec_decode,
            "runner_dcp": runner.use_dcp,
            "model_text_type": getattr(config.model_config.hf_text_config, "model_type", ""),
            "compiler": compiler,
            "static": static,
            "metadata_only_not_a_numerical_parity_proof": True,
        }
        if arm == "dynamic":
            activity = self.get_private_even_width_state()
            result["dynamic_activity"] = activity
        return result
