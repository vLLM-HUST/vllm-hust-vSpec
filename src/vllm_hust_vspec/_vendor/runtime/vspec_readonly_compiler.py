"""Additional named compiler-state observer; no model or runtime mutations."""
from .vspec_readonly_observer import ReadOnlyVSpecWorkerObserver
class ReadOnlyCompilerWorkerObserver(ReadOnlyVSpecWorkerObserver):
    def get_vspec_readonly_compiler_state(self):
        import os
        import sys
        from vllm_ascend.ascend_config import get_ascend_config
        config=self.vllm_config;ascend=get_ascend_config();compile=ascend.ascend_compilation_config
        module=sys.modules.get('npugraph_ex')
        return dict(pid=os.getpid(),cudagraph_mode=str(config.compilation_config.cudagraph_mode),
            actual_enable_npugraph_ex=bool(compile.enable_npugraph_ex),
            actual_enable_static_kernel=bool(compile.enable_static_kernel),
            compiler_module_file=getattr(module,'__file__',None),
            dtype=str(config.model_config.dtype),quantization=config.model_config.quantization,
            maximum_sequences=config.scheduler_config.max_num_seqs,
            maximum_model_length=config.model_config.max_model_len,
            all_capture_keys=list(config.compilation_config.cudagraph_capture_sizes),
            Python_metadata_only_no_tensor_reads_writes_or_resets=True)
