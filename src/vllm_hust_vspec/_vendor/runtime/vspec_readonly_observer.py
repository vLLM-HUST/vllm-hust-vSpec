"""Named worker-extension RPC; read Python metadata AFTER fixed timers only.

No constructor, overrides, model writes, tensor reads, stream calls, resets,
or insecure function serialization. Framework checks extension name conflicts.
"""
class ReadOnlyVSpecWorkerObserver:
    def get_vspec_readonly_software_paths(self):
        import os
        import torch
        from vllm_ascend.quantization.quant_type import QuantType
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
        rows=[]
        for role,model in (('target',self.model_runner.model),('draft',self.model_runner.drafter.model)):
            for _,owner in model.named_modules():
                if not isinstance(owner,AscendMoERunner):continue
                shared=owner.ascend_shared_experts
                rows.append(dict(role=role,layer=owner.layer_name,
                    actual_multistream_overlap=bool(shared.multistream_overlap),
                    shared_weights_replicated=bool(shared.weights_replicated),
                    original_same_and_cross_stream_events_restored=not owner._q35_single_stream_events and not shared._q35_single_stream_events,
                    incompatible_serial_cache_absent=not hasattr(owner,'_q35_immutable_cache') and not hasattr(shared,'_q35_immutable_cache'),
                    native_unquantized_BF16=owner.quant_type==QuantType.NONE and owner.gate.weight.dtype==torch.bfloat16))
        assert sum(r['role']=='target' for r in rows)==40 and sum(r['role']=='draft' for r in rows)==1
        assert all(r['native_unquantized_BF16'] for r in rows)
        return dict(pid=os.getpid(),all41_actual_component_flags_read=True,
            Python_metadata_only_no_tensor_reads_writes_resets_or_stream_calls=True,rows=rows)
