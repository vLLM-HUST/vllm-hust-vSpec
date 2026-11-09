"""Read native MoE execution-path metadata after an evaluation timing window."""

import os


class ReadOnlyVSpecWorkerObserver:
    def get_vspec_readonly_software_paths(self):
        import torch
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
        from vllm_ascend.quantization.quant_type import QuantType

        rows = []
        for role, model in (
            ("target", self.model_runner.model),
            ("draft", self.model_runner.drafter.model),
        ):
            for _, owner in model.named_modules():
                if not isinstance(owner, AscendMoERunner):
                    continue
                shared = owner.ascend_shared_experts
                rows.append(
                    {
                        "role": role,
                        "layer": owner.layer_name,
                        "actual_multistream_overlap": bool(shared.multistream_overlap),
                        "shared_weights_replicated": bool(shared.weights_replicated),
                        "original_same_and_cross_stream_events_restored": (
                            not getattr(owner, "_q35_single_stream_events", False)
                            and not getattr(shared, "_q35_single_stream_events", False)
                        ),
                        "incompatible_serial_cache_absent": (
                            not hasattr(owner, "_q35_immutable_cache")
                            and not hasattr(shared, "_q35_immutable_cache")
                        ),
                        "native_unquantized_BF16": (
                            owner.quant_type == QuantType.NONE
                            and owner.gate.weight.dtype == torch.bfloat16
                        ),
                    }
                )
        if sum(row["role"] == "target" for row in rows) != 40:
            raise RuntimeError("Native target MoE layer count differs")
        if sum(row["role"] == "draft" for row in rows) != 1:
            raise RuntimeError("Native draft MoE layer count differs")
        if not all(row["native_unquantized_BF16"] for row in rows):
            raise RuntimeError("Native MoE paths are not unquantized BF16")
        return {
            "pid": os.getpid(),
            "all41_actual_component_flags_read": True,
            "Python_metadata_only_no_tensor_reads_writes_resets_or_stream_calls": True,
            "rows": rows,
        }
