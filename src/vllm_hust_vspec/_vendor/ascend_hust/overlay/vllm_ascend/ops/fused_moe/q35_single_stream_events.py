"""Default-off removal of unused same-stream BF16 shared-expert markers.

All routed/shared tensor operations and communication remain ordered exactly
as before. Admission is constructor-time CPU metadata only. The per-call gate
retains every original event for non-ALLGATHER communication or multistream
execution; it never changes target/draft graph-update fences.
"""

import os


def admit(runner):
    from vllm_ascend import envs

    mode = envs.VLLM_ASCEND_Q35_SINGLE_STREAM_MOE_EVENTS
    if mode not in ('off', 'validate', 'on'):
        raise ValueError('Single-stream MoE events mode must be off,validate,on')
    if mode == 'off':
        return False
    import torch
    from vllm.config import get_current_vllm_config
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.ops.fused_moe.shared_experts import SharedExpertParallelMode

    config = get_current_vllm_config()
    parallel, model = config.parallel_config, config.model_config
    shared = runner.ascend_shared_experts
    if (getattr(model.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')
            or model.dtype != torch.bfloat16 or getattr(model, 'quantization', None) is not None
            or config.lora_config is not None or config.speculative_config is None
            or config.speculative_config.method != 'mtp'
            or parallel.tensor_parallel_size != 2 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or not parallel.enable_expert_parallel
            or runner.quant_type != QuantType.NONE or runner.hidden_size != 2048
            or runner.moe_config.num_experts != 256 or runner.moe_config.num_local_experts != 128
            or runner.moe_config.experts_per_token != 8 or runner.moe_config.is_sequence_parallel
            or shared is None or shared.multistream_overlap or shared.weights_replicated
            or shared.parallel_mode() is not SharedExpertParallelMode.TENSOR_PARALLEL
            or shared.quant_type != QuantType.NONE):
        raise RuntimeError('Private event elision requires unquantized Q35 TP2/EP2/DP1/PP1 serial shared experts')
    print(f'[q35-single-stream-moe-admitted] pid={os.getpid()} mode={mode} '
          'tensor_math_and_communication_unchanged=True cross_stream_fences_retained=True', flush=True)
    return True


def skip_events(component):
    if not getattr(component, '_q35_single_stream_events', False):
        return False
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

    # A different communication policy must fall back to all native events.
    return _EXTRA_CTX.moe_comm_type is MoECommType.ALLGATHER
