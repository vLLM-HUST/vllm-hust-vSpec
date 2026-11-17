"""Read-only gamma6->4 fence; accepted checkpoint must fit prefix q5.

Called only after queue draining. No model, state, counts or draft tensor is
rewritten. Keeping the previous draft stride6 is mandatory for native input
gather after CPU scheduling discards the two unverified suffix candidates.
"""

import os

import torch

from vllm_ascend import envs


def prepare_prefix_shrink(runner, transition_id, expected_roster):
    if envs.VLLM_ASCEND_Q35_QUIESCENT_GROWTH not in ('validate', 'on'):
        raise RuntimeError('Prefix shrink is explicitly opt-in')
    if type(transition_id) is not int or transition_id <= 0:
        raise ValueError('Positive shrink transition required')
    runner._q35_pending_prefix_shrink_fence = None
    torch.npu.synchronize()
    n = runner.input_batch.num_reqs
    roster = tuple(runner.input_batch.req_ids[:n])
    proof = dict(eligible=False, pid=os.getpid(), transition_id=transition_id, stage='prefix_shrink',
        old_gamma=6, new_gamma=4, old_actual_width=7, new_actual_width=5,
        roster=list(roster), device_fenced=True, cache_tensors_rewritten=False,
        sampled_counts_rewritten=False, previous_draft_tensor_rewritten=False)

    def reject(reason):
        proof['reason'] = reason
        return proof

    if not 1 <= n <= 7 or len(set(roster)) != n or len(expected_roster) != n or set(expected_roster) != set(roster):
        return reject('worker_roster_not_exact_current_small_cohort')
    if (not runner.use_async_spec_decode or runner.use_dcp or runner.cache_config.mamba_cache_mode != 'align'
            or runner.speculative_config.method != 'mtp' or runner.speculative_config.num_speculative_tokens != 6
            or runner.parallel_config.tensor_parallel_size != 2
            or getattr(runner.parallel_config, 'data_parallel_size', 1) != 1
            or runner.dtype != torch.bfloat16 or runner.lora_config is not None
            or getattr(runner.model_config, 'quantization', None) is not None
            or getattr(runner.model_config.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')):
        raise RuntimeError('Prefix shrink requires pinned native BF16 Q35 async TP2 gamma6 geometry')
    if getattr(runner, '_vspec_mtp_synchronized_gamma', None) != 6:
        return reject('runtime_not_wide6')
    refresh = getattr(runner, '_q35_fused_gdn_refresh', None)
    if refresh is None or refresh.last_gamma != 6:
        return reject('last_actual_target_metadata_not_gamma6')
    signatures = refresh.storage_signatures.get(7)
    if not isinstance(signatures, tuple) or len(signatures) != 3:
        return reject('missing_actual_q7_captured_storage_signature')
    if not isinstance(runner._draft_token_ids, torch.Tensor) or runner._draft_token_ids.shape != (n, 6):
        return reject('previous_draft_tensor_not_native6')
    if runner.prev_num_spec_tokens != 6:
        return reject('previous_draft_stride_not12')
    counts = runner.valid_sampled_token_count_gpu
    if not isinstance(counts, torch.Tensor) or counts.numel() < n:
        return reject('missing_device_accepted_counts')
    accepted = counts.flatten()[:n].cpu().tolist()
    proof['accepted_counts'] = accepted
    if any(not 1 <= count <= 5 for count in accepted):
        return reject('accepted_checkpoint_outside_new_q5_prefix')
    if len(runner.attn_groups) != 4 or any(len(group) != 1 for group in runner.attn_groups):
        return reject('unexpected_attention_groups')
    builders = [(gid, runner.attn_groups[gid][0].get_metadata_builder(0)) for gid in (1, 2, 3)]
    block_sizes = [builder.kv_cache_spec.block_size for _, builder in builders]
    if any(type(size) is not int or size <= 0 for size in block_sizes) or len(set(block_sizes)) != 1:
        return reject('inconsistent_actual_mamba_block_sizes')
    block_size = block_sizes[0]
    seq = runner.seq_lens[:n]
    lengths = seq.cpu().tolist()
    # Old q7 optimistic lengths can move backward after rejected candidates
    # are corrected. Exclude the entire old-stride backward interval as well
    # as the new q5 forward interval; no block remapping is admitted here.
    if any(value <= 7 or (value - 7 - 1) // block_size != (value - 1) // block_size
           or (value - 1) // block_size != (value + 5 - 1) // block_size for value in lengths):
        return reject('prefix_shrink_near_bidirectional_mamba_boundary')
    groups = []
    for index, (gid, builder) in enumerate(builders):
        host_bound_width = builder.spec_state_indices_tensor.shape[1]
        # The width context restores the HOST q5 view after executing q7.
        # Read the stable actual-width buffer, not that restored host view.
        previous = getattr(builder, '_vspec_adaptive_spec_state_indices_by_width', {}).get(7)
        if not isinstance(previous, torch.Tensor):
            return reject('missing_previous_actual_q7_width_cache')
        if previous.ndim != 2 or previous.shape[1] != 7 or previous.shape[0] < n:
            return reject('previous_actual_state_layout_not_q7')
        signature = signatures[index]
        if signature[0] != id(builder) or signature[1][0] != (previous.data_ptr(), tuple(previous.shape), previous.dtype):
            return reject('actual_q7_storage_address_signature_mismatch')
        table = runner.input_batch.block_table[gid].get_device_tensor(n)
        start = ((seq - 1) // block_size).long()
        if not bool(((start >= 0) & (start + 7 <= table.shape[1])).all().cpu().item()):
            return reject('canonical_source_columns_out_of_bounds')
        columns = start[:, None] + torch.arange(7, dtype=torch.int64, device=seq.device)[None, :]
        canonical = table.gather(1, columns)
        if not torch.equal(canonical, previous[:n]) or not bool((canonical[:, :5] > 0).all().cpu().item()):
            return reject('current_wide_state_not_canonical_prefix')
        groups.append(dict(gid=gid, mamba_block_size=block_size, previous_wide_canonical=True,
                           accepted_state_retained_in_prefix=True, next_q5_slots_allocated=True,
                           host_bound_query_width=host_bound_width, actual_previous_width=7,
                           actual_width_cache_address_signature_validated=True))
    manager = runner._q35_gdn_input_manager
    if len(manager.layers) != 30:
        return reject('missing30_gdn_layers')
    for helper in manager.layers.values():
        conv, recurrent = helper.attention.kv_cache
        if (conv.dtype != torch.bfloat16 or tuple(conv.shape[-2:]) != (9, 4096)
                or conv.stride(-1) != 1 or recurrent.dtype != torch.float32
                or tuple(recurrent.shape[-3:]) != (16, 128, 128)):
            return reject('cache_layout_outside_native_prefix_shrink_proof')
    # One batched D2H at the already-drained fence, never per decode step.
    # Final after-timer counters must exceed this snapshot for the new q5.
    layer_names = list(manager.layers)
    counter_values = torch.stack([manager.layers[name].counts[n * 5] for name in layer_names]).cpu().tolist()
    proof.update(eligible=True, reason='ADMITTED_READ_ONLY_ACCEPTED_PREFIX_SHRINK',
        previous_draft_width=6, previous_draft_stride=6, mamba_block_size=block_size,
        old_end_seq_lens=lengths, all30_cache_layouts_validated=True,
        accepted_checkpoint_fits_new_width=True, block_boundary_excluded=True, groups=groups,
        next_width_tokens=n * 5,
        previous_actual_layout_from_width_cache=True,
        bidirectional_mamba_block_boundary_excluded=True, backward_guard_tokens=7,
        next_width_input_counter_snapshot=dict(zip(layer_names, counter_values)))
    runner._q35_pending_prefix_shrink_fence = proof
    return proof
