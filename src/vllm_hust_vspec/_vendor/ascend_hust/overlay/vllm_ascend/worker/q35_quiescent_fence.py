"""Read-only GPU cache-prefix fence at a fully drained live-growth boundary.

Never called per decode step or during graph capture. Synchronization and
small D2H validations are intentional and counted inside generation timers.
No tensor/model/cache state is rewritten. Only an admitted private receipt is
stored on the runner for the next preparation call to consume exactly once.
"""

import os

import torch

from vllm_ascend import envs


def prepare_growth_fence(runner, transition_id, stage, expected_roster):
    if envs.VLLM_ASCEND_Q35_QUIESCENT_GROWTH not in ('validate', 'on'):
        raise RuntimeError('Live-growth fence is explicitly opt-in')
    if stage not in ('bridge', 'wide') or type(transition_id) is not int or transition_id <= 0:
        raise ValueError('Invalid live-growth transition or stage')
    runner._q35_pending_growth_fence = None
    torch.npu.synchronize()
    n = runner.input_batch.num_reqs
    roster = tuple(runner.input_batch.req_ids[:n])
    receipt = dict(eligible=False, pid=os.getpid(), transition_id=transition_id, stage=stage,
                   old_gamma=4, new_gamma=6, old_actual_width=5, new_actual_width=7,
                   roster=list(roster), device_fenced=True, cache_tensors_rewritten=False)

    def reject(reason):
        receipt['reason'] = reason
        return receipt

    if n != 14 or len(roster) != len(set(roster)) or set(roster) != set(expected_roster):
        return reject('old_worker_roster_not_exact_pinned14')
    if (not runner.use_async_spec_decode or runner.use_dcp
            or runner.cache_config.mamba_cache_mode != 'align'
            or runner.speculative_config.method != 'mtp'
            or runner.speculative_config.num_speculative_tokens != 6
            or runner.parallel_config.tensor_parallel_size != 2
            or getattr(runner.parallel_config, 'data_parallel_size', 1) != 1
            or runner.dtype != torch.bfloat16 or runner.lora_config is not None
            or getattr(runner.model_config, 'quantization', None) is not None
            or getattr(runner.model_config.hf_text_config, 'model_type', '') not in ('qwen3_5_moe', 'qwen3_5_moe_text')):
        raise RuntimeError('Live growth requires exact private Q35 BF16 async TP2 maxGamma6 geometry')
    expected_draft = 4 if stage == 'bridge' else 6
    expected_runtime = 4 if stage == 'bridge' else 6
    if getattr(runner, '_vspec_mtp_synchronized_gamma', None) != expected_runtime:
        return reject('unexpected_current_runtime_gamma')
    if stage == 'wide' and getattr(runner, '_q35_last_consumed_growth_fence', None) != (transition_id, 'bridge'):
        return reject('old_width_bridge_receipt_not_consumed')
    draft = runner._draft_token_ids
    if not isinstance(draft, torch.Tensor) or draft.shape != (n, expected_draft):
        return reject('previous_draft_tensor_not_expected_width')
    if runner.prev_num_spec_tokens != expected_draft:
        return reject('previous_draft_stride_mismatch')
    counts = runner.valid_sampled_token_count_gpu
    if not isinstance(counts, torch.Tensor) or counts.numel() < n:
        return reject('missing_device_valid_counts')
    accepted = counts.flatten()[:n].cpu().tolist()
    if any(not 1 <= value <= 5 for value in accepted):
        return reject('accepted_count_outside_old_actual_width')
    receipt['accepted_counts'] = accepted
    if len(runner.attn_groups) != 4 or any(len(group) != 1 for group in runner.attn_groups):
        return reject('unexpected_attention_group_geometry')
    builders = [(gid, runner.attn_groups[gid][0].get_metadata_builder(0)) for gid in (1, 2, 3)]
    block_sizes = [builder.kv_cache_spec.block_size for _, builder in builders]
    receipt['mamba_block_sizes_by_group'] = block_sizes
    if any(type(value) is not int or value <= 0 for value in block_sizes) or len(set(block_sizes)) != 1:
        return reject('inconsistent_or_invalid_actual_mamba_block_sizes')
    block_size = block_sizes[0]
    receipt['mamba_block_size'] = block_size
    seq = runner.seq_lens[:n]
    lengths = seq.cpu().tolist()
    receipt['old_end_seq_lens'] = lengths
    # Bridge q5, then first q7: the largest old-end increment is5+7.
    # After bridge has completed, the remaining first-q7 increment is<=7.
    increment = 12 if stage == 'bridge' else 7
    if any(value <= 0 or (value - 1) // block_size != (value + increment - 1) // block_size for value in lengths):
        return reject('mamba_block_boundary_too_close')
    groups = []
    for gid, builder in builders:
        previous = builder.spec_state_indices_tensor
        if previous.ndim != 2 or previous.shape[1] != 5 or previous.shape[0] < n:
            return reject('old_actual_gdn_builder_width_not5')
        table = runner.input_batch.block_table[gid].get_device_tensor(n)
        start = ((seq - 1) // block_size).long()
        if not bool(((start >= 0) & (start + 7 <= table.shape[1])).all().cpu().item()):
            return reject('canonical_table_column_bounds')
        columns = start[:, None] + torch.arange(7, dtype=torch.int64, device=seq.device)[None, :]
        canonical = table.gather(1, columns)
        if not bool((canonical > 0).all().cpu().item()):
            return reject('wide_state_slot_not_allocated')
        if not torch.equal(canonical[:, :5].contiguous(), previous[:n]):
            return reject('old_actual_state_indices_not_canonical_prefix')
        groups.append(dict(gid=gid, mamba_block_size=block_size, old_prefix_exact=True,
                           wide_slots_allocated=True, source_column_bounds=True))
    manager = runner._q35_gdn_input_manager
    if len(manager.layers) != 30:
        return reject('missing30_layer_geometry')
    for helper in manager.layers.values():
        conv, recurrent = helper.attention.kv_cache
        if (conv.dtype != torch.bfloat16 or tuple(conv.shape[-2:]) != (9, 4096)
                or conv.stride(-1) != 1 or recurrent.dtype != torch.float32
                or tuple(recurrent.shape[-3:]) != (16, 128, 128)):
            return reject('cache_layout_outside_native_growth_micro_proof')
    receipt.update(eligible=True, reason='ADMITTED_READ_ONLY_QUIESCENT_GROWTH',
                   accepted_counts=accepted, old_end_seq_lens=lengths, groups=groups,
                   all30_cache_layouts_validated=True, block_boundary_excluded=True,
                   previous_draft_width=expected_draft)
    runner._q35_pending_growth_fence = receipt
    return receipt
