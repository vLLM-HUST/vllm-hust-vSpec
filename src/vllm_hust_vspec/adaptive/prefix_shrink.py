"""Consume actual two-rank accepted-prefix proofs before a live shrink."""

import os
import json

TAIL_WINDOW_STEPS = 7
MAX_TAIL_MEAN_PROGRESS = 3.0
MAX_TAIL_FENCE_ATTEMPTS = 2


def drop_unverified_cpu_suffix(requests):
    if any(request.num_output_placeholders or len(request.spec_token_ids) != 6 for request in requests):
        raise RuntimeError('CPU draft truncation requires drained native6 placeholders')
    for request in requests:
        request.spec_token_ids = request.spec_token_ids[:4]


class PrefixShrinkPhase:
    def __init__(self, parent):
        self.parent = parent
        self.poll_steps = 0
        self.pending = None
        self.used = False
        self.transition_sequence = 0
        self.host_sample = None
        self.fence_attempts = 0

    def reset_empty(self):
        self.pending, self.used = None, False
        self.host_sample = None
        self.fence_attempts = 0

    def consider(self):
        parent, scheduler = self.parent, self.parent.scheduler
        requests = parent.running()
        if not requests:
            self.reset_empty()
            return
        self.poll_steps += 1
        if self.used or self.pending is not None or self.fence_attempts >= MAX_TAIL_FENCE_ATTEMPTS:
            return
        if (not parent.used or parent.bridge_wait or parent.ready is not None or parent.pending is not None
                or not 1 <= len(requests) <= 7
                or scheduler._vspec_adaptive_controller.current_gamma != 6
                or scheduler.waiting or scheduler.skipped_waiting or scheduler.num_waiting_for_streaming_input
                or any(request.use_structured_output for request in requests)):
            self.host_sample = None
            return
        # Safety-only uniform-first-frame guard, not an output-cap override:
        # leave native handling for requests already at their terminal edge.
        if any(request.max_tokens - request.num_output_tokens < 5 for request in requests):
            self.host_sample = None
            return
        pairs = tuple(sorted((request.request_id, request.num_output_tokens) for request in requests))
        if any(type(length) is not int or length < 0 for _, length in pairs):
            self.host_sample = None
            return
        previous = self.host_sample
        if previous is None or tuple(key for key, _ in previous[1]) != tuple(key for key, _ in pairs):
            self.host_sample = self.poll_steps, pairs
            return
        elapsed = self.poll_steps - previous[0]
        if elapsed < TAIL_WINDOW_STEPS:
            return
        self.host_sample = self.poll_steps, pairs
        progress = tuple((length - old_length) / elapsed
                         for (_, length), (_, old_length) in zip(pairs, previous[1]))
        if not all(0 < rate <= MAX_TAIL_MEAN_PROGRESS for rate in progress):
            return
        self.pending = parent.roster()
        print('[q35-prefix-tail-trigger] ' + json.dumps(dict(
            requests=len(requests), window_steps=elapsed, host_verified_progress=list(progress),
            maximum_allowed_progress=MAX_TAIL_MEAN_PROGRESS, prior_fence_attempts=self.fence_attempts,
            maximum_fence_attempts=MAX_TAIL_FENCE_ATTEMPTS, no_new_GPU_read=True,
            no_labels_or_text=True, physical_fences_still_required=True)), flush=True)

    def fence(self, core):
        parent, scheduler = self.parent, self.parent.scheduler
        requests, roster = parent.running(), parent.roster()
        if (self.fence_attempts >= MAX_TAIL_FENCE_ATTEMPTS or not requests
                or roster != self.pending or core.batch_queue
                or any(request.num_output_placeholders or len(request.spec_token_ids) != 6
                       or request.max_tokens - request.num_output_tokens < 5 for request in requests)):
            self.pending = None
            return False
        self.transition_sequence += 1
        self.fence_attempts += 1
        receipts = core.model_executor.collective_rpc('fence_private_q35_prefix_shrink',
            args=(self.transition_sequence, roster))
        if (len(receipts) != 2 or len({row['pid'] for row in receipts}) != 2
                or not all(row['eligible'] for row in receipts)
                or len({row.get('mamba_block_size') for row in receipts}) != 1):
            print('[q35-prefix-shrink-rejected] ' + json.dumps(dict(transition_id=self.transition_sequence,
                  receipts=receipts)), flush=True)
            self.pending = None
            return False
        drop_unverified_cpu_suffix(requests)
        scheduler._vspec_adaptive_controller.force(4, len(requests), 0, reason='quiescent_accepted_prefix_shrink')
        parent.ready = dict(transition_id=self.transition_sequence, stage='prefix_shrink', old_gamma=6, new_gamma=4,
            queue_depth=0, placeholders_zero=True, cpu_draft_prefix_only=True,
            current_roster=list(roster), original_roster=list(roster), worker_receipts=receipts)
        self.pending, self.used = None, True
        print('[q35-prefix-shrink-fence] ' + json.dumps(parent.ready), flush=True)
        return True


def consume_prefix_shrink(runner, marker, previous_gamma, gamma):
    own = getattr(runner, '_q35_pending_prefix_shrink_fence', None)
    if (previous_gamma != 6 or gamma != 4 or not isinstance(marker, dict) or not isinstance(own, dict)
            or marker.get('stage') != 'prefix_shrink' or marker.get('old_gamma') != 6
            or marker.get('new_gamma') != 4 or marker.get('queue_depth') != 0
            or marker.get('placeholders_zero') is not True or own.get('pid') != os.getpid()
            or marker.get('transition_id') != own.get('transition_id') or not own.get('eligible')
            or marker.get('cpu_draft_prefix_only') is not True):
        raise RuntimeError('Missing native accepted-prefix shrink receipt')
    workers = marker.get('worker_receipts', [])
    if len(workers) != 2 or len({row.get('pid') for row in workers}) != 2 or own not in workers:
        raise RuntimeError('Both TP workers must fence the same shrink')
    roster = marker.get('current_roster', [])
    n = len(roster)
    if not 1 <= n <= 7 or len(set(roster)) != n:
        raise RuntimeError('Only the exact remaining small cohort can shrink')
    for row in workers:
        block_size = row.get('mamba_block_size')
        if (row.get('transition_id') != marker['transition_id'] or row.get('stage') != 'prefix_shrink'
                or not row.get('eligible') or not row.get('device_fenced')
                or row.get('old_gamma') != 6 or row.get('new_gamma') != 4
                or row.get('old_actual_width') != 7 or row.get('new_actual_width') != 5
                or row.get('previous_draft_width') != 6 or row.get('previous_draft_stride') != 6
                or len(row.get('roster', [])) != n or set(row['roster']) != set(roster)
                or row.get('cache_tensors_rewritten') is not False
                or row.get('sampled_counts_rewritten') is not False
                or row.get('previous_draft_tensor_rewritten') is not False
                or not row.get('all30_cache_layouts_validated')
                or not row.get('accepted_checkpoint_fits_new_width') or not row.get('block_boundary_excluded')
                or not row.get('previous_actual_layout_from_width_cache')
                or not row.get('bidirectional_mamba_block_boundary_excluded') or row.get('backward_guard_tokens') != 7
                or row.get('next_width_tokens') != n * 5
                or len(row.get('next_width_input_counter_snapshot', {})) != 30
                or any(type(count) is not int or count < 0 for count in row['next_width_input_counter_snapshot'].values())
                or len(row.get('accepted_counts', [])) != n
                or any(not 1 <= count <= 5 for count in row['accepted_counts'])
                or type(block_size) is not int or block_size <= 0
                or len(row.get('old_end_seq_lens', [])) != n
                or any(value <= 7 or (value - 7 - 1) // block_size != (value - 1) // block_size
                       or (value - 1) // block_size != (value + 5 - 1) // block_size
                       for value in row['old_end_seq_lens'])
                or len(row.get('groups', [])) != 3
                or {entry.get('gid') for entry in row['groups']} != {1, 2, 3}
                or not all(entry.get('previous_wide_canonical') and entry.get('accepted_state_retained_in_prefix')
                           and entry.get('next_q5_slots_allocated') and entry.get('mamba_block_size') == block_size
                           and entry.get('actual_previous_width') == 7
                           and entry.get('actual_width_cache_address_signature_validated')
                           for entry in row['groups'])):
            raise RuntimeError('Incomplete physical accepted-prefix shrink proof')
    if len({row['mamba_block_size'] for row in workers}) != 1:
        raise RuntimeError('TP ranks disagree about Mamba block geometry')
    if runner.prev_num_spec_tokens != 6 or tuple(runner._draft_token_ids.shape) != (n, 6):
        raise RuntimeError('Do not overwrite previous draft tensor/stride before native input gather')
    runner._q35_pending_prefix_shrink_fence = None
    runner._q35_last_consumed_prefix_shrink = marker['transition_id']
    print(f'[q35-prefix-shrink-worker-consumed] pid={os.getpid()} transition={marker["transition_id"]} '
          'previous_gamma=6 gamma=4 previous_draft_stride=6', flush=True)
    return 'prefix_shrink'
