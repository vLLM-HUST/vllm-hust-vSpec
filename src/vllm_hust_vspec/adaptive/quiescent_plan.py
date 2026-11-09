"""Private, default-off monotonic growth invariants, independently CPU tested.

Growth is monotonic within a live cohort. Shrinking a live GDN width needs a
separate accepted-state commit proof and is explicitly out of this prototype.
Native engine draining and first-frame placeholder behavior are tested against
their actual source AST, not a reimplementation of those methods.
"""


def suffix_is_periodic(tokens, window=32, max_period=8):
    if len(tokens) < window or window < 2 or not 1 <= max_period < window:
        return False
    tail = tokens[-window:]
    if any(not isinstance(token, int) or token < 0 for token in tail):
        return False
    return any(all(tail[index] == tail[index - period] for index in range(period, window))
               for period in range(1, max_period + 1))


def growth_readiness(old_gamma, new_gamma, queue_depth, placeholders, old_roster, current_roster,
                     *, pending_prefill=False, structured=False, pipeline_parallel=1, data_parallel=1):
    if old_gamma != 4 or new_gamma not in (6, 8, 12) or new_gamma <= old_gamma:
        return False
    if queue_depth != 0 or not placeholders or any(value != 0 for value in placeholders):
        return False
    if len(old_roster) != len(placeholders) or tuple(old_roster) != tuple(current_roster):
        return False
    if len(set(old_roster)) != len(old_roster):
        return False
    return not pending_prefill and not structured and pipeline_parallel == data_parallel == 1


def accepted_state_prefix_preserved(old_table, new_table, accepted):
    if not old_table or len(old_table) != len(new_table) or len(accepted) != len(old_table):
        return False
    for old, new, count in zip(old_table, new_table, accepted):
        if len(new) <= len(old) or new[:len(old)] != old or not 1 <= count <= len(old):
            return False
        if old[count - 1] != new[count - 1]:
            return False
    return True


def consume_worker_fence(runner, marker, previous_gamma, gamma):
    """Admit only a matching real worker fence, never a bare scheduler flag."""
    import os
    own = getattr(runner, '_q35_pending_growth_fence', None)
    if (not isinstance(marker, dict) or not isinstance(own, dict)
            or marker.get('queue_depth') != 0 or marker.get('placeholders_zero') is not True
            or marker.get('old_gamma') != 4 or marker.get('new_gamma') != gamma or gamma != 6
            or marker.get('stage') not in ('bridge', 'wide')
            or marker.get('transition_id') != own.get('transition_id')
            or marker.get('stage') != own.get('stage') or not own.get('eligible')
            or own.get('pid') != os.getpid()):
        raise RuntimeError('Missing actual quiescent worker-fence receipt')
    workers = marker.get('worker_receipts', [])
    if (len(workers) != 2 or len({row.get('pid') for row in workers}) != 2
            or not all(row.get('eligible') and row.get('device_fenced') and row.get('block_boundary_excluded')
                       and row.get('stage') == marker['stage'] and row.get('transition_id') == marker['transition_id']
                       for row in workers) or own not in workers):
        raise RuntimeError('Both TP workers must prove the same quiescent transition')
    for row in workers:
        block_size = row.get('mamba_block_size')
        increment = 12 if marker['stage'] == 'bridge' else 7
        if (not row.get('all30_cache_layouts_validated') or row.get('old_gamma') != 4
                or row.get('new_gamma') != 6 or row.get('old_actual_width') != 5
                or row.get('new_actual_width') != 7 or len(row.get('accepted_counts', [])) != 14
                or any(not 1 <= count <= 5 for count in row['accepted_counts'])
                or {entry.get('gid') for entry in row.get('groups', [])} != {1, 2, 3}
                or len(row['groups']) != 3
                or type(block_size) is not int or block_size <= 0
                or len(row.get('old_end_seq_lens', [])) != 14
                or any(value <= 0 or (value - 1) // block_size != (value + increment - 1) // block_size
                       for value in row['old_end_seq_lens'])
                or not all(entry.get('old_prefix_exact') and entry.get('wide_slots_allocated')
                           and entry.get('source_column_bounds') and entry.get('mamba_block_size') == block_size
                           for entry in row['groups'])):
            raise RuntimeError('Incomplete physical cache-prefix proof')
    if len({row['mamba_block_size'] for row in workers}) != 1:
        raise RuntimeError('TP workers disagree on actual Mamba block geometry')
    expected_previous = 4 if marker['stage'] == 'bridge' else 6
    if previous_gamma != expected_previous:
        raise RuntimeError('Live growth stages cannot shrink, repeat or skip the bridge')
    runner._q35_pending_growth_fence = None
    runner._q35_last_consumed_growth_fence = (marker['transition_id'], marker['stage'])
    print(f"[q35-phase-worker-consumed] pid={os.getpid()} transition={marker['transition_id']} "
          f"stage={marker['stage']} previous_gamma={previous_gamma} gamma={gamma}", flush=True)
    return marker['stage']


def drain_one_existing_frame(core, original_step):
    """Exercise native queue-pop/update behavior without scheduling another frame.

    Caller owns the engine-core thread. Temporary method binding is restored
    exactly, including descriptor lookup when there was no instance override.
    No request/token is deleted/rewritten and no worker method is invoked by
    this helper. Actual native-engine AST execution is independently tested.
    """
    if not core.batch_queue:
        raise ValueError('Only drain an existing frame; no fake empty batches')
    scheduler = core.scheduler
    present = 'has_requests' in scheduler.__dict__
    prior = scheduler.__dict__.get('has_requests')
    scheduler.has_requests = lambda: False
    try:
        return original_step(core)
    finally:
        if present:
            scheduler.has_requests = prior
        else:
            del scheduler.has_requests
