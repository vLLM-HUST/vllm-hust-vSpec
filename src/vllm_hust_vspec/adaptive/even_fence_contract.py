"""CPU admission of actual two-rank even-width cache fence receipts.

No flag alone admits a live transition. GPU proofs are built by the private
worker; all three physical cache groups and all thirty GDN layers are required.
"""
def geometry(old,new,stage):
    if old not in (2,4,6,8) or new not in (2,4,6,8) or abs(new-old)!=2:
        raise ValueError('Only adjacent complete MTP2 groups in [2,8]')
    if stage not in (('bridge','wide') if new>old else ('shrink',)):
        raise ValueError('Wrong live transition stage')
    return dict(old_width=old+1,new_width=new+1,
                runtime=old if stage in ('bridge','shrink') else new,
                previous_draft=old if stage in ('bridge','shrink') else new,
                accepted_max=min(old+1,new+1),
                backward=old+1,
                forward=(old+1)+(new+1) if stage=='bridge' else new+1)

def consume(runner,marker,previous_gamma,gamma,pid):
    if not isinstance(marker,dict):raise RuntimeError('Missing transition marker')
    old,new,stage=marker.get('old_gamma'),marker.get('new_gamma'),marker.get('stage')
    g=geometry(old,new,stage)
    if (gamma!=new or previous_gamma!=g['runtime'] or marker.get('queue_depth')!=0
            or marker.get('placeholders_zero') is not True):
        raise RuntimeError('Runtime width/queue/placeholder mismatch')
    own=getattr(runner,'_vspec_even_pending_fence',None)
    workers=marker.get('worker_receipts',[])
    roster=marker.get('roster',[]);n=len(roster)
    if (not 1<=n<=16 or len(set(roster))!=n or len(workers)!=2
            or len({r.get('pid') for r in workers})!=2 or own not in workers
            or not isinstance(own,dict) or own.get('pid')!=pid):
        raise RuntimeError('Missing own actual TP2 roster/fence receipt')
    for row in workers:
        size=row.get('mamba_block_size')
        if (not row.get('eligible') or not row.get('device_fenced')
                or row.get('transition_id')!=marker.get('transition_id')
                or row.get('stage')!=stage or row.get('old_gamma')!=old or row.get('new_gamma')!=new
                or set(row.get('roster',[]))!=set(roster) or len(row.get('roster',[]))!=n
                or row.get('previous_draft_width')!=g['previous_draft']
                or row.get('previous_draft_stride')!=g['previous_draft']
                or row.get('old_actual_width')!=g['old_width'] or row.get('new_actual_width')!=g['new_width']
                or row.get('cache_tensors_rewritten') is not False
                or row.get('sampled_counts_rewritten') is not False
                or row.get('previous_draft_tensor_rewritten') is not False
                or row.get('all30_cache_layouts_validated') is not True
                or row.get('actual_width_cache_address_signature_validated') is not True
                or type(size) is not int or size<=0):
            raise RuntimeError('Incomplete actual cache/stride/fence proof')
        accepted=row.get('accepted_counts',[]);lengths=row.get('old_end_seq_lens',[])
        if (len(accepted)!=n or any(type(a) is not int or not 1<=a<=g['accepted_max'] for a in accepted)
                or len(lengths)!=n or any(type(x) is not int or x<=g['backward']
                    or (x-g['backward']-1)//size!=(x-1)//size
                    or (x-1)//size!=(x+g['forward']-1)//size for x in lengths)):
            raise RuntimeError('Accepted prefix or bidirectional block bound not proven')
        groups=row.get('groups',[])
        if (len(groups)!=3 or {r.get('gid') for r in groups}!={1,2,3}
                or any(not r.get('previous_actual_canonical') or not r.get('accepted_state_retained')
                    or not r.get('next_slots_allocated') or r.get('mamba_block_size')!=size
                    or r.get('actual_previous_width')!=g['old_width'] for r in groups)):
            raise RuntimeError('All physical cache groups must prove canonical accepted state')
    if len({r['mamba_block_size'] for r in workers})!=1:
        raise RuntimeError('TP workers disagree on Mamba block geometry')
    if runner.prev_num_spec_tokens!=g['previous_draft'] or tuple(runner._draft_token_ids.shape)!=(n,g['previous_draft']):
        raise RuntimeError('Original previous draft stride must survive native input gather')
    if stage=='wide' and getattr(runner,'_vspec_even_consumed_fence',None)!=(marker['transition_id'],'bridge'):
        raise RuntimeError('Wide stage requires this exact consumed bridge')
    runner._vspec_even_pending_fence=None
    runner._vspec_even_consumed_fence=(marker['transition_id'],stage)
    return stage
