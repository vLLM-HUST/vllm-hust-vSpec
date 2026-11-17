"""Read-only native replay-tail counters, RPC only AFTER benchmark timers."""
def snapshot(wrappers):
    rows=[]
    for wrapper in wrappers:
        tail=getattr(wrapper,'_q35_replay_tail',None)
        if tail is None or not tail.enabled or not tail.counts.get('tail_records',0):
            continue
        if tail.role not in ('target','draft') or tail.mode!='validate':
            raise RuntimeError('Expected actual original native validated target/draft tail')
        rows.append(dict(wrapper_id=id(wrapper),role=tail.role,mode=tail.mode,
            enabled=tail.enabled,counts=dict(tail.counts),owner=tail.owner,
            stream_key=tail.stream_key,event_present=tail.event is not None,
            pending=tail.pending,native_before_after_and_task_update_code_UNCHANGED=True,
            read_only_RPC_AFTER_timers=True))
    return rows
