"""Default-off disjoint-domain composition of existing single and twin fences.

Only N1..3 q5 native graphs use the existing per-wrapper ReplayTail. No new
graph, workspace, tensor math, host attention arguments or ExternalEvent.
Other descriptors keep the original twin/native route. Domain exit drains
the last pending single graph before invalidating its ownership.
"""
from functools import wraps
import importlib
import json
import os
from ..config import q35_small_single_tail_enabled,q35_tail_prefix_shrink_enabled


def selected(manager,wrapper,context,native):
    d=context.batch_descriptor
    return bool(manager.enabled and manager.mode in ('on','validate')
        and context.cudagraph_runtime_mode==wrapper.runtime_mode
        and d is not None and d.uniform and d.num_reqs in (1,2,3)
        and d.num_tokens==d.num_reqs*5
        and not native._EXTRA_CTX.is_draft_model_prefill
        and not native.descriptor_admitted(d))


def install_small_single_tail():
    if not q35_small_single_tail_enabled():return False
    if not q35_tail_prefix_shrink_enabled():raise RuntimeError('Small single tail requires pinned prefix lane')
    native=importlib.import_module('vllm_ascend.compilation.q35_twin_graph')
    if native.envs.VLLM_ASCEND_Q35_TWIN_GRAPH not in ('on','validate'):
        raise RuntimeError('Small single tail requires original twin lane')
    original=native.Q35TwinGraph.call
    if getattr(original,'_q35_private_small_single_tail',False):return False

    @wraps(original)
    def call(manager,wrapper,run_native,*args,**kwargs):
        tail=getattr(wrapper,'_q35_replay_tail',None)
        if tail is None:return original(manager,wrapper,run_native,*args,**kwargs)
        context=native.get_forward_context() if manager.enabled else None
        use=(context is not None and selected(manager,wrapper,context,native)
             and not tail.counts.get('unsupported_owner_or_stream',0))
        if not use:
            if tail.enabled:
                if tail.pending:
                    # Conservative domain exit fence, not an unguarded invalidate.
                    tail.event.synchronize()
                    tail.counts['small_single_domain_exit_waits']+=1
                tail.invalidate()
                tail._small_single_seen=set()  # Logging-only: witness each real domain re-entry.
            tail.enabled=False
            return original(manager,wrapper,run_native,*args,**kwargs)
        tail.mode=manager.mode
        tail.enabled=True
        role='draft' if native._EXTRA_CTX.is_draft_model else 'target'
        previous=tail.counts.get('previous_graph_waits',0)
        result=original(manager,wrapper,run_native,*args,**kwargs)
        after=tail.counts.get('previous_graph_waits',0)
        if after>previous:
            n=context.batch_descriptor.num_reqs
            seen=getattr(tail,'_small_single_seen',None)
            if seen is None:seen=tail._small_single_seen=set()
            key=role,n
            if key not in seen or after in (1,2,32,128,256,1024):
                seen.add(key)
                print('[q35-small-single-tail] '+json.dumps(dict(
                    pid=os.getpid(),mode=manager.mode,role=role,requests=n,width=5,
                    previous_graph_waits=after,tail_records=tail.counts.get('tail_records',0),
                    completion_checks=tail.counts.get('previous_graph_completion_checks',0),
                    pending=tail.pending,event_identity=id(tail.event),no_extra_graph_or_workspace=True,
                    both_roles_keep_independent_original_events=True,
                    domain_exit_waits=tail.counts.get('small_single_domain_exit_waits',0))),flush=True)
        return result
    call._q35_private_small_single_tail=True
    call._q35_original_twin_call=original
    native.Q35TwinGraph.call=call
    return True
