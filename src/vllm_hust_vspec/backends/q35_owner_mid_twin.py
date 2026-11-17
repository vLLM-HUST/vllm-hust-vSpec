"""Default-off gamma-owned extra twins; original native fallback and fences retained."""
import ast
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import textwrap
from ..config import q35_owner_mid_twin_enabled,q35_two_bank_embedding_enabled,q35_replicated_inputs

from .experimental_paths import artifact_dir, require_current_npu_proof

PRIMITIVE=artifact_dir('VSPEC_OWNER_MID_TWIN_DIR')
OLD_PRIMITIVE=artifact_dir('VSPEC_OWNER_TAIL_TWIN_DIR')
MID=(4,5,6)
ALLOW=ContextVar('q35_gamma_owned_all_twins',default=True)
COUNTS=Counter()


def verify_primitive():
    paths=[OLD_PRIMITIVE/'primitive-native-audit-r1.json',PRIMITIVE/'primitive-native-audit-r1.json']
    for path,count in zip(paths,(2160,3024)):
        p=json.loads(path.read_text())
        require_current_npu_proof(p)
        assert p['passed'] and p['scope']=='target-draft'
        assert p['numerical_comparisons']==count and p['all_consumed_output_rows_finite_and_bitwise_exact']
        assert p['original_per_group_update_and_ExternalEvent_order_preserved']
        assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in p['sha256'].items()),path
    return paths


def actual_gamma(wrapper):
    owner=getattr(getattr(wrapper,'runnable',None),'__self__',None)
    value=getattr(owner,'num_speculative_tokens',None)
    return value if type(value) is int and value in (4,6) else None


def mid_descriptor(d):
    return bool(d is not None and d.uniform and type(d.num_reqs) is int
        and d.num_reqs in MID and d.num_tokens==d.num_reqs*7)


def expanded_host_function(original,globals_dict):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));tree.body[0].decorator_list=[]
    found=[]
    class Domain(ast.NodeTransformer):
        def visit_Tuple(self,node):
            self.generic_visit(node)
            if all(isinstance(n,ast.Constant) for n in node.elts) and [n.value for n in node.elts]==[1,2,3,14]:
                found.append(node);node.elts=[ast.Constant(value=n) for n in (1,2,3,4,5,6,14)]
            return node
    Domain().visit(tree)
    if len(found)!=1:raise RuntimeError('Only original request-domain tuple may be extended')
    ast.fix_missing_locations(tree);local={}
    exec(compile(tree,'<private-native-host-mid456-domain>','exec'),globals_dict,local)
    return wraps(original)(local[original.__name__])


def install_owner_mid_twin():
    if not q35_owner_mid_twin_enabled():return False
    from ..config import q35_tail_prefix_shrink_enabled,q35_narrow_tail_twin_enabled,q35_hybrid_tail_enabled
    if (not q35_two_bank_embedding_enabled() or q35_replicated_inputs()!='embedding'
            or q35_tail_prefix_shrink_enabled() or q35_narrow_tail_twin_enabled() or q35_hybrid_tail_enabled()):
        raise RuntimeError('Owner/mid twin requires pinned no-prefix two-bank lane')
    native=importlib.import_module('vllm_ascend.compilation.q35_twin_graph')
    host=importlib.import_module('vllm_ascend.attention.q35_native_host_batch')
    if native.envs.VLLM_ASCEND_Q35_TWIN_GRAPH not in ('on','validate'):
        raise RuntimeError('Owner/mid requires original native twin mode')
    original=native.descriptor_admitted
    if getattr(original,'_q35_owner_mid_twin',False):
        if (not getattr(native.Q35TwinGraph.call,'_q35_owner_mid_twin',False)
                or host.PRIMITIVE!=PRIMITIVE
                or not getattr(host.NativeFIAHostBatch.try_update,'_q35_owner_mid_twin',False)):
            raise RuntimeError('Conflicting partial owner/mid installation')
        return False
    verify_primitive()
    original_call=native.Q35TwinGraph.call

    @wraps(original)
    def admitted(d):return bool(ALLOW.get() and (original(d) or mid_descriptor(d)))

    @wraps(original_call)
    def call(manager,wrapper,run_native,*args,**kwargs):
        token=None
        if manager.enabled:
            context=native.get_forward_context();d=context.batch_descriptor
            if (context.cudagraph_runtime_mode==wrapper.runtime_mode
                    and not native._EXTRA_CTX.is_draft_model_prefill and native._EXTRA_CTX.is_draft_model
                    and (original(d) or mid_descriptor(d))):
                width=d.num_tokens//d.num_reqs
                gamma=actual_gamma(wrapper)
                if not ((gamma==4 and width==5) or (gamma==6 and width==7)):
                    token=ALLOW.set(False)
                    key=f'gamma{gamma}:N{d.num_reqs}:width{width}'
                    COUNTS[key]+=1
                    seen=getattr(manager,'_q35_owner_mid_seen',None)
                    if seen is None:seen=manager._q35_owner_mid_seen=set()
                    if key not in seen:
                        seen.add(key)
                        print('[q35-owner-mid-native-fallback] '+json.dumps(dict(pid=os.getpid(),
                            actual_draft_gamma=gamma,requests=d.num_reqs,width=width,
                            original_native_graph_and_barriers_retained=True,
                            no_extra_pair_workspace_handle_or_event=True)),flush=True)
        try:return original_call(manager,wrapper,run_native,*args,**kwargs)
        finally:
            if token is not None:ALLOW.reset(token)

    old_eligible=host.eligible_rows
    expanded_eligible=expanded_host_function(old_eligible,host.__dict__)
    @wraps(old_eligible)
    def eligible(rows,role,tokens,requests=14):
        if requests in MID and tokens==requests*7:return expanded_eligible(rows,role,tokens,requests)
        return old_eligible(rows,role,tokens,requests)
    old_update=host.NativeFIAHostBatch.try_update
    expanded_update=expanded_host_function(old_update,host.__dict__)
    @wraps(old_update)
    def update(manager,stream,context,tokens,*args,**kwargs):
        if mid_descriptor(getattr(context,'batch_descriptor',None)):
            return expanded_update(manager,stream,context,tokens,*args,**kwargs)
        return old_update(manager,stream,context,tokens,*args,**kwargs)
    old_metrics,old_reset=host.NativeFIAHostBatch.metrics,host.NativeFIAHostBatch.reset
    @wraps(old_metrics)
    def metrics(manager):
        record=old_metrics(manager)
        record['owner_mid_twin']=dict(only_new_requests=list(MID),only_new_width=7,
            actual_native_fallback_calls_after_original_named_reset=dict(COUNTS),
            original_native_fallback_and_all_twin_fences_unchanged=True,
            no_shared_workspace_handle_or_event=True,stats_read_after_timers_only=True,
            primitive=str(PRIMITIVE))
        return record
    @wraps(old_reset)
    def reset(manager):
        result=old_reset(manager);COUNTS.clear();return result
    update._q35_owner_mid_twin=True
    admitted._q35_owner_mid_twin=True
    call._q35_owner_mid_twin=True
    call._q35_owner_mid_original_call=original_call
    native.descriptor_admitted=admitted;native.Q35TwinGraph.call=call
    host.eligible_rows=eligible;host.NativeFIAHostBatch.try_update=update
    host.NativeFIAHostBatch.metrics=metrics;host.NativeFIAHostBatch.reset=reset
    host.PRIMITIVE=PRIMITIVE;host.verify_primitive=verify_primitive
    print('[q35-owner-mid-install] '+json.dumps(dict(pid=os.getpid(),
        new_width7_requests=list(MID),draft_q7_twins_only_actual_gamma6=True,
        draft_q5_twins_only_actual_gamma4=True,unknown_owner_keeps_native=True,
        all_original_native_FIA_arguments_and_event_order_retained=True)),flush=True)
    return True
