"""Narrow q5 twins only for target and bound actual-gamma4 draft owners.

Keep gamma6 bridge q5 drafts on their original native path. Do not share
workspaces or handles or remove any native twin wait; avoid unused pairs.
"""
from contextvars import ContextVar
from functools import wraps
import importlib
import json
import os
from ..config import q35_narrow_tail_twin_enabled,q35_tail_prefix_shrink_enabled

_ALLOW_NARROW=ContextVar('q35_gamma_owned_narrow_twin',default=True)


def draft_owner_gamma(wrapper):
    owner=getattr(getattr(wrapper,'runnable',None),'__self__',None)
    gamma=getattr(owner,'num_speculative_tokens',None)
    return gamma if type(gamma) is int and gamma in (4,6) else None


def install_narrow_tail_twin():
    if not q35_narrow_tail_twin_enabled():
        return False
    if not q35_tail_prefix_shrink_enabled():
        raise RuntimeError('Narrow twin requires the pinned prefix-shrink lane')
    native=importlib.import_module('vllm_ascend.compilation.q35_twin_graph')
    if native.envs.VLLM_ASCEND_Q35_TWIN_GRAPH not in ('on','validate'):
        raise RuntimeError('Narrow twin requires matching native twin mode')
    original=native.descriptor_admitted
    if getattr(original,'_q35_private_narrow_tail_twin',False):
        if not getattr(native.Q35TwinGraph.call,'_q35_gamma_owned_narrow_twin',False):
            raise RuntimeError('Conflicting narrow-twin call ownership')
        return False

    @wraps(original)
    def admitted(descriptor):
        return bool(original(descriptor) or
            (_ALLOW_NARROW.get() and descriptor is not None and descriptor.uniform
             and descriptor.num_reqs in (1,2,3)
             and descriptor.num_tokens==descriptor.num_reqs*5))

    original_call=native.Q35TwinGraph.call

    @wraps(original_call)
    def call(manager,wrapper,run_native,*args,**kwargs):
        token=None
        if manager.enabled:
            context=native.get_forward_context()
            descriptor=context.batch_descriptor
            narrow=(descriptor is not None and descriptor.uniform
                    and descriptor.num_reqs in (1,2,3)
                    and descriptor.num_tokens==descriptor.num_reqs*5)
            if narrow and native._EXTRA_CTX.is_draft_model:
                gamma=draft_owner_gamma(wrapper)
                if gamma!=4:
                    token=_ALLOW_NARROW.set(False)
                    seen=getattr(manager,'_q35_gamma_owned_fallback_seen',None)
                    if seen is None:
                        seen=manager._q35_gamma_owned_fallback_seen=set()
                    key=descriptor.num_reqs,gamma
                    if key not in seen:
                        seen.add(key)
                        print('[q35-narrow-twin-gamma-fallback] '+json.dumps(dict(
                            pid=os.getpid(),requests=descriptor.num_reqs,actual_draft_gamma=gamma,
                            original_native_graph_retained=True,no_extra_twin_workspace=True)),flush=True)
        try:
            return original_call(manager,wrapper,run_native,*args,**kwargs)
        finally:
            if token is not None:_ALLOW_NARROW.reset(token)

    admitted._q35_private_narrow_tail_twin=True
    admitted._q35_original_descriptor_admitted=original
    call._q35_gamma_owned_narrow_twin=True
    call._q35_original_twin_call=original_call
    native.descriptor_admitted=admitted
    native.Q35TwinGraph.call=call
    return True
