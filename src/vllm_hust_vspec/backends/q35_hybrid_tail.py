"""Default-off N1/N3 twins plus original N2 single-graph event protocol."""
from functools import wraps
import importlib
import json
import os
from ..config import (q35_hybrid_tail_enabled,q35_small_single_tail_enabled,
    q35_narrow_tail_twin_enabled,q35_tail_prefix_shrink_enabled)


def install_hybrid_tail():
    if not q35_hybrid_tail_enabled():return False
    if not (q35_small_single_tail_enabled() and q35_narrow_tail_twin_enabled()
            and q35_tail_prefix_shrink_enabled()):
        raise RuntimeError('Hybrid tail requires small-single narrow and prefix flags')
    from ..config import q35_hybrid_n1_only_enabled
    n1_only=q35_hybrid_n1_only_enabled()
    singles=(2,3) if n1_only else (2,)
    native=importlib.import_module('vllm_ascend.compilation.q35_twin_graph')
    if native.envs.VLLM_ASCEND_Q35_TWIN_GRAPH not in ('on','validate'):
        raise RuntimeError('Hybrid requires matching original native twin mode')
    original=native.descriptor_admitted
    if getattr(original,'_q35_hybrid_tail',False):
        if not getattr(native.Q35TwinGraph.call,'_q35_private_small_single_tail',False):
            raise RuntimeError('Conflicting hybrid single-tail ownership')
        return False
    if not getattr(original,'_q35_private_narrow_tail_twin',False):
        raise RuntimeError('Install original gamma-owned narrow twins first')

    @wraps(original)
    def admitted(d):
        if d is not None and d.uniform and d.num_reqs in singles and d.num_tokens==d.num_reqs*5:
            return False
        return original(d)

    admitted._q35_hybrid_tail=True
    admitted._q35_hybrid_original_descriptor=original
    native.descriptor_admitted=admitted
    from .q35_small_single_tail import install_small_single_tail
    if not install_small_single_tail():
        raise RuntimeError('Hybrid must freshly install its outer single-tail guard')
    print('[q35-hybrid-tail-install] '+json.dumps(dict(pid=os.getpid(),
        q5_twin_requests=[1] if n1_only else [1,3],q5_original_single_requests=list(singles),
        original_wide_q7_and_fullN14_twins_unchanged=True,
        no_shared_workspace_handle_or_event=True)),flush=True)
    return True
