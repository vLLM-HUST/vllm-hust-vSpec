"""Drain/arm a verified output-prefix bridge, retain ALL physical shrink guards."""
from functools import wraps
import json
import os
import time
from .quiescent_plan import drain_one_existing_frame

def basic_ready(s,ticket,controller):
    requests=list(s.running)
    return bool(ticket is not None and ticket.new==ticket.old-2
        and requests and len(requests)<=16 and not s.waiting and not s.skipped_waiting
        and not s.num_waiting_for_streaming_input and controller.current_gamma==ticket.old
        and all(not r.use_structured_output and r.num_computed_tokens>=r.num_prompt_tokens
            and r.max_tokens-r.num_output_tokens>=max(ticket.old,ticket.new)+1 for r in requests))

def install_core_step():
    from vllm.v1.engine.core import EngineCore
    original=EngineCore.step_with_batch_queue
    if getattr(original,'_private_prefix_bridge',False):return
    assert getattr(original,'_vspec_even_round',False)
    native_step=original.__wrapped__
    def finish_committed(phase,controller):
        plan=getattr(phase,'_private_prefix_clip_ticket',None)
        if plan is not None and controller.current_gamma==plan.new:
            controller.prefix_bridge_estimator_hold=False
            phase._private_prefix_clip_ticket=None
            print('[vspec-prefix-bridge-committed] '+json.dumps(plan.__dict__),flush=True)
    @wraps(original)
    def step(core):
        phase=getattr(core.scheduler,'_vspec_even_phase',None)
        if phase is None:return original(core)
        phase.check_thread();s=phase.scheduler;c=s._vspec_adaptive_controller
        if c.fixed4_ablation:return original(core)
        finish_committed(phase,c)
        plan=getattr(phase,'_private_prefix_clip_ticket',None)
        ticket=c.even.pending
        if plan is not None and (ticket!=plan or not phase.requests()):
            if core.batch_queue:return drain_one_existing_frame(core,native_step)
            receipts=core.model_executor.collective_rpc('disarm_private_prefix_clip_bridge',args=(plan.sequence,))
            assert (len(receipts)==2 and len({r['pid'] for r in receipts})==2
                and all(r['disarmed'] and r['device_fenced'] and r['sequence']==plan.sequence for r in receipts))
            c.prefix_bridge_estimator_hold=False;phase._private_prefix_clip_ticket=None
            c.prefix_bridge_arm_ticket=None;c.prefix_bridge_arm_rpc_ms=0.0
            print('[vspec-prefix-bridge-disarmed] '+json.dumps(dict(ticket=plan.__dict__,receipts=receipts)),flush=True)
            return original(core)
        if plan is None and not phase.bridge_wait and basic_ready(s,ticket,c):
            accepted=c.even_last_accepted
            already_fits=(c.even_last_feedback_gamma==ticket.old
                and all(r.request_id in accepted and accepted[r.request_id]<=ticket.new for r in phase.requests()))
            if not already_fits:
                if core.batch_queue:return drain_one_existing_frame(core,native_step)
                if any(r.num_output_placeholders for r in phase.requests()):
                    raise RuntimeError('Clip arm requires native drain and zero placeholders')
                roster=phase.roster()
                rpc_start=time.monotonic()
                receipts=core.model_executor.collective_rpc('arm_private_prefix_clip_bridge',
                    args=(ticket.sequence,ticket.old,ticket.new,roster))
                arm_rpc_ms=(time.monotonic()-rpc_start)*1000
                assert (len(receipts)==2 and len({r['pid'] for r in receipts})==2
                    and all(r['armed'] and r['device_fenced'] and r['no_cache_or_existing_count_mutation']
                        and (r['sequence'],r['old'],r['new'])==(ticket.sequence,ticket.old,ticket.new)
                        and set(r['roster'])==set(roster) for r in receipts))
                phase._private_prefix_clip_ticket=ticket;c.prefix_bridge_estimator_hold=True
                c.prefix_bridge_arm_ticket=ticket;c.prefix_bridge_arm_rpc_ms=arm_rpc_ms
                print('[vspec-prefix-bridge-armed] '+json.dumps(dict(ticket=ticket.__dict__,receipts=receipts,
                    queue_depth=0,placeholders_zero=True,original_physical_fence_still_required=True,
                    actual_arm_RPC_wall_ms=arm_rpc_ms,
                    original_count_producer_still_required=True,all_costs_in_measured_wall=True)),flush=True)
        try:return original(core)
        finally:finish_committed(phase,c)
    step._private_prefix_bridge=True
    EngineCore.step_with_batch_queue=step

def install(settings):
    if os.environ.get('HUST_ISSUE18_VERIFIED_PREFIX_BRIDGE')!='1':return False
    if os.environ.get('HUST_ISSUE18_B16_ARM')=='fixed4':return False
    if settings.method!='mtp' or os.environ.get('HUST_VSPEC_EVEN_ROUND_ENABLE')!='1':
        raise RuntimeError('Prefix bridge requires existing even-round MTP runtime')
    from ..backends.prefix_clip_bridge import install as install_worker
    install_worker() # Hard independent component proof check before selection.
    from .even_online import EvenRoundOnlineController
    original_commit=EvenRoundOnlineController.commit_even
    if not getattr(original_commit,'_private_prefix_bridge_cost',False):
        @wraps(original_commit)
        def commit(controller,ticket,batch_size,cost_ms):
            matching=getattr(controller,'prefix_bridge_arm_ticket',None)==ticket
            extra=getattr(controller,'prefix_bridge_arm_rpc_ms',0.0) if matching else 0.0
            result=original_commit(controller,ticket,batch_size,cost_ms+extra)
            if matching:
                controller.prefix_bridge_estimator_hold=False
                controller.prefix_bridge_arm_ticket=None;controller.prefix_bridge_arm_rpc_ms=0.0
                print('[vspec-prefix-bridge-policy-cost] '+json.dumps(dict(ticket=ticket.__dict__,
                    original_fence_RPC_wall_ms=cost_ms,additional_arm_RPC_wall_ms=extra,
                    total_admission_RPC_wall_ms=cost_ms+extra,
                    measured_generation_wall_NOT_modified=True)),flush=True)
            return result
        commit._private_prefix_bridge_cost=True
        EvenRoundOnlineController.commit_even=commit
    from vllm.v1.core.sched.scheduler import Scheduler
    original=Scheduler.__init__
    if getattr(original,'_private_prefix_bridge',False):return False
    @wraps(original)
    def initialize(s,*args,**kwargs):
        original(s,*args,**kwargs)
        assert hasattr(s,'_vspec_even_phase')
        s._vspec_even_phase._private_prefix_clip_ticket=None
        install_core_step()
    initialize._private_prefix_bridge=True
    Scheduler.__init__=initialize
    return True
