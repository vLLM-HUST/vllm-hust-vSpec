"""Drain/fence application of per-round proposals; native tensor math stays put."""
from functools import wraps
import json
import os
import threading
import time
from .quiescent_plan import drain_one_existing_frame

class EvenPhase:
    def __init__(self,scheduler):
        self.scheduler=scheduler;self.owner=None
        self.ticket=self.active=self.ready=None
        self.pending_since=None;self.bridge_wait=False;self.bridge_roster=None
        self.admission_rpc_ms=0.0
    def check_thread(self):
        current=threading.get_ident()
        if self.owner is None:self.owner=current
        if self.owner!=current:raise RuntimeError('One engine-core owner thread required')
    def requests(self):return list(self.scheduler.running)
    def roster(self):return tuple(sorted(r.request_id for r in self.requests()))
    def consider(self,core):
        controller=self.scheduler._vspec_adaptive_controller
        ticket=controller.even.pending
        if ticket is None:
            self.ticket=self.pending_since=None
            return False
        if self.ticket!=ticket:
            self.ticket=ticket;self.pending_since=None
            self.admission_rpc_ms=0.0
        requests=self.requests();s=self.scheduler
        ready=bool(requests and len(requests)<=16 and not s.waiting and not s.skipped_waiting
                    and not s.num_waiting_for_streaming_input
                    and controller.current_gamma==ticket.old
                    and all(not r.use_structured_output and r.num_computed_tokens>=r.num_prompt_tokens
                            and r.max_tokens-r.num_output_tokens>=max(ticket.old,ticket.new)+1 for r in requests))
        if ready and ticket.new<ticket.old:
            # Native verifier feedback is already CPU-visible. Do not drain or
            # issue a D2H fence while the narrower accepted prefix cannot fit.
            accepted=controller.even_last_accepted
            ready=(controller.even_last_feedback_gamma==ticket.old
                   and all(r.request_id in accepted and accepted[r.request_id]<=ticket.new for r in requests))
        if ready and self.pending_since is None:self.pending_since=time.monotonic()
        return ready
    def fence(self,core,stage):
        controller=self.scheduler._vspec_adaptive_controller
        ticket=self.active if stage=='wide' else controller.even.pending
        if ticket is None:raise RuntimeError('No pending actual even transition')
        requests=self.requests();roster=self.roster()
        if core.batch_queue or not requests or any(r.num_output_placeholders for r in requests):
            raise RuntimeError('Live transition requires a fully drained real cohort')
        if stage!='wide' and any(len(r.spec_token_ids)!=ticket.old for r in requests):
            controller.even.reject(ticket);return False
        physical_roster=self.bridge_roster if stage=='wide' else roster
        if stage=='wide' and (physical_roster is None or not set(roster).issubset(physical_roster)):
            raise RuntimeError('No new request can enter a partially admitted bridge')
        rpc_start=time.monotonic()
        receipts=core.model_executor.collective_rpc('fence_private_even_mtp_transition',
                     args=(ticket.sequence,ticket.old,ticket.new,stage,physical_roster))
        self.admission_rpc_ms+=(time.monotonic()-rpc_start)*1000
        eligible=(len(receipts)==2 and len({r.get('pid') for r in receipts})==2
                  and all(r.get('eligible') and r.get('device_fenced')
                    and r.get('transition_id')==ticket.sequence and r.get('stage')==stage
                    and r.get('old_gamma')==ticket.old and r.get('new_gamma')==ticket.new
                    and set(r.get('roster',[]))==set(physical_roster) for r in receipts))
        if not eligible:
            print('[vspec-even-fence-rejected] '+json.dumps(dict(stage=stage,receipts=receipts)),flush=True)
            if stage=='wide':raise RuntimeError('Cannot fallback after partially admitted even bridge')
            retryable={'accepted_checkpoint_outside_preserved_prefix','near_bidirectional_mamba_block_boundary',
                       'worker_roster_not_exact_current_cohort'}
            if all(r.get('reason') in retryable for r in receipts):
                self.pending_since=None
                return False
            controller.even.reject(ticket);return False
        self.ready=dict(kind='even_round',transition_id=ticket.sequence,stage=stage,
                        old_gamma=ticket.old,new_gamma=ticket.new,queue_depth=0,
                        placeholders_zero=True,roster=list(physical_roster),
                        scheduled_survivor_roster=list(roster),worker_receipts=receipts)
        self.ready['policy_cost_metadata']=dict(admission_rpc_wall_ms_accumulated_for_ticket=self.admission_rpc_ms,
            elapsed_since_most_recent_ready_ms=(time.monotonic()-self.pending_since)*1000,
            measured_generation_wall_NOT_modified=True,
            useful_queue_and_bridge_work_NOT_counted_as_exclusive_admission_cost=True)
        if stage=='shrink':
            # Discard ONLY unverified CPU candidate suffixes. Native GPU previous
            # tensor/stride and all accepted states remain untouched for gather.
            for r in requests:r.spec_token_ids=r.spec_token_ids[:ticket.new]
        if stage in ('bridge','shrink'):
            controller.commit_even(ticket,len(requests),self.admission_rpc_ms)
        if stage=='bridge':self.active=ticket;self.bridge_roster=roster
        if stage=='wide':
            controller.even.switch_cost_ms=self.admission_rpc_ms
            self.active=None
        self.bridge_wait=False
        print('[vspec-even-fence-admitted] '+json.dumps(self.ready),flush=True)
        return True

def install_core_step():
    from vllm.v1.engine.core import EngineCore
    original=EngineCore.step_with_batch_queue
    if getattr(original,'_vspec_even_round',False):return
    @wraps(original)
    def step(core):
        phase=getattr(core.scheduler,'_vspec_even_phase',None)
        if phase is None:return original(core)
        phase.check_thread()
        if phase.bridge_wait:
            if core.batch_queue:return drain_one_existing_frame(core,original)
            if not phase.requests():
                # Every old request has ended. No old state will be consumed;
                # the next new cohort uses native empty-boundary initialization.
                phase.bridge_wait=False;phase.active=None;phase.bridge_roster=None
                return original(core)
            phase.fence(core,'wide');return original(core)
        if phase.consider(core):
            if core.batch_queue:return drain_one_existing_frame(core,original)
            requests=phase.requests()
            if all(r.num_output_placeholders==0 for r in requests):
                ticket=phase.scheduler._vspec_adaptive_controller.even.pending
                phase.fence(core,'bridge' if ticket.new>ticket.old else 'shrink')
        return original(core)
    step._vspec_even_round=True
    EngineCore.step_with_batch_queue=step

def install(settings):
    if os.environ.get('HUST_VSPEC_EVEN_ROUND_ENABLE')!='1':return False
    if (settings.method!='mtp' or not settings.adaptive_speculation or not settings.adaptive_async
            or settings.adaptive_policy!='online' or (settings.adaptive_min_gamma,settings.adaptive_max_gamma)!=(2,8)
            or os.environ.get('VLLM_ASCEND_Q35_QUIESCENT_GROWTH')!='off'
            or os.environ.get('HUST_VSPEC_MTP_ASYNC_RUNTIME_SWITCH','0')=='1'):
        raise RuntimeError('Even rounds require dedicated even-gamma/max8 async lane, old4to6 growth OFF')
    from . import runtime
    from .even_online import EvenRoundOnlineController
    runtime.OnlineGammaController=EvenRoundOnlineController
    from ..backends.q35_even_fence import install as install_rpc
    install_rpc()
    from vllm.v1.core.sched.scheduler import Scheduler
    original_init,original_schedule=Scheduler.__init__,Scheduler.schedule
    if getattr(original_init,'_vspec_even_round',False):return False
    @wraps(original_init)
    def initialize(s,*args,**kwargs):
        original_init(s,*args,**kwargs)
        c=s.vllm_config
        if (c.speculative_config.num_speculative_tokens!=8 or c.parallel_config.tensor_parallel_size!=2
                or c.parallel_config.pipeline_parallel_size!=1 or c.parallel_config.data_parallel_size!=1
                or str(c.model_config.dtype)!='torch.bfloat16' or c.max_concurrent_batches!=2
                or getattr(c.model_config,'quantization',None) is not None or c.lora_config is not None):
            raise RuntimeError('Even rounds require unquantized BF16 TP2/PP1/DP1 native8 capacity queue2')
        s._vspec_even_phase=EvenPhase(s)
        install_core_step()
    @wraps(original_schedule)
    def schedule(s,*args,**kwargs):
        phase=s._vspec_even_phase;marker=phase.ready
        previous=s.max_num_running_reqs
        if marker is not None:s.max_num_running_reqs=len(s.running)
        try:output=original_schedule(s,*args,**kwargs)
        finally:s.max_num_running_reqs=previous
        if marker is not None and output.num_scheduled_tokens:
            actual=set(output.num_scheduled_tokens)
            width=marker['old_gamma'] if marker['stage']=='bridge' else marker['new_gamma']
            if (output.scheduled_new_reqs or not actual.issubset(marker['roster'])
                    or output.num_spec_tokens_to_schedule!=marker['new_gamma']
                    or any(len(output.scheduled_spec_decode_tokens.get(key,()))!=width for key in actual)):
                raise RuntimeError('Actual target and next draft widths must match admitted stage')
            output.vspec_q35_quiescent_growth_receipt=marker
            phase.ready=None;phase.bridge_wait=marker['stage']=='bridge'
        return output
    initialize._vspec_even_round=True
    Scheduler.__init__,Scheduler.schedule=initialize,schedule
    return True
