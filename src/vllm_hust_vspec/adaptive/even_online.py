"""Bridge actual verifier feedback to the CPU policy, without changing math."""
import os
from .online import OnlineGammaController
from .cpu_memo import MemoizedEvenRoundSelector as EvenRoundSelector
from .even_probe_budget import probe_budget
from .even_empty_return import measured_empty_return

class EvenRoundOnlineController(OnlineGammaController):
    def __init__(self,**options):
        self.fixed4_ablation=os.environ.get('HUST_ISSUE18_B16_ARM')=='fixed4'
        initial=4 if self.fixed4_ablation else 2
        if (options['min_gamma'],options['max_gamma'],options['candidate_stride'],options['initial_gamma'])!=(2,8,2,initial):
            raise RuntimeError('Even-round controller requires start2, stride2, max8')
        super().__init__(**options)
        self.even=EvenRoundSelector(window=options['window_size'],hysteresis=options['hysteresis'],
                                    amortization_rounds=max(64,options['window_size']))
        if self.fixed4_ablation:
            self.even.current_gamma=self.even.incumbent=4
            self.even.bootstrap_committed_widths={4}
        self.even_refs={};self.even_acceptance={}
        self.even_last_accepted={};self.even_last_feedback_gamma=None
        self.even_last_logged_sequence=0
        self.even_horizon_enabled=os.environ.get('HUST_ISSUE18_EVEN_HORIZON')=='1'
        self.even_horizon_cancellations=0
        self.even_remaining_rounds=None
        self.even_last_probe_budget=None
        self.even_empty_boundary_returns=0
        self.prefix_bridge_estimator_hold=False
        self.prefix_bridge_estimator_excluded=0
        self.prefix_bridge_arm_ticket=None
        self.prefix_bridge_arm_rpc_ms=0.0

    def begin_verified_round(self,scheduler,output):
        self.even_refs={key:(scheduler.requests[key],len(scheduler.requests[key].output_token_ids))
                        for key in output.num_scheduled_tokens if key in scheduler.requests}

    def begin_step_feedback(self):
        super().begin_step_feedback()
        self.even_acceptance={}

    def observe(self,num_draft_tokens,num_accepted_tokens,*,request_id=None):
        super().observe(num_draft_tokens,num_accepted_tokens,request_id=request_id)
        if request_id is not None:self.even_acceptance[request_id]=num_accepted_tokens

    def complete_step(self,**data):
        if self.fixed4_ablation:
            super().complete_step(**data)
            return
        previous=self._last_completion_ms
        accepted=dict(self.even_acceptance)
        super().complete_step(**data)
        end=data.get('completed_at_ms')
        latency=data.get('latency_ms') if end is None else None if previous is None else end-previous
        keys=tuple(key for key in self.even_refs if key in accepted)
        if latency is None or latency<=0 or len(keys)!=data['batch_size']:return
        actual=[len(self.even_refs[key][0].output_token_ids)-self.even_refs[key][1] for key in keys]
        if not actual or sum(actual)<=0:return
        a=[accepted[key] for key in keys]
        self.even_last_accepted=dict(zip(keys,a))
        self.even_last_feedback_gamma=data['gamma']
        if any(c<0 or c>v+1 for v,c in zip(a,actual)):
            raise RuntimeError('Actual committed outputs disagree with verifier feedback')
        # Batch and CPU-known context length isolate comparable cost windows.
        # ONLY estimator arrays are identity-aligned. Original feedback maps,
        # accepted counts and every physical-fence RPC argument stay intact.
        utility_order=sorted(range(len(keys)),key=lambda i:keys[i])
        utility_roster=tuple(keys[i] for i in utility_order)
        context=(data['batch_size'],max(self.even_refs[key][0].num_computed_tokens for key in keys)//2048,
            utility_roster)
        if self.even.pending is not None and self.even.context!=context:
            self.even.reject(self.even.pending)
        if getattr(self,'prefix_bridge_estimator_hold',False):
            # Actual native acceptance/committed counts above still drive the
            # original physical fence. Only steady utility fitting excludes
            # deliberately shortened bridge rounds; measured wall keeps ALL.
            self.prefix_bridge_estimator_excluded+=1
            ticket=self.even.pending
        else:
            ticket=self.even.observe(gamma=data['gamma'],accepted=[a[i] for i in utility_order],
                committed=[actual[i] for i in utility_order],latency_ms=latency,context=context)
        if self.even_horizon_enabled:
            refs=[self.even_refs[key][0] for key in keys]
            known=all(type(r.max_tokens) is int and r.max_tokens>=0 for r in refs)
            if known:
                live=[r for r in refs if not (callable(getattr(r,'is_finished',None)) and r.is_finished())]
                # A same-batch calibration can end when the FIRST request
                # hits its known output cap, not when the last one ends.
                # EOS can still arrive earlier: this remains a heuristic.
                remaining=min((max(0,r.max_tokens-len(r.output_token_ids)) for r in live),default=0)
                rows=self.even.samples[data['gamma']]
                requests=sum(len(r[3]) for r in rows)
                progress=sum(r[0] for r in rows)/requests if requests else sum(actual)/len(actual)
                self.even_remaining_rounds=remaining/max(1.0,progress)
                required=self.even.window+self.even.amortization_rounds
                if (ticket is not None
                        and ticket.reason in ('adjacent_higher_trial','adjacent_lower_trial_missing_context')):
                    self.even_last_probe_budget=probe_budget(self.even,ticket,list(rows))
                    required=self.even_last_probe_budget['required_rounds']
                if (ticket is not None
                        and ticket.reason in ('adjacent_higher_trial','adjacent_lower_trial_missing_context')
                        and self.even_remaining_rounds<required):
                    # Cancel only an UNCOMMITTED new probe. A measured-loser
                    # return, measured lower gain and admitted bridge remain
                    # untouched; all worker prefix/fence proofs still apply.
                    self.even.cancel_unmeasured_probe(ticket)
                    self.even_horizon_cancellations+=1
                    import json
                    print('[vspec-even-horizon-hold] '+json.dumps(dict(ticket=ticket.__dict__,
                        remaining_tokens=remaining,estimated_remaining_rounds=self.even_remaining_rounds,
                        minimum_probe_plus_payback_rounds=required,actual_gamma=self.even.current_gamma)),flush=True)
                    ticket=None
            else:self.even_remaining_rounds=None
        if ticket is not None and ticket.sequence!=self.even_last_logged_sequence:
            self.even_last_logged_sequence=ticket.sequence
            import json
            rows=self.even.samples[data['gamma']]
            print('[vspec-even-round-proposal] '+json.dumps(dict(ticket=ticket.__dict__,
                context=context,rounds=self.even.rounds,window_n=len(rows),
                useful_tokens=sum(r[0] for r in rows),latency_ms=sum(r[1] for r in rows))),flush=True)

    def choose(self,batch_size,context_tokens,**kwargs):
        ticket=measured_empty_return(self.even,
            at_cohort_boundary=kwargs.get('at_cohort_boundary'),fixed_control=self.fixed4_ablation)
        if ticket is not None:
            self.even.commit(ticket,transition_cost_ms=0.0)
            self.even_empty_boundary_returns+=1
            import json
            print('[vspec-even-empty-measured-return] '+json.dumps(dict(ticket=ticket.__dict__,
                authoritative_empty_cohort_marker_still_required=True,
                original_worker_device_synchronization_unchanged=True,
                no_live_KV_or_sampled_counts_rewritten=True)),flush=True)
        del kwargs
        return self.force(self.even.current_gamma,batch_size,context_tokens,reason='even_round_actual_width_hold')

    def commit_even(self,ticket,batch_size,cost_ms):
        self.even.commit(ticket,transition_cost_ms=cost_ms)
        return self.force(ticket.new,batch_size,0,reason=ticket.reason)

    def summary(self):
        result=super().summary()
        result['even_round_selector']=self.even.summary()
        if self.fixed4_ablation:
            result['even_round_selector']['initial_gamma']=4
            result['controller_only_fixed4_ablation']=True
            result['fixed4_no_adaptive_feedback_or_transition_decisions']=True
            assert self.current_gamma==self.even.current_gamma==4 and self.even.pending is None
        result['actual_committed_output_counts_used']=not self.fixed4_ablation
        result['verified_prefix_bridge_estimator']=dict(active=getattr(self,'prefix_bridge_estimator_hold',False),
            bridge_rounds_excluded_from_steady_estimator_ONLY=getattr(self,'prefix_bridge_estimator_excluded',0),
            original_actual_acceptance_feedback_and_benchmark_wall_NOT_excluded=True)
        result['measured_empty_boundary_returns']=self.even_empty_boundary_returns
        result['even_horizon_guard']=dict(enabled=self.even_horizon_enabled,
            remaining_work_basis='first_live_request_known_output_cap; EOS_may_end_earlier',
            new_probes_cancelled=self.even_horizon_cancellations,
            estimated_remaining_rounds=self.even_remaining_rounds,
            maximum_probe_plus_payback_rounds=self.even.window+self.even.amortization_rounds,
            last_online_probe_budget=self.even_last_probe_budget,
            no_labels_or_question_ids_or_fixed_winning_gamma=True)
        return result
