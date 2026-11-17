"""Bridge actual verifier feedback to the CPU policy, without changing math."""
import os
from .online import OnlineGammaController
from .legacy_even_round import EvenRoundSelector

class EvenRoundOnlineController(OnlineGammaController):
    def __init__(self,**options):
        if (options['min_gamma'],options['max_gamma'],options['candidate_stride'],options['initial_gamma'])!=(2,8,2,8):
            raise RuntimeError('Even-round controller requires start8, stride2, max8')
        super().__init__(**options)
        self.even=EvenRoundSelector(window=options['window_size'],hysteresis=options['hysteresis'],
                                    amortization_rounds=max(64,options['window_size']))
        self.even_refs={};self.even_acceptance={}
        self.even_last_accepted={};self.even_last_feedback_gamma=None
        self.even_last_logged_sequence=0
        self.even_horizon_enabled=os.environ.get('HUST_ISSUE18_EVEN_HORIZON')=='1'
        self.even_horizon_cancellations=0
        self.even_remaining_rounds=None

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
        context=(data['batch_size'],max(self.even_refs[key][0].num_computed_tokens for key in keys)//2048)
        if self.even.pending is not None and self.even.context!=context:
            self.even.reject(self.even.pending)
        ticket=self.even.observe(gamma=data['gamma'],accepted=a,committed=actual,latency_ms=latency,context=context)
        if self.even_horizon_enabled:
            refs=[self.even_refs[key][0] for key in keys]
            known=all(type(r.max_tokens) is int and r.max_tokens>=0 for r in refs)
            if known:
                live=[r for r in refs if not (callable(getattr(r,'is_finished',None)) and r.is_finished())]
                remaining=max((max(0,r.max_tokens-len(r.output_token_ids)) for r in live),default=0)
                rows=self.even.samples[data['gamma']]
                requests=sum(len(r[3]) for r in rows)
                progress=sum(r[0] for r in rows)/requests if requests else sum(actual)/len(actual)
                self.even_remaining_rounds=remaining/max(1.0,progress)
                required=self.even.window+self.even.amortization_rounds
                if (ticket is not None
                        and ticket.reason in ('adjacent_higher_trial','adjacent_lower_trial_missing_context')
                        and self.even_remaining_rounds<required):
                    # Cancel only an UNCOMMITTED new probe. A measured-loser
                    # return, measured lower gain and admitted bridge remain
                    # untouched; all worker prefix/fence proofs still apply.
                    self.even.reject(ticket)
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
        del kwargs
        return self.force(self.even.current_gamma,batch_size,context_tokens,reason='even_round_actual_width_hold')

    def commit_even(self,ticket,batch_size,cost_ms):
        self.even.commit(ticket,transition_cost_ms=cost_ms)
        return self.force(ticket.new,batch_size,0,reason=ticket.reason)

    def summary(self):
        result=super().summary()
        result['even_round_selector']=self.even.summary()
        result['actual_committed_output_counts_used']=True
        result['even_horizon_guard']=dict(enabled=self.even_horizon_enabled,
            new_probes_cancelled=self.even_horizon_cancellations,
            estimated_remaining_rounds=self.even_remaining_rounds,
            minimum_probe_plus_payback_rounds=self.even.window+self.even.amortization_rounds,
            no_labels_or_question_ids_or_fixed_winning_gamma=True)
        return result
