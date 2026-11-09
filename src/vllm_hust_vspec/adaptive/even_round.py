"""CPU-only adjacent even-gamma policy; application requires a runtime fence.

This module does not change scheduler/model state, read labels, or synchronize
devices. Requests are proposals only, committed after the runtime's own proof.
"""
from collections import deque, OrderedDict
from dataclasses import dataclass
import math
import statistics

_POST_RPC_STEADY_OBSERVATIONS = 2


@dataclass(frozen=True)
class Transition:
    sequence: int
    old: int
    new: int
    reason: str


class EvenRoundSelector:
    def __init__(self, *, maximum=8, window=16, hysteresis=0.03,
                 change_trigger=0.15, amortization_rounds=64):
        if maximum < 2 or maximum > 8 or maximum % 2 or window < 2:
            raise ValueError('Require even maximum in [2,8] and window>=2')
        if not 0 <= hysteresis < 1 or not 0 < change_trigger < 1 or amortization_rounds < window:
            raise ValueError('Invalid stability/cost contract')
        self.candidates = tuple(range(2, maximum + 1, 2))
        self.current_gamma = self.incumbent = 2
        self.window, self.hysteresis = window, hysteresis
        self.change_trigger, self.amortization_rounds = change_trigger, amortization_rounds
        self.samples = {g: deque(maxlen=window) for g in self.candidates}
        self.context_memory = OrderedDict()
        self.maximum_contexts = 128
        self.pending = None
        self.sequence = self.rounds = 0
        self.trial = None
        self.rejected_at = {}
        self.switch_cost_ms = 0.0
        self.context = None
        self.decisions = []
        # A rejected device admission or a new geometry must not erase an
        # already measured losing-probe return. This is policy intent ONLY;
        # current_gamma still changes exclusively through the native fence.
        self.deferred_measured_return = None
        self.deferred_return_reissues = 0
        # Per-engine live latency scalars only: no imported profile, labels,
        # generated answers, cached KV, model tensors or fixed winning gamma.
        self.cost_profiles = {}
        self.cost_last_geometry = None
        self.cost_stable_rounds = 0
        self.cost_gate_checks = self.cost_gate_vetoes = 0
        self.cost_gate_last = None
        self.lower_gate_checks = self.lower_gate_vetoes = 0
        self.lower_gate_last = None
        self.steady_feedback_wait = 0
        self.steady_feedback_excluded = 0
        self.steady_settling_observations = _POST_RPC_STEADY_OBSERVATIONS
        self.unmeasured_probe_cancellations = 0
        self.bootstrap_committed_widths = {2}
        self.bootstrap_unmeasured_higher_proposals = 0
        # Fresh-engine integer prefix observations only; never imported
        # latency tables, task labels, model values or a winning gamma.
        self.higher_tail_profiles = {}
        self.higher_tail_last_round = {}
        self.higher_tail_checks = self.higher_tail_applied = self.higher_tail_shifted = 0
        self.higher_tail_changed_vetoes = 0

    def _observe_higher_tail(self, gamma, accepted, committed, context):
        if not (isinstance(context,tuple) and context and type(context[0]) is int
                and context[0]==len(accepted) and self.cost_stable_rounds>2):
            return
        key=(len(accepted),gamma)
        self.higher_tail_profiles.setdefault(key,deque(maxlen=32)).append((tuple(accepted),tuple(committed)))
        self.higher_tail_last_round[key]=self.rounds+1

    def _measured_higher_tail_bound(self, higher, rows, full):
        context=self.context;gamma=self.current_gamma
        if not (isinstance(context,tuple) and context and type(context[0]) is int):return None
        key=(context[0],higher);profile=self.higher_tail_profiles.get(key,())
        if len(profile)<4:return None
        observed_total=sum(len(a) for a,c in profile)
        surviving=sum(a>=gamma and c>=gamma+1 for aa,cc in profile for a,c in zip(aa,cc))
        current_total=sum(len(r[2]) for r in rows)
        if surviving<4 or not current_total:return None
        current_fraction=full/current_total;observed_fraction=surviving/observed_total
        # The measured quantity is CONDITIONAL on surviving this prefix.
        # Changes in the fraction failing much earlier do not invalidate that
        # conditional tail. Compare the same last-prefix-token hazard instead
        # of comparing unrelated unconditional full-prefix population shares.
        previous=sum(a>=gamma-1 and c>=gamma for r in rows for a,c in zip(r[2],r[3]))
        observed_previous=sum(a>=gamma-1 and c>=gamma for aa,cc in profile for a,c in zip(aa,cc))
        if previous<4 or observed_previous<4:return None
        current_conditional=full/previous;observed_conditional=surviving/observed_previous
        shift=abs(current_conditional/observed_conditional-1)
        self.higher_tail_checks+=1
        result=dict(method='live_measured_deeper_tail_conditioned_on_surviving_current_prefix',
            batch=context[0],observed_gamma=higher,observed_frames=len(profile),
            observed_surviving_current_prefixes=surviving,
            current_full_prefix_fraction=current_fraction,observed_full_prefix_fraction=observed_fraction,
            shift_basis='conditional_survival_of_last_current_prefix_token',
            current_previous_prefixes=previous,observed_previous_prefixes=observed_previous,
            current_last_prefix_conditional_survival=current_conditional,
            observed_last_prefix_conditional_survival=observed_conditional,
            relative_prefix_distribution_shift=shift,existing_change_trigger=self.change_trigger,
            last_observed_policy_round=self.higher_tail_last_round[key],
            age_in_policy_observations=self.rounds-self.higher_tail_last_round[key],
            no_labels_imported_profile_or_fixed_winner=True)
        if shift>=self.change_trigger:
            # A changed live conditional prefix hazard makes the old deeper-tail
            # observation inapplicable. This permits fresh re-evaluation,
            # not a periodic forced sweep or permanently forbidden arm.
            self.higher_tail_shifted+=1
            result.update(applied=False,reason='material_live_prefix_distribution_change')
            return result
        useful_extra=sum(max(0,min(c,min(a,higher)+1)-(gamma+1))
            for aa,cc in profile for a,c in zip(aa,cc) if a>=gamma and c>=gamma+1)
        conditional_extra=useful_extra/surviving
        result.update(applied=True,observed_committed_extra=useful_extra,
            conditional_extra_per_surviving_prefix=conditional_extra,
            extra_bound_for_current_window=full*conditional_extra)
        self.higher_tail_applied+=1
        return result

    def _observe_cost(self, gamma, latency_ms, context, batch):
        if not (isinstance(context,tuple) and context and type(context[0]) is int
                and context[0]==batch and batch>0):
            return
        geometry=(context,gamma)
        if geometry==self.cost_last_geometry:self.cost_stable_rounds+=1
        else:self.cost_last_geometry=geometry;self.cost_stable_rounds=1
        # Filter transition/geometry warm boundaries for the ESTIMATOR only.
        # Benchmark wall time still includes ALL work, fences and cohort gaps.
        if self.cost_stable_rounds<=2:return
        self.cost_profiles.setdefault((batch,gamma),deque(maxlen=32)).append(latency_ms)

    def _allow_higher(self, higher, rows):
        context=self.context;gamma=self.current_gamma
        if not (isinstance(context,tuple) and context and type(context[0]) is int):return True
        batch=context[0]
        costs={g:statistics.median(v) for (n,g),v in self.cost_profiles.items() if n==batch and len(v)>=4}
        if higher not in costs and higher not in self.bootstrap_committed_widths:
            # One initial exposure per previously uncommitted width, never a
            # periodic sweep. A fitted cost line cannot know the first real
            # cost of another shape-specific kernel. Existing full-window,
            # remaining-horizon and all device admission proofs still apply.
            self.bootstrap_unmeasured_higher_proposals += 1
            self.cost_gate_checks += 1
            self.cost_gate_last=dict(batch=batch,old=gamma,new=higher,
                feedback_round=self.rounds,method='bounded_initial_unmeasured_width_trial',
                current_goodput=0.,predicted_goodput=0.,allowed=True,
                no_predicted_speed_gain_claimed=True)
            return True
        if len(costs)<2 or gamma not in costs:return True
        actual_ms=sum(r[1] for r in rows);useful=self._utility(rows)
        if not actual_ms or not useful:return True
        current_ms=statistics.median(r[1] for r in rows)
        if higher in costs:
            predicted_ms=current_ms*costs[higher]/costs[gamma]
            uncertainty=0.0;method='observed_same_batch_cost_ratio'
        else:
            low,high=min(costs),max(costs)
            slope=max(0.0,(costs[high]-costs[low])/(high-low))
            predicted_ms=current_ms+slope*(higher-gamma)
            uncertainty=.05;method='online_two_width_affine_cost'
        # Fresh verifier-prefix survival estimates the extra useful suffix.
        # EOS/truncation-censored rows cannot claim hypothetical extra tokens.
        full=sum(a==gamma and c>=gamma+1 for r in rows for a,c in zip(r[2],r[3]))
        previous=sum(a>=gamma-1 and c>=gamma for r in rows for a,c in zip(r[2],r[3]))
        survival=full/previous if previous else 0.0
        extra=full*sum(survival**k for k in range(1,higher-gamma+1))
        if self._aligned_roster(rows):
            # Same actual requests, in stable identity order. A fast majority
            # cannot pay for a wider grade which leaves the bottleneck behind.
            totals=self._request_totals(rows)
            future=[]
            for i,total in enumerate(totals):
                f=sum(r[2][i]==gamma and r[3][i]>=gamma+1 for r in rows)
                p=sum(r[2][i]>=gamma-1 and r[3][i]>=gamma for r in rows)
                probability=f/p if p else 0.
                future.append(total+f*sum(probability**k for k in range(1,higher-gamma+1)))
            extra=max(0.,self._fair_progress(future)-useful)
        extrapolated_extra=extra
        measured_tail=self._measured_higher_tail_bound(higher,rows,full)
        if measured_tail is not None and measured_tail['applied']:
            extra=min(extra,measured_tail['extra_bound_for_current_window'])
        charged=self.switch_cost_ms*self.window/self.amortization_rounds
        predicted_total=actual_ms*predicted_ms/current_ms+charged
        predicted_rate=(useful+extra)/predicted_total
        shallow_extrapolated_rate=(useful+extrapolated_extra)/predicted_total
        current_rate=useful/actual_ms
        required=current_rate*(1+self.hysteresis)*(1+uncertainty)
        allowed=predicted_rate>required
        self.cost_gate_checks+=1
        self.cost_gate_vetoes+=not allowed
        self.cost_gate_last=dict(batch=batch,old=gamma,new=higher,method=method,
            feedback_round=self.rounds,
            live_measured_widths=sorted(costs),predicted_round_ms=predicted_ms,
            current_round_ms=current_ms,prefix_tail_survival=survival,
            predicted_useful_extra=extra,predicted_goodput=predicted_rate,
            uncapped_shallow_prefix_extrapolated_extra=extrapolated_extra,
            uncapped_shallow_prefix_predicted_goodput=shallow_extrapolated_rate,
            live_measured_higher_tail=measured_tail,
            current_goodput=current_rate,required_goodput=required,
            unknown_width_uncertainty=uncertainty,allowed=allowed)
        self.cost_gate_last['utility_basis']=self._utility_basis(rows)
        if (measured_tail is not None and measured_tail['applied']
                and shallow_extrapolated_rate>required and not allowed):
            self.higher_tail_changed_vetoes+=1
            if self.higher_tail_changed_vetoes<=8 or self.higher_tail_changed_vetoes%64==0:
                print('[vspec-measured-deeper-tail-veto] '+repr(self.cost_gate_last),flush=True)
        return allowed

    def _allow_unmeasured_lower(self, lower, rows):
        # A missing SAME-context lower window is not evidence that the lower
        # arm deserves priority. Reuse only fresh-engine measured scalar
        # latency ratios; all utility comes from THIS current prefix window.
        context=self.context;gamma=self.current_gamma
        if not (isinstance(context,tuple) and context and type(context[0]) is int):return True
        batch=context[0]
        costs={g:statistics.median(v) for (n,g),v in self.cost_profiles.items() if n==batch and len(v)>=4}
        if gamma not in costs:return True
        actual_ms=sum(r[1] for r in rows);useful=self._utility(rows)
        if not actual_ms or not useful:return True
        current_ms=statistics.median(r[1] for r in rows)
        transferred=[];uncertainty=0.
        method='fresh_prefix_utility_and_live_same_batch_measured_cost_ratio'
        if lower in costs:
            predicted_ms=current_ms*costs[lower]/costs[gamma]
        else:
            # Only current-engine scalar timings: a missing tail-batch arm
            # need not silently fall back to an80-round impossible probe.
            # Use the MOST conservative observed lower/current ratio, and
            # retain the existing5percent unknown-shape uncertainty margin.
            for n in sorted({n for n,g in self.cost_profiles}):
                lo=self.cost_profiles.get((n,lower),());hi=self.cost_profiles.get((n,gamma),())
                if len(lo)>=4 and len(hi)>=4:
                    ratio=statistics.median(lo)/statistics.median(hi)
                    if math.isfinite(ratio) and ratio>0:transferred.append((n,ratio))
            if not transferred:return True
            predicted_ms=current_ms*max(r for n,r in transferred)
            uncertainty=.05;method='same_engine_cross_batch_relative_cost_upper_bound'
        lower_useful=self._utility(rows,lower)
        charged=self.switch_cost_ms*len(rows)/self.amortization_rounds
        predicted_rate=lower_useful/(actual_ms*predicted_ms/current_ms+charged)
        current_rate=useful/actual_ms
        required=current_rate*(1+self.hysteresis)*(1+uncertainty)
        allowed=predicted_rate>required
        self.lower_gate_checks+=1;self.lower_gate_vetoes+=not allowed
        self.lower_gate_last=dict(batch=batch,old=gamma,new=lower,
            feedback_round=self.rounds,
            method=method,cross_batch_relative_cost_samples=transferred,
            unknown_shape_uncertainty=uncertainty,required_goodput=required,
            live_measured_widths=sorted(costs),predicted_round_ms=predicted_ms,
            current_round_ms=current_ms,predicted_useful_tokens=lower_useful,
            current_useful_tokens=useful,predicted_goodput=predicted_rate,
            current_goodput=current_rate,allowed=allowed)
        self.lower_gate_last['utility_basis']=self._utility_basis(rows)
        return allowed

    def _request(self, new, reason):
        if new == self.current_gamma:
            return None
        if new not in self.candidates or abs(new-self.current_gamma) != 2:
            raise ValueError('Round policy changes one even-gamma grade at a time')
        self.sequence += 1
        self.pending = Transition(self.sequence, self.current_gamma, new, reason)
        self.decisions.append(self.pending)
        return self.pending

    def commit(self, transition, *, transition_cost_ms):
        if transition != self.pending or transition.old != self.current_gamma:
            raise ValueError('Stale, unrequested or repeated transition')
        if not math.isfinite(transition_cost_ms) or transition_cost_ms < 0:
            raise ValueError('Require actual nonnegative transition wall time')
        self.switch_cost_ms = transition_cost_ms
        # Match the existing cost-profile's two-round settling filter. Only
        # selection estimation skips these boundary samples; all generated
        # tokens, acceptance feedback and benchmark wall time remain intact.
        self.steady_feedback_wait = _POST_RPC_STEADY_OBSERVATIONS if transition_cost_ms > 0 else 0
        self.current_gamma = transition.new
        self.bootstrap_committed_widths.add(transition.new)
        if transition.new == self.deferred_measured_return:
            self.deferred_measured_return = None
            self.incumbent = transition.new
        self.samples[transition.new].clear()
        self.pending = None

    def reject(self, transition):
        if transition != self.pending:
            raise ValueError('Unknown transition')
        rows = self.samples[self.current_gamma]
        if rows:
            self.rejected_at[(self.context,transition.new)] = self._rate(rows)
        rows.clear()
        self.pending = None
        if self.trial is not None:
            self.trial = None

    def cancel_unmeasured_probe(self, transition):
        if (transition != self.pending or transition.old != self.current_gamma
                or transition.reason not in ('adjacent_higher_trial', 'adjacent_lower_trial_missing_context')):
            raise ValueError('Only an uncommitted new probe may be cancelled without a loss record')
        # No device trial happened. A short horizon is not evidence that this
        # arm is slow. Preserve genuine past loss records but add no fake one.
        self.samples[self.current_gamma].clear()
        self.pending = None
        self.trial = None
        self.unmeasured_probe_cancellations += 1

    def _aligned_roster(self,rows):
        return bool(rows and isinstance(self.context,tuple) and len(self.context)==3
            and isinstance(self.context[2],tuple) and len(self.context[2])==len(rows[0][3])
            and all(len(r[3])==len(self.context[2]) for r in rows))

    @staticmethod
    def _request_totals(rows,clip_gamma=None):
        n=len(rows[0][3])
        return [sum(r[3][i] if clip_gamma is None else min(r[3][i],min(r[2][i],clip_gamma)+1)
            for r in rows) for i in range(n)]

    def _utility_basis(self,rows):
        return ('harmonic_same_request_window_progress' if self._aligned_roster(rows)
            else 'legacy_aggregate_without_identity_alignment')

    @staticmethod
    def _fair_progress(totals):
        # Harmonic progress is the reciprocal of mean per-request time per
        # token. It respects slow requests without letting one noisy minimum
        # discard the genuine bulk throughput of every other request.
        # Divide the harmonic mean by roster size. Constant within a context,
        # so comparisons are identical; each coordinate derivative is<=1,
        # making the existing aggregate-tail bound conservative in score units.
        return 1./sum(1./x for x in totals) if totals and min(totals)>0 else 0.

    def _utility(self,rows,clip_gamma=None):
        if not rows:return 0
        if self._aligned_roster(rows):return self._fair_progress(self._request_totals(rows,clip_gamma))
        return (sum(r[0] for r in rows) if clip_gamma is None else
            sum(min(c,min(a,clip_gamma)+1) for r in rows for a,c in zip(r[2],r[3])))

    def _rate(self,rows,extra_ms=0.0):
        return self._utility(rows)/(sum(r[1] for r in rows)+extra_ms)

    def observe(self, *, gamma, accepted, committed, latency_ms, context):
        """One completed stable verifier round, with actual request-level counts.

        `accepted` is accepted draft prefix length, `committed` counts actual
        emitted tokens including the target token (EOS/tail truncation allowed).
        Context must identify comparable serving geometry, not question labels.
        """
        if gamma != self.current_gamma or gamma not in self.candidates:
            raise ValueError('Only feedback for the actually committed width')
        if not accepted or len(accepted) != len(committed):
            raise ValueError('Require same nonempty actual request roster')
        if any(type(a) is not int or not 0 <= a <= gamma for a in accepted):
            raise ValueError('Invalid accepted prefix')
        if any(type(c) is not int or not 0 <= c <= a+1 for a,c in zip(accepted,committed)):
            raise ValueError('Committed count exceeds actual verified output')
        if not math.isfinite(latency_ms) or latency_ms <= 0 or sum(committed) <= 0:
            raise ValueError('Require finite positive latency and useful progress')
        hash(context)
        if self.pending is not None and self.context != context:
            raise ValueError('Runtime must reject pending transition before geometry changes')
        self._observe_cost(gamma,latency_ms,context,len(accepted))
        self._observe_higher_tail(gamma,accepted,committed,context)
        if self.context != context:
            self.context = context
            if context not in self.context_memory:
                self.context_memory[context] = {g: deque(maxlen=self.window) for g in self.candidates}
            self.context_memory.move_to_end(context)
            while len(self.context_memory) > self.maximum_contexts:
                expired, _ = self.context_memory.popitem(last=False)
                self.rejected_at = {k:v for k,v in self.rejected_at.items() if k[0] != expired}
            self.samples = self.context_memory[context]
            # Reuse only measured same-geometry alternative-arm latency.
            # Current acceptance/output MUST be a fresh window; historical
            # question rewards are never used as the current round's reward.
            self.samples[gamma].clear()
            self.trial = None
            self.incumbent = (self.current_gamma if self.deferred_measured_return is None
                              else self.deferred_measured_return)
        self.rounds += 1
        rows = self.samples[gamma]
        if self.steady_feedback_wait:
            self.steady_feedback_wait -= 1
            self.steady_feedback_excluded += 1
            return self.pending
        rows.append((sum(committed), latency_ms, tuple(accepted), tuple(committed)))
        if self.deferred_measured_return is not None and self.pending is None:
            # The wrapper has rejected the OLD ticket before changing its
            # geometry. Reissue a fresh ticket; never reuse old worker proofs,
            # roster, accepted counts or cache state. The unchanged runtime
            # must still prove every prefix, allocation and device fence.
            # This fallback does NOT assert that an old-context rate is valid
            # here. Normal same-context learning resumes after safe return.
            self.deferred_return_reissues += 1
            return self._request(self.deferred_measured_return,
                                 'deferred_measured_probe_not_better_return')
        if self.pending is not None or len(rows) < self.window:
            return self.pending
        rate = self._rate(rows)
        if rate<=0:
            # A wholly nonprogressing member has no finite measured completion
            # rate. Hold this estimator decision; physical feedback and ALL
            # generated work/wall time remain recorded by the caller.
            return self.pending
        if self.trial is not None:
            old, old_rate = self.trial
            # Charge the actual switch cost over a fixed, declared horizon.
            charged = self.switch_cost_ms*self.window/self.amortization_rounds
            if self._rate(rows,charged) > old_rate*(1+self.hysteresis):
                self.incumbent = gamma
                self.trial = None
            else:
                self.rejected_at[(self.context,gamma)] = old_rate
                self.trial = None
                self.incumbent = old
                self.deferred_measured_return = old
                return self._request(old,'measured_probe_not_better_return')
        lower = gamma-2
        lower_rows = self.samples.get(lower)
        if lower_rows and len(lower_rows) == self.window:
            # Lower-gamma useful work is observable from this round's accepted
            # prefixes; its latency is an explicit same-context measured prior.
            lower_tokens = self._utility(rows,lower)
            lower_time = sum(r[1] for r in lower_rows)
            projected = lower_tokens/(lower_time+self.switch_cost_ms*self.window/self.amortization_rounds)
            if projected > rate*(1+self.hysteresis):
                self.incumbent = lower
                return self._request(lower,'lower_grade_counterfactual_gain')
        elif lower in self.candidates:
            rejected = self.rejected_at.get((self.context,lower))
            if ((rejected is None or abs(rate/rejected-1) >= self.change_trigger)
                    and self._allow_unmeasured_lower(lower,rows)):
                self.trial = (gamma,rate)
                return self._request(lower,'adjacent_lower_trial_missing_context')
        higher = gamma+2
        if higher <= self.candidates[-1]:
            rejected = self.rejected_at.get((self.context,higher))
            # No periodic forced sweep. Revisit a losing neighbor only after a
            # material serving-rate change, or a different geometry context.
            if rejected is None or abs(rate/rejected-1) >= self.change_trigger:
                if not self._allow_higher(higher,rows):return None
                self.trial = (gamma,rate)
                return self._request(higher,'adjacent_higher_trial')
        return None

    def summary(self):
        return dict(candidates=list(self.candidates),initial_gamma=2,current_gamma=self.current_gamma,
                    incumbent=self.incumbent,rounds=self.rounds,window=self.window,
                    pending=None if self.pending is None else self.pending.__dict__,
                    decisions=[d.__dict__ for d in self.decisions],
                    context=self.context,context_memory_size=len(self.context_memory),
                    window_objective=dict(basis='harmonic_same_request_window_progress_when_identity_aligned',
                        surrogate_normalization='harmonic_mean_divided_by_roster_size',
                        request_identity_alignment_required=True,labels_or_answers_used=False,
                        original_aggregate_output_statistics_still_recorded=True,
                        benchmark_timer_and_workload_NOT_changed=True),
                    last_switch_policy_cost_ms=self.switch_cost_ms,
                    steady_feedback_estimator=dict(post_RPC_observations_waiting=self.steady_feedback_wait,
                        boundary_samples_excluded_from_estimator_ONLY=self.steady_feedback_excluded,
                        all_generated_tokens_and_benchmark_wall_still_included=True,
                        original_online_accepted_prefix_feedback_still_updated=True),
                    switch_policy_cost_definition='admission_RPC_wall_only; benchmark still includes all queue/bridge/fence work',
                    deferred_measured_return=self.deferred_measured_return,
                    deferred_return_reissues=self.deferred_return_reissues,
                    unmeasured_probe_cancellations_without_fake_loss=self.unmeasured_probe_cancellations,
                    initial_width_calibration=dict(committed_widths=sorted(self.bootstrap_committed_widths),
                        unmeasured_higher_proposals=self.bootstrap_unmeasured_higher_proposals,
                        once_per_previously_uncommitted_width_NOT_periodic_sweep=True,
                        no_winning_gamma_or_imported_profile=True),
                    online_cost_gate=dict(checks=self.cost_gate_checks,vetoes=self.cost_gate_vetoes,
                        last_prediction=self.cost_gate_last,profiled_batch_widths=len(self.cost_profiles),
                        cost_profiles_live_per_engine_only=True,no_imported_winner_or_latency_table=True),
                    measured_deeper_tail_model=dict(profiled_batch_widths=len(self.higher_tail_profiles),
                        shift_basis='conditional_survival_of_last_current_prefix_token',
                        checks=self.higher_tail_checks,applied=self.higher_tail_applied,
                        ignored_for_material_prefix_shift=self.higher_tail_shifted,
                        shallow_allowed_but_measured_vetoed=self.higher_tail_changed_vetoes,
                        existing_change_trigger=self.change_trigger,
                        no_periodic_forced_sweep=True,no_fixed_winning_gamma=True,
                        scalar_prefix_counts_only_live_per_engine=True),
                    online_lower_probe_gate=dict(checks=self.lower_gate_checks,vetoes=self.lower_gate_vetoes,
                        last_prediction=self.lower_gate_last,
                        no_gamma_hardcoded_winner=True,no_imported_profile=True,
                        missing_lower_window_does_not_unconditionally_starve_higher_neighbor=True),
                    context_windows=[dict(context=k,arms={str(g):dict(n=len(rows),
                        useful_tokens=sum(r[0] for r in rows),latency_ms=sum(r[1] for r in rows))
                        for g,rows in arms.items()}) for k,arms in self.context_memory.items()],
                    real_runtime_fence_required=True,no_model_or_speed_or_quality_admission=True)
