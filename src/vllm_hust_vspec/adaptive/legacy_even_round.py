"""CPU-only adjacent even-gamma policy; application requires a runtime fence.

This module does not change scheduler/model state, read labels, or synchronize
devices. Requests are proposals only, committed after the runtime's own proof.
"""
from collections import deque, OrderedDict
from dataclasses import dataclass
import math


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
        self.current_gamma = self.incumbent = self.candidates[-1]
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
        self.current_gamma = transition.new
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

    @staticmethod
    def _rate(rows, extra_ms=0.0):
        return sum(r[0] for r in rows)/(sum(r[1] for r in rows)+extra_ms)

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
            self.incumbent = self.current_gamma
        self.rounds += 1
        rows = self.samples[gamma]
        rows.append((sum(committed), latency_ms, tuple(accepted), tuple(committed)))
        if self.pending is not None or len(rows) < self.window:
            return self.pending
        rate = self._rate(rows)
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
                return self._request(old,'measured_probe_not_better_return')
        lower = gamma-2
        lower_rows = self.samples.get(lower)
        if lower_rows and len(lower_rows) == self.window:
            # Lower-gamma useful work is observable from this round's accepted
            # prefixes; its latency is an explicit same-context measured prior.
            lower_tokens = sum(sum(min(c,min(a,lower)+1) for a,c in zip(r[2],r[3])) for r in rows)
            lower_time = sum(r[1] for r in lower_rows)
            projected = lower_tokens/(lower_time+self.switch_cost_ms*self.window/self.amortization_rounds)
            if projected > rate*(1+self.hysteresis):
                self.incumbent = lower
                return self._request(lower,'lower_grade_counterfactual_gain')
        elif lower in self.candidates:
            rejected = self.rejected_at.get((self.context,lower))
            if rejected is None or abs(rate/rejected-1) >= self.change_trigger:
                self.trial = (gamma,rate)
                return self._request(lower,'adjacent_lower_trial_missing_context')
        higher = gamma+2
        if higher <= self.candidates[-1]:
            rejected = self.rejected_at.get((self.context,higher))
            # No periodic forced sweep. Revisit a losing neighbor only after a
            # material serving-rate change, or a different geometry context.
            if rejected is None or abs(rate/rejected-1) >= self.change_trigger:
                self.trial = (gamma,rate)
                return self._request(higher,'adjacent_higher_trial')
        return None

    def summary(self):
        return dict(candidates=list(self.candidates),initial_gamma=self.candidates[-1],current_gamma=self.current_gamma,
                    incumbent=self.incumbent,rounds=self.rounds,window=self.window,
                    pending=None if self.pending is None else self.pending.__dict__,
                    decisions=[d.__dict__ for d in self.decisions],
                    context=self.context,context_memory_size=len(self.context_memory),
                    context_windows=[dict(context=k,arms={str(g):dict(n=len(rows),
                        useful_tokens=sum(r[0] for r in rows),latency_ms=sum(r[1] for r in rows))
                        for g,rows in arms.items()}) for k,arms in self.context_memory.items()],
                    real_runtime_fence_required=True,no_model_or_speed_or_quality_admission=True)
