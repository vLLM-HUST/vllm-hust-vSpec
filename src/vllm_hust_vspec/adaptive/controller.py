"""Goodput optimizer with online acceptance feedback."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .profile import AdaptiveProfile


def expected_generated_tokens(acceptance_rate: float, gamma: int) -> float:
    """Return E[accepted Draft tokens + one Target token]."""
    if not 0 <= acceptance_rate <= 1:
        raise ValueError("acceptance_rate must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    return sum(acceptance_rate**position for position in range(gamma + 1))


def expected_generated_tokens_by_position(
    acceptance_rates: Sequence[float],
    gamma: int,
) -> float:
    """Return expected output using conditional acceptance per position."""
    if gamma < 0 or gamma > len(acceptance_rates):
        raise ValueError("gamma must fit the position acceptance vector")
    expected = 1.0
    accepted_prefix_probability = 1.0
    for position in range(gamma):
        rate = acceptance_rates[position]
        if not 0 <= rate <= 1:
            raise ValueError("acceptance rates must be in [0, 1]")
        accepted_prefix_probability *= rate
        expected += accepted_prefix_probability
    return expected


@dataclass(frozen=True)
class ControllerDecision:
    gamma: int
    previous_gamma: int
    acceptance_rate: float
    predicted_goodput: float
    previous_goodput: float
    changed: bool
    reason: str


class GoodputController:
    """Choose a batch-uniform gamma using a latency model and AC EWMA."""

    def __init__(
        self,
        profile: AdaptiveProfile,
        *,
        max_gamma: int,
        min_gamma: int = 1,
        ewma_weight: float = 0.1,
        control_interval: int = 1,
        hysteresis: float = 0.03,
        min_observations: int = 32,
        max_gamma_step: int = 1,
        max_batched_tokens: int | None = None,
        latency_ewma_weight: float = 0.2,
    ) -> None:
        if not 0 <= min_gamma <= max_gamma:
            raise ValueError("min_gamma must be in [0, max_gamma]")
        if max_gamma > profile.max_speculative_tokens:
            raise ValueError(
                "configured max gamma exceeds adaptive profile capacity: "
                f"configured={max_gamma}, profile={profile.max_speculative_tokens}"
            )
        if not 0 < ewma_weight <= 1:
            raise ValueError("ewma_weight must be in (0, 1]")
        if control_interval <= 0 or min_observations <= 0 or max_gamma_step <= 0:
            raise ValueError(
                "control_interval, min_observations, and max_gamma_step must be positive"
            )
        if hysteresis < 0:
            raise ValueError("hysteresis must be nonnegative")
        if max_batched_tokens is not None and max_batched_tokens <= 0:
            raise ValueError("max_batched_tokens must be positive when provided")
        if not 0 < latency_ewma_weight <= 1:
            raise ValueError("latency_ewma_weight must be in (0, 1]")

        self.profile = profile
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        self.ewma_weight = ewma_weight
        self.control_interval = control_interval
        self.hysteresis = hysteresis
        self.min_observations = min_observations
        self.max_gamma_step = max_gamma_step
        self.max_batched_tokens = max_batched_tokens
        self.latency_ewma_weight = latency_ewma_weight
        self.acceptance_rate = profile.default_acceptance_rate
        self.position_acceptance_rates = [profile.default_acceptance_rate] * max_gamma
        self.current_gamma = min(
            max(profile.initial_gamma, min_gamma),
            max_gamma,
        )
        self.total_observations = 0
        self._pending_successes = 0
        self._pending_trials = 0
        self._pending_position_successes = [0] * max_gamma
        self._pending_position_trials = [0] * max_gamma
        self._control_step = 0
        self._warmup_policy_resume_active = False
        self.latency_correction_factors = [1.0] * (max_gamma + 1)
        self.latency_observations = [0] * (max_gamma + 1)

    def observe_step_latency(
        self,
        gamma: int,
        batch_size: int,
        context_tokens: int,
        latency_ms: float,
    ) -> None:
        """Calibrate the offline latency curve from a completed decode step."""
        if not self.min_gamma <= gamma <= self.max_gamma:
            raise ValueError("observed gamma is outside the configured range")
        if batch_size <= 0 or context_tokens < 0:
            raise ValueError("batch_size must be positive and context_tokens nonnegative")
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise ValueError("latency_ms must be finite and positive")

        predicted_ms = self.profile.predict_step_latency_ms(
            gamma,
            batch_size,
            context_tokens,
        )
        # A single transition or host stall must not permanently dominate the
        # online model. The wide clamp still permits substantial correction.
        observed_factor = min(max(latency_ms / predicted_ms, 0.1), 10.0)
        observations = self.latency_observations[gamma]
        if observations == 0:
            updated = observed_factor
        else:
            updated = (
                self.latency_ewma_weight * observed_factor
                + (1 - self.latency_ewma_weight) * self.latency_correction_factors[gamma]
            )
        self.latency_correction_factors[gamma] = updated
        self.latency_observations[gamma] = observations + 1

    def latency_correction_factor(self, gamma: int) -> float:
        """Return a gamma-local correction, or the observed global fallback."""
        if not self.min_gamma <= gamma <= self.max_gamma:
            raise ValueError("gamma is outside the configured range")
        if self.latency_observations[gamma]:
            return self.latency_correction_factors[gamma]
        total = sum(self.latency_observations)
        if total == 0:
            return 1.0
        return (
            sum(
                factor * observations
                for factor, observations in zip(
                    self.latency_correction_factors,
                    self.latency_observations,
                    strict=True,
                )
            )
            / total
        )

    def observe(self, num_draft_tokens: int, num_accepted_tokens: int) -> None:
        if num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be nonnegative")
        if not 0 <= num_accepted_tokens <= num_draft_tokens:
            raise ValueError("accepted tokens must be in [0, num_draft_tokens]")
        if num_draft_tokens == 0:
            return
        if num_draft_tokens > self.max_gamma:
            raise ValueError("draft tokens exceed configured max gamma")

        # Prefix acceptance reveals one failed Bernoulli trial unless all
        # proposed tokens were accepted. Tokens after the first rejection do
        # not provide independent acceptance observations.
        self._pending_successes += num_accepted_tokens
        self._pending_trials += num_accepted_tokens + (num_accepted_tokens < num_draft_tokens)
        for position in range(num_accepted_tokens):
            self._pending_position_successes[position] += 1
            self._pending_position_trials[position] += 1
        if num_accepted_tokens < num_draft_tokens:
            self._pending_position_trials[num_accepted_tokens] += 1

    def predicted_goodput(
        self,
        gamma: int,
        batch_size: int,
        context_tokens: int,
    ) -> float:
        useful_tokens = batch_size * expected_generated_tokens_by_position(
            self.position_acceptance_rates,
            gamma,
        )
        latency_ms = self.profile.predict_step_latency_ms(
            gamma,
            batch_size,
            context_tokens,
        )
        latency_ms *= self.latency_correction_factor(gamma)
        return useful_tokens / latency_ms

    def candidate_is_feasible(self, gamma: int, batch_size: int) -> bool:
        if not self.min_gamma <= gamma <= self.max_gamma:
            return False
        return self.max_batched_tokens is None or (
            batch_size * (gamma + 1) <= self.max_batched_tokens
        )

    def force(
        self,
        gamma: int,
        batch_size: int,
        context_tokens: int,
        *,
        reason: str,
    ) -> ControllerDecision:
        """Force a safe runtime state without discarding acceptance history."""
        if not self.min_gamma <= gamma <= self.max_gamma:
            raise ValueError("forced gamma is outside the configured range")
        previous_gamma = self.current_gamma
        previous_goodput = self.predicted_goodput(
            previous_gamma,
            batch_size,
            context_tokens,
        )
        selected_goodput = self.predicted_goodput(
            gamma,
            batch_size,
            context_tokens,
        )
        self.current_gamma = gamma
        return ControllerDecision(
            gamma=gamma,
            previous_gamma=previous_gamma,
            acceptance_rate=self.acceptance_rate,
            predicted_goodput=selected_goodput,
            previous_goodput=previous_goodput,
            changed=gamma != previous_gamma,
            reason=reason,
        )

    def choose(self, batch_size: int, context_tokens: int) -> ControllerDecision:
        if batch_size <= 0 or context_tokens < 0:
            raise ValueError("batch_size must be positive and context_tokens nonnegative")
        self._control_step += 1
        self._apply_pending_feedback()

        previous_gamma = self.current_gamma
        previous_goodput = self.predicted_goodput(
            previous_gamma,
            batch_size,
            context_tokens,
        )
        feasible_candidates = [
            gamma
            for gamma in range(self.min_gamma, self.max_gamma + 1)
            if self.candidate_is_feasible(gamma, batch_size)
        ]
        if not feasible_candidates:
            raise RuntimeError(
                "no adaptive gamma fits max_num_batched_tokens: "
                f"batch={batch_size}, min_gamma={self.min_gamma}, "
                f"budget={self.max_batched_tokens}"
            )
        if previous_gamma not in feasible_candidates:
            selected_gamma = max(feasible_candidates)
            return self.force(
                selected_gamma,
                batch_size,
                context_tokens,
                reason="batch_token_budget",
            )
        if self.total_observations < self.min_observations:
            policy_gamma = self.profile.preferred_gamma(batch_size)
            if policy_gamma is not None and (
                previous_gamma == 0 or self._warmup_policy_resume_active
            ):
                policy_target = min(
                    max(policy_gamma, self.min_gamma),
                    self.max_gamma,
                )
                policy_target = max(
                    gamma for gamma in feasible_candidates if gamma <= policy_target
                )
                lower = max(
                    self.min_gamma,
                    previous_gamma - self.max_gamma_step,
                )
                upper = min(
                    self.max_gamma,
                    previous_gamma + self.max_gamma_step,
                )
                selected_gamma = min(max(policy_target, lower), upper)
                if selected_gamma != previous_gamma:
                    selected_goodput = self.predicted_goodput(
                        selected_gamma,
                        batch_size,
                        context_tokens,
                    )
                    self.current_gamma = selected_gamma
                    self._warmup_policy_resume_active = selected_gamma != policy_target
                    return ControllerDecision(
                        gamma=selected_gamma,
                        previous_gamma=previous_gamma,
                        acceptance_rate=self.acceptance_rate,
                        predicted_goodput=selected_goodput,
                        previous_goodput=previous_goodput,
                        changed=True,
                        reason=(
                            "batch_policy_warmup"
                            if selected_gamma == policy_target
                            else "batch_policy_warmup_step"
                        ),
                    )
            return ControllerDecision(
                gamma=previous_gamma,
                previous_gamma=previous_gamma,
                acceptance_rate=self.acceptance_rate,
                predicted_goodput=previous_goodput,
                previous_goodput=previous_goodput,
                changed=False,
                reason="warmup",
            )
        if self._control_step % self.control_interval:
            return ControllerDecision(
                gamma=previous_gamma,
                previous_gamma=previous_gamma,
                acceptance_rate=self.acceptance_rate,
                predicted_goodput=previous_goodput,
                previous_goodput=previous_goodput,
                changed=False,
                reason="control_interval",
            )

        policy_gamma = self.profile.preferred_gamma(batch_size)
        if policy_gamma is not None:
            policy_target = min(max(policy_gamma, self.min_gamma), self.max_gamma)
            policy_target = max(gamma for gamma in feasible_candidates if gamma <= policy_target)
            scored = [
                (
                    self.predicted_goodput(gamma, batch_size, context_tokens),
                    gamma,
                )
                for gamma in feasible_candidates
            ]
            best_goodput, best_gamma = max(
                scored,
                key=lambda item: (item[0], -item[1]),
            )
            policy_goodput = self.predicted_goodput(
                policy_target,
                batch_size,
                context_tokens,
            )
            online_override = best_gamma != policy_target and best_goodput > policy_goodput * (
                1 + self.hysteresis
            )
            target_gamma = best_gamma if online_override else policy_target
            lower = max(self.min_gamma, previous_gamma - self.max_gamma_step)
            upper = min(self.max_gamma, previous_gamma + self.max_gamma_step)
            selected_gamma = min(max(target_gamma, lower), upper)
            selected_goodput = self.predicted_goodput(
                selected_gamma,
                batch_size,
                context_tokens,
            )
            self.current_gamma = selected_gamma
            return ControllerDecision(
                gamma=selected_gamma,
                previous_gamma=previous_gamma,
                acceptance_rate=self.acceptance_rate,
                predicted_goodput=selected_goodput,
                previous_goodput=previous_goodput,
                changed=selected_gamma != previous_gamma,
                reason=(
                    ("online_policy_override" if online_override else "batch_policy")
                    if selected_gamma == target_gamma
                    else ("online_policy_override_step" if online_override else "batch_policy_step")
                ),
            )

        scored = [
            (
                self.predicted_goodput(gamma, batch_size, context_tokens),
                gamma,
            )
            for gamma in feasible_candidates
        ]
        best_goodput, best_gamma = max(scored, key=lambda item: (item[0], -item[1]))
        if best_goodput <= previous_goodput * (1 + self.hysteresis):
            return ControllerDecision(
                gamma=previous_gamma,
                previous_gamma=previous_gamma,
                acceptance_rate=self.acceptance_rate,
                predicted_goodput=previous_goodput,
                previous_goodput=previous_goodput,
                changed=False,
                reason="hysteresis",
            )

        lower = max(self.min_gamma, previous_gamma - self.max_gamma_step)
        upper = min(self.max_gamma, previous_gamma + self.max_gamma_step)
        selected_gamma = min(max(best_gamma, lower), upper)
        selected_goodput = self.predicted_goodput(
            selected_gamma,
            batch_size,
            context_tokens,
        )
        if selected_goodput <= previous_goodput * (1 + self.hysteresis):
            return ControllerDecision(
                gamma=previous_gamma,
                previous_gamma=previous_gamma,
                acceptance_rate=self.acceptance_rate,
                predicted_goodput=previous_goodput,
                previous_goodput=previous_goodput,
                changed=False,
                reason="step_limit",
            )
        self.current_gamma = selected_gamma
        return ControllerDecision(
            gamma=selected_gamma,
            previous_gamma=previous_gamma,
            acceptance_rate=self.acceptance_rate,
            predicted_goodput=selected_goodput,
            previous_goodput=previous_goodput,
            changed=selected_gamma != previous_gamma,
            reason="argmax_goodput",
        )

    def _apply_pending_feedback(self) -> None:
        if self._pending_trials == 0:
            return
        observed_rate = self._pending_successes / self._pending_trials
        self.acceptance_rate = (
            self.ewma_weight * observed_rate + (1 - self.ewma_weight) * self.acceptance_rate
        )
        for position, trials in enumerate(self._pending_position_trials):
            if not trials:
                continue
            observed_position_rate = self._pending_position_successes[position] / trials
            self.position_acceptance_rates[position] = (
                self.ewma_weight * observed_position_rate
                + (1 - self.ewma_weight) * self.position_acceptance_rates[position]
            )
        self.total_observations += self._pending_trials
        self._pending_successes = 0
        self._pending_trials = 0
        self._pending_position_successes = [0] * self.max_gamma
        self._pending_position_trials = [0] * self.max_gamma
