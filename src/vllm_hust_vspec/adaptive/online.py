"""Profile-free online gamma control for serial speculative decoding."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .controller import ControllerDecision


@dataclass
class _ArmWindow:
    useful_tokens: deque[float]
    latency_ms: deque[float]
    total_observations: int = 0
    prior_reward: float = 0.0
    prior_weight: int = 0

    @classmethod
    def create(cls, window_size: int) -> _ArmWindow:
        return cls(
            useful_tokens=deque(maxlen=window_size),
            latency_ms=deque(maxlen=window_size),
        )

    @property
    def observations(self) -> int:
        return self.total_observations

    @property
    def effective_observations(self) -> int:
        return self.total_observations + self.prior_weight

    @property
    def mean_reward(self) -> float:
        total_latency = sum(self.latency_ms)
        measured_reward = sum(self.useful_tokens) / total_latency if total_latency else 0.0
        if not self.prior_weight:
            return measured_reward
        if not self.total_observations:
            return self.prior_reward
        return (
            measured_reward * self.total_observations + self.prior_reward * self.prior_weight
        ) / self.effective_observations

    def record(self, useful_tokens: int, latency_ms: float) -> None:
        self.useful_tokens.append(float(useful_tokens))
        self.latency_ms.append(latency_ms)
        self.total_observations += 1


@dataclass
class _BatchBucket:
    arms: dict[int, _ArmWindow] = field(default_factory=dict)
    selected_gamma: int | None = None
    selected_at_observations: int = 0
    scheduled_since_selection: int = 0
    launched_arms: set[int] = field(default_factory=set)
    warmup_return_completed: bool = False
    warmup_target_gamma: int | None = None
    bootstrap_target_gamma: int | None = None
    seeded_from_bucket: int | None = None
    seed_initialized: bool = False
    frontier_deferred_at_observations: int | None = None
    anchor_revalidation_at_observations: int | None = None


class OnlineGammaController:
    """Learn the best gamma directly from completed serving steps.

    The reward is committed output tokens divided by end-to-end stable decode
    latency. Measurements and UCB confidence are isolated by power-of-two batch
    buckets. Every feasible gamma is an arm; no offline profile is required.
    """

    def __init__(
        self,
        *,
        max_gamma: int,
        min_gamma: int = 1,
        ewma_weight: float = 0.1,
        control_interval: int = 1,
        hysteresis: float = 0.03,
        max_gamma_step: int = 1,
        max_batched_tokens: int | None = None,
        window_size: int = 32,
        exploration: float = 0.15,
        warmup_samples: int = 4,
        warmup_return: str = "best",
        burn_in_steps: int | None = None,
        inflight_warmup_padding: int = 0,
        initial_gamma: int | None = None,
        probe_shared_cost_units: float = 2.0,
        candidate_stride: int = 1,
        candidate_batch_limits: dict[int, int] | None = None,
    ) -> None:
        if not 0 <= min_gamma <= max_gamma:
            raise ValueError("min_gamma must be in [0, max_gamma]")
        if not 0 < ewma_weight <= 1:
            raise ValueError("ewma_weight must be in (0, 1]")
        if control_interval <= 0 or max_gamma_step <= 0:
            raise ValueError("control_interval and max_gamma_step must be positive")
        if hysteresis < 0 or exploration < 0:
            raise ValueError("hysteresis and exploration must be nonnegative")
        if window_size <= 0 or warmup_samples <= 0:
            raise ValueError("window_size and warmup_samples must be positive")
        if warmup_return not in {"best", "incumbent"}:
            raise ValueError("warmup_return must be best or incumbent")
        if burn_in_steps is not None and burn_in_steps < 0:
            raise ValueError("burn_in_steps must be nonnegative")
        if inflight_warmup_padding < 0:
            raise ValueError("inflight_warmup_padding must be nonnegative")
        if max_batched_tokens is not None and max_batched_tokens <= 0:
            raise ValueError("max_batched_tokens must be positive when provided")
        if not math.isfinite(probe_shared_cost_units) or probe_shared_cost_units <= 0:
            raise ValueError("probe_shared_cost_units must be finite and positive")
        if candidate_stride <= 0:
            raise ValueError("candidate_stride must be positive")
        if (max_gamma - min_gamma) % candidate_stride:
            raise ValueError("candidate_stride must include max_gamma")
        if max_gamma > min_gamma and max_gamma_step < candidate_stride:
            raise ValueError("max_gamma_step must reach the next candidate gamma")

        self.candidate_gammas = tuple(range(min_gamma, max_gamma + 1, candidate_stride))
        batch_limits = dict(candidate_batch_limits or {})
        invalid_limits = set(batch_limits).difference(self.candidate_gammas)
        if invalid_limits:
            raise ValueError("candidate_batch_limits contains a non-candidate gamma")
        if any(limit <= 0 for limit in batch_limits.values()):
            raise ValueError("candidate batch limits must be positive")
        if initial_gamma is not None and initial_gamma not in self.candidate_gammas:
            raise ValueError("initial_gamma must be one of candidate_gammas")

        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        self.candidate_stride = candidate_stride
        self.candidate_batch_limits = batch_limits
        numeric_midpoint = (min_gamma + max_gamma + 1) // 2
        midpoint_gamma = min(
            self.candidate_gammas,
            key=lambda gamma: (abs(gamma - numeric_midpoint), -gamma),
        )
        self.initial_gamma = initial_gamma
        self.anchor_gamma = initial_gamma if initial_gamma is not None else midpoint_gamma
        self.ewma_weight = ewma_weight
        self.control_interval = control_interval
        self.hysteresis = hysteresis
        self.max_gamma_step = max_gamma_step
        self.max_batched_tokens = max_batched_tokens
        self.window_size = window_size
        self.exploration = exploration
        self.warmup_samples = warmup_samples
        self.warmup_return = warmup_return
        self.inflight_warmup_padding = inflight_warmup_padding
        self.probe_shared_cost_units = probe_shared_cost_units
        self.model_probe_min_observations = window_size
        self.deferred_exploration_observations = (
            window_size * 8 if min_gamma == 0 and max_gamma - min_gamma >= 4 else 0
        )
        self._burn_in_remaining = warmup_samples if burn_in_steps is None else burn_in_steps
        # An explicit initial arm is only a cold-start prior. Every feasible arm
        # remains available to warmup and UCB selection.
        self.current_gamma = (
            initial_gamma
            if initial_gamma is not None
            else self.anchor_gamma
            if self.deferred_exploration_observations
            else max_gamma
        )
        self.acceptance_rate = 1.0
        self.position_acceptance_rates = [1.0] * max_gamma
        self.total_observations = 0
        self.position_observations = [0] * max_gamma
        self.latency_observations = [0] * (max_gamma + 1)
        self._latency_sums = [0.0] * (max_gamma + 1)
        self._buckets: dict[int, _BatchBucket] = {}
        self.selection_counts = [0] * (max_gamma + 1)
        self.gamma_switches = 0
        self.reason_counts: dict[str, int] = {}

        self._pending_successes = 0
        self._pending_trials = 0
        self._pending_position_successes = [0] * max_gamma
        self._pending_position_trials = [0] * max_gamma
        self._step_useful_tokens = 0
        self._step_requests = 0
        self._last_completion_ms: float | None = None
        self._active_bucket_key: int | None = None
        self._request_previous_acceptance: dict[str, tuple[int, int]] = {}
        self._second_token_transition = [[0, 0], [0, 0]]

    @staticmethod
    def batch_bucket(batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return 1 << (batch_size - 1).bit_length()

    def _bucket(self, batch_size: int) -> _BatchBucket:
        key = self.batch_bucket(batch_size)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _BatchBucket()
            self._buckets[key] = bucket
        for gamma in self.candidate_gammas:
            bucket.arms.setdefault(gamma, _ArmWindow.create(self.window_size))
        should_refresh_seed = (
            bucket.seeded_from_bucket is not None
            and not bucket.warmup_return_completed
            and len(bucket.launched_arms) < len(bucket.arms)
        )
        if not bucket.seed_initialized or should_refresh_seed:
            # A readiness request can create B1 before the production buckets
            # exist. Keep retrying until an adjacent bucket has useful online
            # state, otherwise that early bucket performs a costly cold sweep
            # when the workload later drains back to B1.
            bucket.seed_initialized = self._seed_bucket_from_online_neighbor(
                key,
                bucket,
            )
        return bucket

    def _seed_bucket_from_online_neighbor(
        self,
        key: int,
        bucket: _BatchBucket,
    ) -> bool:
        candidates = [
            (candidate_key, candidate)
            for candidate_key, candidate in self._buckets.items()
            # Transfer only from a fuller production bucket into a draining
            # bucket. The reverse direction lets transient B8 startup data
            # seed B16 and can then create a circular B8 <-> B16 prior.
            if candidate_key > key
            and abs(math.log2(candidate_key) - math.log2(key)) <= 1
            and any(
                candidate.arms[gamma].effective_observations for gamma in candidate.launched_arms
            )
        ]
        if not candidates:
            return False
        source_key, source = min(
            candidates,
            key=lambda item: abs(math.log2(item[0]) - math.log2(key)),
        )
        # A transferred prior is useful for scoring, but it is not evidence
        # that the source bucket actually ran that arm.  Restrict another
        # transfer to launched arms so a chain of draining buckets cannot turn
        # an unmeasured gamma into a bootstrap target.
        observed = [
            gamma for gamma in source.launched_arms if source.arms[gamma].effective_observations
        ]
        if not observed:
            return False
        scale = key / source_key
        for gamma, arm in bucket.arms.items():
            source_gamma = min(
                observed,
                key=lambda candidate: (abs(candidate - gamma), -candidate),
            )
            arm.prior_reward = source.arms[source_gamma].mean_reward * scale
            arm.prior_weight = self.warmup_samples
        bucket.seeded_from_bucket = source_key
        bucket.bootstrap_target_gamma = self._preferred_measured_gamma(
            source,
            observed,
        )
        # Once an adjacent bucket has completed its sweep, reuse that sweep for
        # short-lived refill/drain buckets. They still collect local rewards and
        # can move under UCB, but do not synchronously replay every cold arm at
        # the tail of a benchmark or request burst.
        source_sweep_complete = source.warmup_return_completed and (
            all(gamma in source.launched_arms for gamma in source.arms)
            or source.frontier_deferred_at_observations is not None
        )
        if source_sweep_complete:
            bucket.launched_arms.update(bucket.arms)
            bucket.warmup_return_completed = True
        return True

    def _preferred_measured_gamma(
        self,
        bucket: _BatchBucket,
        candidates: list[int],
    ) -> int:
        """Prefer the range midpoint when measured rewards are near-tied."""
        best_reward = max(bucket.arms[gamma].mean_reward for gamma in candidates)
        deadband = abs(best_reward) * self.hysteresis * 0.5
        near_best = [
            gamma
            for gamma in candidates
            if best_reward - bucket.arms[gamma].mean_reward <= deadband
        ]
        return min(
            near_best,
            key=lambda gamma: (abs(gamma - self.anchor_gamma), -gamma),
        )

    def _continuation_acceptance_rate(self) -> float:
        previous_accepted = self._second_token_transition[1]
        transition_observations = sum(previous_accepted)
        if transition_observations:
            return previous_accepted[1] / transition_observations
        observed_positions = [
            self.position_acceptance_rates[position]
            for position, observations in enumerate(self.position_observations)
            if observations
        ]
        return observed_positions[-1] if observed_positions else self.acceptance_rate

    def _expected_useful_tokens(self, gamma: int) -> float:
        continuation_rate = self._continuation_acceptance_rate()
        survival = 1.0
        useful_tokens = 1.0
        for position in range(gamma):
            # A newly opened frontier can have only a handful of samples at
            # its last position.  Treating an accidental 100% there as a
            # stable estimate makes the model overvalue the next, even wider
            # arm.  Until a position has one full observation window, use the
            # request-level continuation rate learned from the established
            # gamma-2 path instead.
            acceptance_rate = (
                self.position_acceptance_rates[position]
                if self.position_observations[position] >= self.model_probe_min_observations
                else continuation_rate
            )
            survival *= min(max(acceptance_rate, 0.0), 1.0)
            useful_tokens += survival
        return useful_tokens

    def _modeled_reward(
        self,
        bucket: _BatchBucket,
        anchor_gamma: int,
        candidate_gamma: int,
    ) -> float:
        """Estimate an unmeasured serial arm without paying its cold-start cost."""
        anchor_reward = bucket.arms[anchor_gamma].mean_reward
        if anchor_reward <= 0:
            return 0.0
        anchor_tokens = self._expected_useful_tokens(anchor_gamma)
        candidate_tokens = self._expected_useful_tokens(candidate_gamma)
        # A serial speculative step has a shared Target component plus one
        # Draft unit per proposed token. The runtime supplies a method-specific
        # shared-cost estimate so a large Target does not make a one-token Draft
        # reduction look like a proportional end-to-end latency reduction.
        shared_cost_units = self.probe_shared_cost_units
        latency_scale = (candidate_gamma + shared_cost_units) / (anchor_gamma + shared_cost_units)
        return anchor_reward * (candidate_tokens / anchor_tokens) / latency_scale

    def candidate_is_feasible(self, gamma: int, batch_size: int) -> bool:
        if gamma not in self.candidate_gammas:
            return False
        batch_limit = self.candidate_batch_limits.get(gamma)
        if batch_limit is not None and batch_size > batch_limit:
            return False
        return self.max_batched_tokens is None or (
            batch_size * (gamma + 1) <= self.max_batched_tokens
        )

    def begin_step_feedback(self) -> None:
        """Start collecting committed tokens for one scheduler output."""
        self._step_useful_tokens = 0
        self._step_requests = 0

    def observe(
        self,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        *,
        request_id: str | None = None,
    ) -> None:
        if num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be nonnegative")
        if not 0 <= num_accepted_tokens <= num_draft_tokens:
            raise ValueError("accepted tokens must be in [0, num_draft_tokens]")
        if num_draft_tokens > self.max_gamma:
            raise ValueError("draft tokens exceed configured max gamma")

        if request_id is not None:
            previous = self._request_previous_acceptance.get(request_id)
            if previous is not None and previous[0] >= 2 and num_draft_tokens >= 2:
                previous_second = int(previous[1] >= 2)
                current_second = int(num_accepted_tokens >= 2)
                self._second_token_transition[previous_second][current_second] += 1
            self._request_previous_acceptance[request_id] = (
                num_draft_tokens,
                num_accepted_tokens,
            )

        self._step_requests += 1
        self._step_useful_tokens += num_accepted_tokens + 1
        if num_draft_tokens == 0:
            return

        self._pending_successes += num_accepted_tokens
        self._pending_trials += num_accepted_tokens + (num_accepted_tokens < num_draft_tokens)
        for position in range(num_accepted_tokens):
            self._pending_position_successes[position] += 1
            self._pending_position_trials[position] += 1
        if num_accepted_tokens < num_draft_tokens:
            self._pending_position_trials[num_accepted_tokens] += 1

    def complete_step(
        self,
        *,
        gamma: int,
        batch_size: int,
        context_tokens: int,
        latency_ms: float | None = None,
        completed_at_ms: float | None = None,
    ) -> None:
        """Commit one stable decode reward after verifier feedback is known."""
        del context_tokens
        if gamma not in self.candidate_gammas:
            raise ValueError("observed gamma is outside the candidate set")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if completed_at_ms is not None:
            if not math.isfinite(completed_at_ms):
                raise ValueError("completed_at_ms must be finite")
            previous_completion_ms = self._last_completion_ms
            self._last_completion_ms = completed_at_ms
            if previous_completion_ms is None:
                self._apply_pending_acceptance()
                self.begin_step_feedback()
                return
            latency_ms = completed_at_ms - previous_completion_ms
        if latency_ms is None or not math.isfinite(latency_ms) or latency_ms <= 0:
            raise ValueError("latency_ms must be finite and positive")

        self._apply_pending_acceptance()
        useful_tokens = self._step_useful_tokens if self._step_requests else batch_size
        if self._burn_in_remaining:
            self._burn_in_remaining -= 1
            self.begin_step_feedback()
            return
        arm = self._bucket(batch_size).arms[gamma]
        arm.record(useful_tokens, latency_ms)
        self.latency_observations[gamma] += 1
        self._latency_sums[gamma] += latency_ms
        self.begin_step_feedback()

    def discard_step_feedback(self, *, completed_at_ms: float | None = None) -> None:
        """Keep AC feedback but discard reward from a non-stable graph step."""
        if completed_at_ms is not None:
            if not math.isfinite(completed_at_ms):
                raise ValueError("completed_at_ms must be finite")
            self._last_completion_ms = completed_at_ms
        self._apply_pending_acceptance()
        self.begin_step_feedback()

    def predicted_goodput(
        self,
        gamma: int,
        batch_size: int,
        context_tokens: int,
    ) -> float:
        del context_tokens
        if gamma not in self.candidate_gammas:
            raise ValueError("gamma is outside the candidate set")
        bucket = self._bucket(batch_size)
        arm = bucket.arms[gamma]
        if arm.observations:
            return arm.mean_reward

        observed = [
            candidate.arms[gamma].mean_reward
            for candidate in self._buckets.values()
            if candidate.arms[gamma].observations
        ]
        return sum(observed) / len(observed) if observed else 0.0

    def latency_correction_factor(self, gamma: int) -> float:
        """Expose a trace-compatible relative latency indicator."""
        if gamma not in self.candidate_gammas:
            raise ValueError("gamma is outside the candidate set")
        count = self.latency_observations[gamma]
        if not count:
            return 1.0
        selected_mean = self._latency_sums[gamma] / count
        all_count = sum(self.latency_observations)
        all_mean = sum(self._latency_sums) / all_count
        return selected_mean / all_mean if all_mean else 1.0

    def force(
        self,
        gamma: int,
        batch_size: int,
        context_tokens: int,
        *,
        reason: str,
    ) -> ControllerDecision:
        if gamma not in self.candidate_gammas:
            raise ValueError("forced gamma is outside the candidate set")
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
        return self._decision(
            gamma,
            previous_gamma,
            selected_goodput,
            previous_goodput,
            reason,
        )

    def summary(self) -> dict[str, object]:
        buckets: dict[str, object] = {}
        for batch_size, bucket in sorted(self._buckets.items()):
            arms = {
                str(gamma): {
                    "observations": arm.observations,
                    "prior_weight": arm.prior_weight,
                    "mean_reward": round(arm.mean_reward, 6),
                }
                for gamma, arm in sorted(bucket.arms.items())
                if arm.observations
            }
            modeled_rewards = {
                str(gamma): round(
                    self._modeled_reward(bucket, self.anchor_gamma, gamma),
                    6,
                )
                for gamma in sorted(bucket.arms)
                if bucket.arms[self.anchor_gamma].mean_reward > 0
            }
            buckets[str(batch_size)] = {
                "selected_gamma": bucket.selected_gamma,
                "selected_at_observations": bucket.selected_at_observations,
                "scheduled_since_selection": (bucket.scheduled_since_selection),
                "launched_arms": sorted(bucket.launched_arms),
                "bootstrap_target_gamma": bucket.bootstrap_target_gamma,
                "seeded_from_bucket": bucket.seeded_from_bucket,
                "frontier_deferred_at_observations": (bucket.frontier_deferred_at_observations),
                "anchor_revalidation_at_observations": (bucket.anchor_revalidation_at_observations),
                "modeled_rewards": modeled_rewards,
                "arms": arms,
            }
        previous_rejected = self._second_token_transition[0]
        previous_accepted = self._second_token_transition[1]

        def conditional_rate(row: list[int]) -> float | None:
            total = sum(row)
            return round(row[1] / total, 6) if total else None

        return {
            "candidate_gammas": list(self.candidate_gammas),
            "candidate_batch_limits": dict(sorted(self.candidate_batch_limits.items())),
            "initial_gamma": self.initial_gamma,
            "warmup_return": self.warmup_return,
            "model_probe_min_observations": self.model_probe_min_observations,
            "deferred_exploration_observations": (self.deferred_exploration_observations),
            "selection_counts": {
                str(gamma): count for gamma, count in enumerate(self.selection_counts) if count
            },
            "gamma_switches": self.gamma_switches,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "burn_in_remaining": self._burn_in_remaining,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "position_acceptance_rates": [
                round(rate, 6) for rate in self.position_acceptance_rates
            ],
            "position_observations": self.position_observations.copy(),
            "second_token_transition": {
                "counts": [
                    previous_rejected.copy(),
                    previous_accepted.copy(),
                ],
                "accept_after_reject": conditional_rate(previous_rejected),
                "accept_after_accept": conditional_rate(previous_accepted),
            },
            "batch_buckets": buckets,
        }

    def choose(self, batch_size: int, context_tokens: int) -> ControllerDecision:
        if batch_size <= 0 or context_tokens < 0:
            raise ValueError("batch_size must be positive and context_tokens nonnegative")
        self._apply_pending_acceptance()
        feasible = [
            gamma
            for gamma in self.candidate_gammas
            if self.candidate_is_feasible(gamma, batch_size)
        ]
        if not feasible:
            raise RuntimeError(
                "no online gamma fits max_num_batched_tokens: "
                f"batch={batch_size}, min_gamma={self.min_gamma}, "
                f"budget={self.max_batched_tokens}"
            )

        previous_gamma = self.current_gamma
        if previous_gamma not in feasible:
            return self.force(
                max(feasible),
                batch_size,
                context_tokens,
                reason="batch_token_budget",
            )

        bucket_key = self.batch_bucket(batch_size)
        bucket = self._bucket(batch_size)
        safe_anchor = min(
            feasible,
            key=lambda gamma: (abs(gamma - self.anchor_gamma), -gamma),
        )
        bucket_changed = (
            self._active_bucket_key is not None and self._active_bucket_key != bucket_key
        )
        self._active_bucket_key = bucket_key
        if bucket_changed and self.initial_gamma is not None:
            measured = [
                gamma
                for gamma in feasible
                if gamma in bucket.launched_arms
                and bucket.arms[gamma].effective_observations >= self.warmup_samples
            ]
            resume_target = (
                bucket.bootstrap_target_gamma
                if bucket.bootstrap_target_gamma in feasible
                else self._preferred_measured_gamma(bucket, measured)
                if measured
                else safe_anchor
            )
            if previous_gamma != resume_target:
                selected = self._step_toward(
                    previous_gamma,
                    resume_target,
                    feasible,
                )
                self._activate_bucket_gamma(bucket, selected)
                self.current_gamma = selected
                bucket.bootstrap_target_gamma = None if selected == resume_target else resume_target
                return self._decision(
                    selected,
                    previous_gamma,
                    bucket.arms[selected].mean_reward,
                    bucket.arms[previous_gamma].mean_reward,
                    "online_ucb_bucket_resume",
                )
        non_anchor_launched = bucket.launched_arms.difference({safe_anchor})
        if (
            self.deferred_exploration_observations
            and not non_anchor_launched
            and bucket.arms[safe_anchor].observations < self.deferred_exploration_observations
        ):
            selected = self._step_toward(
                previous_gamma,
                safe_anchor,
                feasible,
            )
            self._activate_bucket_gamma(bucket, selected)
            bucket.launched_arms.add(selected)
            self.current_gamma = selected
            return self._decision(
                selected,
                previous_gamma,
                bucket.arms[selected].mean_reward,
                bucket.arms[previous_gamma].mean_reward,
                "online_ucb_safe_burn_in",
            )
        bootstrap_target = bucket.bootstrap_target_gamma
        if bootstrap_target is not None:
            feasible_target = min(
                feasible,
                key=lambda gamma: (abs(gamma - bootstrap_target), -gamma),
            )
            if previous_gamma != feasible_target:
                selected = self._step_toward(
                    previous_gamma,
                    feasible_target,
                    feasible,
                )
                self._activate_bucket_gamma(bucket, selected)
                self.current_gamma = selected
                if selected == feasible_target:
                    bucket.bootstrap_target_gamma = None
                return self._decision(
                    selected,
                    previous_gamma,
                    bucket.arms[selected].mean_reward,
                    bucket.arms[previous_gamma].mean_reward,
                    "online_ucb_bucket_transfer",
                )
            bucket.bootstrap_target_gamma = None

        # Keep moving toward the arm selected for the cold-start sweep.  With
        # max_gamma_step=1 a distant arm can require several decisions to
        # reach.  Intermediate arms may already be launched; applying their
        # normal control window here needlessly turns a bounded transition
        # into several extra probe steps.
        warmup_target = bucket.warmup_target_gamma
        if warmup_target is not None:
            feasible_target = min(
                feasible,
                key=lambda gamma: (abs(gamma - warmup_target), -gamma),
            )
            if previous_gamma != feasible_target:
                selected = self._step_toward(
                    previous_gamma,
                    feasible_target,
                    feasible,
                )
                self._activate_bucket_gamma(bucket, selected)
                self.current_gamma = selected
                if selected == feasible_target:
                    bucket.warmup_target_gamma = None
                return self._decision(
                    selected,
                    previous_gamma,
                    bucket.arms[selected].mean_reward,
                    bucket.arms[previous_gamma].mean_reward,
                    "online_ucb_warmup_step",
                )
            bucket.warmup_target_gamma = None
        if bucket.selected_gamma != previous_gamma:
            bucket.selected_gamma = previous_gamma
            bucket.selected_at_observations = bucket.arms[previous_gamma].observations
            bucket.scheduled_since_selection = 0
        previous_goodput = bucket.arms[previous_gamma].mean_reward
        required_window = self.control_interval
        if previous_gamma not in bucket.launched_arms:
            required_window = self.warmup_samples + self.inflight_warmup_padding
        if bucket.scheduled_since_selection < required_window:
            bucket.scheduled_since_selection += 1
            reason = (
                "online_ucb_warmup_window"
                if previous_gamma not in bucket.launched_arms
                else "online_ucb_window"
            )
            return self._decision(
                previous_gamma,
                previous_gamma,
                previous_goodput,
                previous_goodput,
                reason,
            )

        bucket.launched_arms.add(previous_gamma)
        if self.initial_gamma is not None:
            feedback_anchor = min(
                feasible,
                key=lambda gamma: (abs(gamma - self.anchor_gamma), -gamma),
            )
            if bucket.arms[feedback_anchor].effective_observations < self.warmup_samples:
                # Scheduling an arm is not evidence that it is good. Async
                # feedback can arrive after a neighboring probe, which used to
                # make that probe the only measured arm and incorrectly open a
                # farther frontier. Stay at the cold-start anchor until its
                # first stable reward is available; the full candidate range
                # remains eligible immediately afterward.
                selected = self._step_toward(
                    previous_gamma,
                    feedback_anchor,
                    feasible,
                )
                if selected != previous_gamma:
                    self._activate_bucket_gamma(bucket, selected)
                    self.current_gamma = selected
                else:
                    bucket.scheduled_since_selection += 1
                return self._decision(
                    selected,
                    previous_gamma,
                    bucket.arms[selected].mean_reward,
                    previous_goodput,
                    "online_ucb_anchor_feedback_wait",
                )
            pending_probe_feedback = [
                gamma
                for gamma in bucket.launched_arms
                if bucket.arms[gamma].effective_observations < self.warmup_samples
            ]
            measured_for_probe = [
                gamma
                for gamma in feasible
                if bucket.arms[gamma].effective_observations >= self.warmup_samples
            ]
            if measured_for_probe:
                best_measured_reward = max(
                    bucket.arms[gamma].mean_reward for gamma in measured_for_probe
                )
                pending_probe_feedback = [
                    gamma
                    for gamma in pending_probe_feedback
                    if not (
                        bucket.arms[gamma].observations > 0
                        and bucket.arms[gamma].mean_reward
                        < best_measured_reward * (1 - self.hysteresis)
                    )
                ]
            if pending_probe_feedback:
                feedback_target = (
                    self._preferred_measured_gamma(bucket, measured_for_probe)
                    if measured_for_probe
                    else feedback_anchor
                )
                selected = self._step_toward(
                    previous_gamma,
                    feedback_target,
                    feasible,
                )
                if selected != previous_gamma:
                    self._activate_bucket_gamma(bucket, selected)
                    self.current_gamma = selected
                else:
                    bucket.scheduled_since_selection += 1
                return self._decision(
                    selected,
                    previous_gamma,
                    bucket.arms[selected].mean_reward,
                    previous_goodput,
                    "online_ucb_probe_feedback_wait",
                )
        unlaunched = [gamma for gamma in feasible if gamma not in bucket.launched_arms]
        if unlaunched and self.initial_gamma is not None:
            measured = [
                gamma
                for gamma in feasible
                if gamma in bucket.launched_arms
                and bucket.arms[gamma].effective_observations >= self.warmup_samples
            ]
            if measured:
                measured_best = self._preferred_measured_gamma(bucket, measured)
                revalidation_start = bucket.anchor_revalidation_at_observations
                if revalidation_start is not None:
                    if previous_gamma != feedback_anchor:
                        selected = self._step_toward(
                            previous_gamma,
                            feedback_anchor,
                            feasible,
                        )
                        self._activate_bucket_gamma(bucket, selected)
                        self.current_gamma = selected
                        return self._decision(
                            selected,
                            previous_gamma,
                            bucket.arms[selected].mean_reward,
                            previous_goodput,
                            "online_ucb_anchor_revalidation_return",
                        )
                    fresh_anchor_observations = (
                        bucket.arms[feedback_anchor].observations - revalidation_start
                    )
                    if fresh_anchor_observations < self.window_size:
                        bucket.scheduled_since_selection += 1
                        return self._decision(
                            previous_gamma,
                            previous_gamma,
                            previous_goodput,
                            previous_goodput,
                            "online_ucb_anchor_revalidation_window",
                        )
                    bucket.anchor_revalidation_at_observations = None
                previous_ready = (
                    bucket.arms[previous_gamma].effective_observations >= self.warmup_samples
                )
                if previous_ready and previous_gamma != measured_best:
                    selected = self._step_toward(
                        previous_gamma,
                        measured_best,
                        feasible,
                    )
                    self._activate_bucket_gamma(bucket, selected)
                    # This is part of the bounded warmup traversal, not a new
                    # steady-state residence window. Let the next decision
                    # compare the other neighbor immediately after returning.
                    bucket.scheduled_since_selection = self.control_interval
                    self.current_gamma = selected
                    return self._decision(
                        selected,
                        previous_gamma,
                        bucket.arms[selected].mean_reward,
                        previous_goodput,
                        "online_ucb_frontier_return",
                    )

                anchor_observations = bucket.arms[feedback_anchor].observations
                supported: list[int] = []
                if anchor_observations >= self.model_probe_min_observations:
                    best_reward = bucket.arms[measured_best].mean_reward
                    required_probe_gain = max(self.hysteresis, self.exploration)
                    for target in unlaunched:
                        target_index = self.candidate_gammas.index(target)
                        boundary_index = target_index + (-1 if target > feedback_anchor else 1)
                        if not 0 <= boundary_index < len(self.candidate_gammas):
                            continue
                        boundary = self.candidate_gammas[boundary_index]
                        if boundary not in measured:
                            continue
                        boundary_reward = bucket.arms[boundary].mean_reward
                        if boundary_reward < best_reward * (1 - self.hysteresis):
                            continue
                        modeled_reward = self._modeled_reward(
                            bucket,
                            boundary,
                            target,
                        )
                        if modeled_reward > best_reward * (1 + required_probe_gain):
                            supported.append(target)
                if supported:
                    # A farther arm becomes eligible only through a measured
                    # adjacent boundary, so max_gamma_step=1 is also respected
                    # by the evidence chain rather than only by the transition.
                    unlaunched = supported
                    bucket.frontier_deferred_at_observations = None
                else:
                    # Unmeasured arms remain candidates. Re-evaluate their
                    # modeled goodput after every control window as AC changes,
                    # but do not pay eager compilation cost for an arm whose
                    # optimistic serial estimate already loses to the incumbent.
                    bucket.frontier_deferred_at_observations = anchor_observations
                    if previous_gamma != feedback_anchor:
                        bucket.anchor_revalidation_at_observations = anchor_observations
                        selected = self._step_toward(
                            previous_gamma,
                            feedback_anchor,
                            feasible,
                        )
                        self._activate_bucket_gamma(bucket, selected)
                        self.current_gamma = selected
                        return self._decision(
                            selected,
                            previous_gamma,
                            bucket.arms[selected].mean_reward,
                            previous_goodput,
                            "online_ucb_anchor_revalidation_return",
                        )
                    bucket.warmup_return_completed = True
                    self._activate_bucket_gamma(bucket, previous_gamma)
                    return self._decision(
                        previous_gamma,
                        previous_gamma,
                        previous_goodput,
                        previous_goodput,
                        (
                            "online_ucb_model_hold"
                            if anchor_observations >= self.model_probe_min_observations
                            else "online_ucb_model_warmup"
                        ),
                    )
        if unlaunched:
            # Explore every arm once in descending, adjacent order. The arm is
            # considered launched from scheduled work rather than delayed
            # async feedback, which bounds the number of in-flight low-gamma
            # steps.
            target = min(
                unlaunched,
                key=lambda gamma: (abs(gamma - previous_gamma), -gamma),
            )
            selected = self._step_toward(previous_gamma, target, feasible)
            if selected != target:
                bucket.warmup_target_gamma = target
            self._activate_bucket_gamma(bucket, selected)
            self.current_gamma = selected
            return self._decision(
                selected,
                previous_gamma,
                bucket.arms[selected].mean_reward,
                previous_goodput,
                "online_ucb_warmup",
            )

        awaiting_feedback = [
            gamma
            for gamma in feasible
            if gamma in bucket.launched_arms
            if bucket.arms[gamma].effective_observations < self.warmup_samples
        ]
        if awaiting_feedback:
            # The exploration windows are already in flight. Wait on a middle
            # arm so delayed async feedback cannot flood the queue with the
            # final (smallest) warmup arm.
            midpoint = round((min(feasible) + max(feasible)) / 2)
            fallback = min(
                feasible,
                key=lambda gamma: (abs(gamma - midpoint), -gamma),
            )
            selected = self._step_toward(previous_gamma, fallback, feasible)
            if selected != previous_gamma:
                self._activate_bucket_gamma(bucket, selected)
                self.current_gamma = selected
            else:
                bucket.scheduled_since_selection += 1
            return self._decision(
                selected,
                previous_gamma,
                bucket.arms[selected].mean_reward,
                previous_goodput,
                "online_ucb_feedback_wait",
            )

        if not bucket.warmup_return_completed:
            # A descending initial sweep ends on the smallest arm. Return to
            # the best arm measured by the completed sweep so the final probe
            # does not become the residence state merely because its sample
            # arrived last. Prefer the larger gamma on a reward tie and keep
            # the transition gradual for wider gamma ranges.
            configured_incumbent = min(
                feasible,
                key=lambda gamma: (abs(gamma - self.anchor_gamma), -gamma),
            )
            measured = [gamma for gamma in feasible if bucket.arms[gamma].effective_observations]
            measured_best = self._preferred_measured_gamma(bucket, measured)
            incumbent = configured_incumbent if self.warmup_return == "incumbent" else measured_best
            if self.warmup_return == "best" and measured_best != configured_incumbent:
                configured_reward = bucket.arms[configured_incumbent].mean_reward
                warmup_deadband = abs(configured_reward) * self.hysteresis * 0.5
                measured_gain = bucket.arms[measured_best].mean_reward - configured_reward
                if measured_gain < warmup_deadband:
                    incumbent = configured_incumbent
            if previous_gamma != incumbent:
                selected = self._step_toward(
                    previous_gamma,
                    incumbent,
                    feasible,
                )
                self._activate_bucket_gamma(bucket, selected)
                self.current_gamma = selected
                bucket.warmup_return_completed = selected == incumbent
                return self._decision(
                    selected,
                    previous_gamma,
                    bucket.arms[selected].mean_reward,
                    previous_goodput,
                    "online_ucb_warmup_return",
                )
            bucket.warmup_return_completed = True

        observations_since_selection = (
            bucket.arms[previous_gamma].observations - bucket.selected_at_observations
        )
        if observations_since_selection <= 0:
            bucket.scheduled_since_selection += 1
            return self._decision(
                previous_gamma,
                previous_gamma,
                previous_goodput,
                previous_goodput,
                "online_ucb_feedback_wait",
            )

        scored_feasible = [gamma for gamma in feasible if bucket.arms[gamma].effective_observations]
        total_samples = sum(bucket.arms[gamma].effective_observations for gamma in scored_feasible)
        reward_scale = max(bucket.arms[gamma].mean_reward for gamma in scored_feasible)

        def score(gamma: int) -> float:
            arm = bucket.arms[gamma]
            bonus = 0.0
            if self.exploration and reward_scale:
                bonus = (
                    self.exploration
                    * reward_scale
                    * math.sqrt(2.0 * math.log(total_samples + 1) / arm.effective_observations)
                )
            return arm.mean_reward + bonus

        best_gamma = max(scored_feasible, key=lambda gamma: (score(gamma), gamma))
        best_score = score(best_gamma)
        previous_score = score(previous_gamma)
        required_gain = abs(previous_score) * self.hysteresis
        if best_gamma == previous_gamma or best_score - previous_score < required_gain:
            bucket.selected_gamma = previous_gamma
            bucket.selected_at_observations = bucket.arms[previous_gamma].observations
            bucket.scheduled_since_selection = 1
            return self._decision(
                previous_gamma,
                previous_gamma,
                previous_goodput,
                previous_goodput,
                "online_ucb_hysteresis",
            )

        selected = self._step_toward(previous_gamma, best_gamma, feasible)
        selected_goodput = bucket.arms[selected].mean_reward
        self._activate_bucket_gamma(bucket, selected)
        self.current_gamma = selected
        return self._decision(
            selected,
            previous_gamma,
            selected_goodput,
            previous_goodput,
            "online_ucb" if selected == best_gamma else "online_ucb_step",
        )

    @staticmethod
    def _activate_bucket_gamma(bucket: _BatchBucket, gamma: int) -> None:
        bucket.selected_gamma = gamma
        bucket.selected_at_observations = bucket.arms[gamma].observations
        # The decision being returned schedules the first step for this arm.
        bucket.scheduled_since_selection = 1

    def _step_toward(
        self,
        current: int,
        target: int,
        feasible: list[int],
    ) -> int:
        lower = max(min(feasible), current - self.max_gamma_step)
        upper = min(max(feasible), current + self.max_gamma_step)
        bounded = min(max(target, lower), upper)
        return min(feasible, key=lambda gamma: (abs(gamma - bounded), -gamma))

    def _decision(
        self,
        gamma: int,
        previous_gamma: int,
        predicted_goodput: float,
        previous_goodput: float,
        reason: str,
    ) -> ControllerDecision:
        self.selection_counts[gamma] += 1
        self.gamma_switches += gamma != previous_gamma
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        return ControllerDecision(
            gamma=gamma,
            previous_gamma=previous_gamma,
            acceptance_rate=self.acceptance_rate,
            predicted_goodput=predicted_goodput,
            previous_goodput=previous_goodput,
            changed=gamma != previous_gamma,
            reason=reason,
        )

    def _apply_pending_acceptance(self) -> None:
        if self._pending_trials == 0:
            return
        observed_rate = self._pending_successes / self._pending_trials
        if self.total_observations:
            self.acceptance_rate = (
                self.ewma_weight * observed_rate + (1 - self.ewma_weight) * self.acceptance_rate
            )
        else:
            self.acceptance_rate = observed_rate
        for position, trials in enumerate(self._pending_position_trials):
            if not trials:
                continue
            had_observations = self.position_observations[position] > 0
            self.position_observations[position] += trials
            observed_rate = self._pending_position_successes[position] / trials
            if had_observations:
                self.position_acceptance_rates[position] = (
                    self.ewma_weight * observed_rate
                    + (1 - self.ewma_weight) * self.position_acceptance_rates[position]
                )
            else:
                self.position_acceptance_rates[position] = observed_rate
        self.total_observations += self._pending_trials
        self._pending_successes = 0
        self._pending_trials = 0
        self._pending_position_successes = [0] * self.max_gamma
        self._pending_position_trials = [0] * self.max_gamma
