"""Online admission control for continuously batched speculative decoding."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _ActionReward:
    useful_tokens: int = 0
    latency_ms: float = 0.0
    observations: int = 0

    @property
    def mean_goodput(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return self.useful_tokens / self.latency_ms

    def observe(self, useful_tokens: int, latency_ms: float) -> None:
        self.useful_tokens += useful_tokens
        self.latency_ms += latency_ms
        self.observations += 1


@dataclass(frozen=True)
class RefillDecision:
    admit: bool
    reason: str
    admit_score: float = 0.0
    hold_score: float = 0.0


class OnlineRefillController:
    """Choose admission timing from measured serving goodput.

    Each exact ``(running requests, gamma)`` state has two actions: execute
    another decode step at the current width, or admit waiting requests into
    free slots. Rewards use completion cadence, which remains meaningful with
    asynchronous scheduling.
    """

    def __init__(
        self,
        *,
        exploration: float = 0.15,
        future_discount: float = 1.0,
        hysteresis: float = 0.03,
        max_consecutive_holds: int = 8,
        cohort_gated: bool = False,
    ) -> None:
        if exploration < 0:
            raise ValueError("refill exploration must be non-negative")
        if future_discount < 0:
            raise ValueError("refill future discount must be non-negative")
        if hysteresis < 0:
            raise ValueError("refill hysteresis must be non-negative")
        if max_consecutive_holds < 0:
            raise ValueError("refill max consecutive holds must be non-negative")
        self.exploration = exploration
        self.future_discount = future_discount
        self.hysteresis = hysteresis
        self.max_consecutive_holds = max_consecutive_holds
        self.cohort_gated = cohort_gated
        self._rewards: dict[tuple[int, int], dict[str, _ActionReward]] = {}
        self._consecutive_holds: dict[tuple[int, int], int] = {}
        self.reason_counts: dict[str, int] = {}

    def _state(self, running_count: int, gamma: int) -> dict[str, _ActionReward]:
        key = (running_count, gamma)
        if key not in self._rewards:
            self._rewards[key] = {
                "hold": _ActionReward(),
                "admit": _ActionReward(),
            }
        return self._rewards[key]

    def _decision(
        self,
        admit: bool,
        reason: str,
        *,
        admit_score: float = 0.0,
        hold_score: float = 0.0,
    ) -> RefillDecision:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        return RefillDecision(admit, reason, admit_score, hold_score)

    def choose(
        self,
        *,
        running_count: int,
        target_count: int,
        waiting_count: int,
        gamma: int,
        current_goodput: float,
        target_goodput: float,
    ) -> RefillDecision:
        if waiting_count <= 0 or target_count <= running_count:
            return self._decision(True, "no_refill_choice")
        if running_count == 0:
            return self._decision(True, "empty_engine")
        if self.cohort_gated:
            return self._decision(False, "cohort_gate")

        key = (running_count, gamma)
        actions = self._state(running_count, gamma)
        holds = self._consecutive_holds.get(key, 0)
        if self.max_consecutive_holds and holds >= self.max_consecutive_holds:
            self._consecutive_holds[key] = 0
            return self._decision(True, "starvation_guard")

        hold = actions["hold"]
        admit = actions["admit"]
        if hold.observations == 0:
            self._consecutive_holds[key] = holds + 1
            return self._decision(False, "measure_hold")

        # Admission pays a one-step prefill/scheduler cost. Amortize that setup
        # over the requests that can be admitted together: a one-slot refill
        # gets little lookahead, while accumulated free slots make a larger
        # gated refill increasingly worthwhile. This is driven by queue state,
        # not a fixed running-request threshold.
        admitted_requests = max(1, target_count - running_count)
        future_steps = self.future_discount * admitted_requests
        target_hold = self._state(target_count, gamma)["hold"]
        current_value = hold.mean_goodput if hold.observations else current_goodput
        target_value = target_hold.mean_goodput if target_hold.observations else target_goodput
        if admit.observations == 0:
            # Use the current stable step as a conservative immediate-reward
            # prior. Admit only when the measured/predicted destination state
            # can amortize the transition over a speculative window.
            hold_score = hold.mean_goodput + future_steps * current_value
            admit_score = current_value + future_steps * target_value
            if admit_score > hold_score * (1 + self.hysteresis):
                self._consecutive_holds[key] = 0
                return self._decision(
                    True,
                    "optimistic_target_gain",
                    admit_score=admit_score,
                    hold_score=hold_score,
                )
            self._consecutive_holds[key] = holds + 1
            return self._decision(
                False,
                "defer_unmeasured_admit",
                admit_score=admit_score,
                hold_score=hold_score,
            )

        total_observations = hold.observations + admit.observations
        reward_scale = max(
            hold.mean_goodput,
            admit.mean_goodput,
            current_goodput,
            target_goodput,
            1e-9,
        )

        def ucb(reward: _ActionReward) -> float:
            bonus = (
                self.exploration
                * reward_scale
                * math.sqrt(math.log(total_observations + 1) / reward.observations)
            )
            return reward.mean_goodput + bonus

        # Bellman lookahead over exact serving states. The gamma controller
        # intentionally pools nearby batch sizes, so its prediction cannot
        # distinguish (for example) B14 from B16. Prefer the refill
        # controller's measured stable-decode reward for an exact state and
        # use the pooled estimate only until that state has been observed.
        hold_score = ucb(hold) + future_steps * current_value
        admit_score = ucb(admit) + future_steps * target_value
        should_admit = admit_score > hold_score * (1 + self.hysteresis)
        self._consecutive_holds[key] = 0 if should_admit else holds + 1
        return self._decision(
            should_admit,
            "goodput_admit" if should_admit else "goodput_hold",
            admit_score=admit_score,
            hold_score=hold_score,
        )

    def observe_step(
        self,
        *,
        running_count: int,
        gamma: int,
        action: str,
        useful_tokens: int,
        latency_ms: float,
    ) -> None:
        if action not in {"hold", "admit"}:
            raise ValueError(f"invalid refill action: {action}")
        if useful_tokens < 0:
            raise ValueError("useful tokens must be non-negative")
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            return
        self._state(running_count, gamma)[action].observe(
            useful_tokens,
            latency_ms,
        )

    def summary(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for (running_count, gamma), actions in sorted(self._rewards.items()):
            states[f"b{running_count}:g{gamma}"] = {
                name: {
                    "observations": reward.observations,
                    "goodput": round(reward.mean_goodput, 6),
                }
                for name, reward in actions.items()
                if reward.observations
            }
        return {
            "cohort_gated": self.cohort_gated,
            "states": states,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


def _predicted_goodput(scheduler: Any, gamma: int, batch_size: int) -> float:
    controller = getattr(scheduler, "_vspec_adaptive_controller", None)
    if controller is None or batch_size <= 0:
        return 0.0
    try:
        return float(controller.predicted_goodput(gamma, batch_size, 0))
    except (RuntimeError, ValueError):
        return 0.0


def install_online_refill_patch(
    *,
    method: str,
    enabled_env: str,
    patch_marker: str,
) -> bool:
    """Install goodput-driven admission control for one speculative backend."""
    if os.getenv(enabled_env, "0") != "1":
        return False

    from vllm.v1.core.sched.scheduler import Scheduler

    original_schedule = Scheduler.schedule
    if getattr(original_schedule, patch_marker, False):
        return False
    original_update_from_output = Scheduler.update_from_output

    exploration = float(os.getenv("HUST_VSPEC_REFILL_EXPLORATION", "0.15"))
    future_discount = float(os.getenv("HUST_VSPEC_REFILL_FUTURE_DISCOUNT", "1.0"))
    hysteresis = float(os.getenv("HUST_VSPEC_REFILL_HYSTERESIS", "0.03"))
    max_holds = int(os.getenv("HUST_VSPEC_REFILL_MAX_HOLDS", "8"))
    cohort_gated = os.getenv("HUST_VSPEC_REFILL_COHORT_GATED", "0") == "1"
    trace = os.getenv("HUST_VSPEC_REFILL_TRACE", "0") == "1"

    def get_controller(scheduler: Any) -> OnlineRefillController:
        controller = getattr(scheduler, "_vspec_online_refill_controller", None)
        if controller is None:
            controller = OnlineRefillController(
                exploration=exploration,
                future_discount=future_discount,
                hysteresis=hysteresis,
                max_consecutive_holds=max_holds,
                cohort_gated=cohort_gated,
            )
            scheduler._vspec_online_refill_controller = controller
        return controller

    @wraps(original_schedule)
    def schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        speculative_config = getattr(self.vllm_config, "speculative_config", None)
        if getattr(speculative_config, "method", None) != method:
            return original_schedule(self, *args, **kwargs)

        running_count = len(self.running) + self.num_waiting_for_streaming_input
        waiting_count = len(self.waiting) + len(self.skipped_waiting)
        capacity = int(self.max_num_running_reqs)
        target_count = min(capacity, running_count + waiting_count)
        gamma_controller = getattr(self, "_vspec_adaptive_controller", None)
        gamma = int(getattr(gamma_controller, "current_gamma", self.num_spec_tokens))
        controller = get_controller(self)
        decision = controller.choose(
            running_count=running_count,
            target_count=target_count,
            waiting_count=waiting_count,
            gamma=gamma,
            current_goodput=_predicted_goodput(self, gamma, running_count),
            target_goodput=_predicted_goodput(self, gamma, target_count),
        )

        original_limit = self.max_num_running_reqs
        if not decision.admit:
            self.max_num_running_reqs = running_count
        try:
            scheduler_output = original_schedule(self, *args, **kwargs)
        finally:
            self.max_num_running_reqs = original_limit

        if scheduler_output.num_scheduled_tokens:
            admitted = bool(scheduler_output.scheduled_new_reqs)
            action = "admit" if decision.admit and admitted else "hold"
            scheduler_output.vspec_refill_feedback = (
                running_count,
                gamma,
                action,
                time.perf_counter() * 1000.0,
            )
            if trace:
                logger.warning(
                    "vSpec refill decision: method=%s running=%d target=%d "
                    "waiting=%d gamma=%d action=%s reason=%s "
                    "admit_score=%.6f hold_score=%.6f",
                    method,
                    running_count,
                    target_count,
                    waiting_count,
                    gamma,
                    action,
                    decision.reason,
                    decision.admit_score,
                    decision.hold_score,
                )
        return scheduler_output

    @wraps(original_update_from_output)
    def update_from_output(
        self: Any,
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> Any:
        result = original_update_from_output(
            self,
            scheduler_output,
            model_runner_output,
        )
        feedback = getattr(scheduler_output, "vspec_refill_feedback", None)
        if feedback is not None:
            running_count, gamma, action, scheduled_at_ms = feedback
            useful_tokens = sum(
                len(token_ids) for token_ids in model_runner_output.sampled_token_ids
            )
            get_controller(self).observe_step(
                running_count=int(running_count),
                gamma=int(gamma),
                action=str(action),
                useful_tokens=useful_tokens,
                latency_ms=time.perf_counter() * 1000.0 - scheduled_at_ms,
            )
        if trace and not self.requests and not getattr(self, "_vspec_refill_summary_logged", False):
            logger.warning("vSpec refill summary: %s", get_controller(self).summary())
            self._vspec_refill_summary_logged = True
        elif self.requests:
            self._vspec_refill_summary_logged = False
        return result

    setattr(schedule, patch_marker, True)
    Scheduler.schedule = schedule
    Scheduler.update_from_output = update_from_output
    logger.info(
        "vSpec enabled online refill control for %s "
        "(exploration=%.3f discount=%.3f hysteresis=%.3f "
        "max_holds=%d cohort_gated=%s)",
        method,
        exploration,
        future_discount,
        hysteresis,
        max_holds,
        cohort_gated,
    )
    return True
