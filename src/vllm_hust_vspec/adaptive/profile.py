"""Latency profile used by the vSpec Adaptive goodput controller."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _finite_float(value: Any, name: str, minimum: float = 0.0) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value}")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer, got {value}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return parsed


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class ForwardLatencyModel:
    """Linear forward latency model from context and batched token counts."""

    alpha_ms_per_context_token: float
    gamma_ms_per_batched_token: float
    delta_ms: float

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        name: str,
    ) -> ForwardLatencyModel:
        allowed_keys = {
            "alpha_ms_per_context_token",
            "gamma_ms_per_batched_token",
            "delta_ms",
        }
        unknown_keys = sorted(set(values) - allowed_keys)
        if unknown_keys:
            raise ValueError(f"unknown {name} field(s): {', '.join(unknown_keys)}")
        model = cls(
            alpha_ms_per_context_token=_finite_float(
                values.get("alpha_ms_per_context_token", 0.0),
                f"{name}.alpha_ms_per_context_token",
            ),
            gamma_ms_per_batched_token=_finite_float(
                values.get("gamma_ms_per_batched_token", 0.0),
                f"{name}.gamma_ms_per_batched_token",
            ),
            delta_ms=_finite_float(
                values.get("delta_ms", 0.0),
                f"{name}.delta_ms",
            ),
        )
        if model.predict(1, 1) <= 0:
            raise ValueError(f"{name} must predict a positive forward latency")
        return model

    def predict(self, context_tokens: int, batched_tokens: int) -> float:
        if context_tokens < 0 or batched_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        return (
            self.alpha_ms_per_context_token * context_tokens
            + self.gamma_ms_per_batched_token * batched_tokens
            + self.delta_ms
        )


@dataclass(frozen=True)
class AdaptiveProfile:
    """Validated offline profile for choosing a batch-uniform Draft length."""

    schema_version: int
    max_speculative_tokens: int
    default_acceptance_rate: float
    initial_gamma: int
    target: ForwardLatencyModel
    draft: ForwardLatencyModel
    draft_parallel: bool = False
    per_gamma_overhead_ms: tuple[tuple[int, float], ...] = ()
    batch_gamma_policy: tuple[tuple[int, int], ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> AdaptiveProfile:
        profile_path = Path(path).expanduser()
        try:
            document = json.loads(profile_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"cannot read adaptive profile {profile_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid adaptive profile JSON {profile_path}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError("adaptive profile root must be a JSON object")
        return cls.from_mapping(document)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AdaptiveProfile:
        allowed_keys = {
            "schema_version",
            "max_speculative_tokens",
            "default_acceptance_rate",
            "initial_gamma",
            "target",
            "draft",
            "draft_parallel",
            "per_gamma_overhead_ms",
            "batch_gamma_policy",
        }
        unknown_keys = sorted(set(values) - allowed_keys)
        if unknown_keys:
            raise ValueError(f"unknown adaptive profile field(s): {', '.join(unknown_keys)}")
        try:
            schema_version = int(values.get("schema_version", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("schema_version must be an integer") from exc
        if schema_version != 1:
            raise ValueError(f"unsupported adaptive profile schema_version={schema_version}")
        max_gamma = _positive_int(
            values.get("max_speculative_tokens"),
            "max_speculative_tokens",
        )
        acceptance_rate = _finite_float(
            values.get("default_acceptance_rate", 0.7),
            "default_acceptance_rate",
        )
        if acceptance_rate > 1:
            raise ValueError("default_acceptance_rate must be <= 1")
        initial_gamma = _nonnegative_int(
            values.get("initial_gamma", max_gamma),
            "initial_gamma",
        )
        if initial_gamma > max_gamma:
            raise ValueError("initial_gamma cannot exceed max_speculative_tokens")

        target_values = values.get("target")
        draft_values = values.get("draft")
        if not isinstance(target_values, dict) or not isinstance(draft_values, dict):
            raise ValueError("adaptive profile requires target and draft objects")

        raw_overhead = values.get("per_gamma_overhead_ms", {})
        if not isinstance(raw_overhead, dict):
            raise ValueError("per_gamma_overhead_ms must be a JSON object")
        overhead: list[tuple[int, float]] = []
        for raw_gamma, raw_latency in raw_overhead.items():
            gamma = int(raw_gamma)
            if gamma < 0 or gamma > max_gamma:
                raise ValueError(
                    "per_gamma_overhead_ms keys must be between 0 and max_speculative_tokens"
                )
            overhead.append(
                (
                    gamma,
                    _finite_float(
                        raw_latency,
                        f"per_gamma_overhead_ms[{gamma}]",
                    ),
                )
            )

        raw_policy = values.get("batch_gamma_policy", {})
        if not isinstance(raw_policy, dict):
            raise ValueError("batch_gamma_policy must be a JSON object")
        policy: list[tuple[int, int]] = []
        for raw_batch_size, raw_gamma in raw_policy.items():
            batch_size = _positive_int(
                raw_batch_size,
                "batch_gamma_policy batch size",
            )
            gamma = _nonnegative_int(
                raw_gamma,
                f"batch_gamma_policy[{batch_size}]",
            )
            if gamma > max_gamma:
                raise ValueError("batch_gamma_policy values cannot exceed max_speculative_tokens")
            policy.append((batch_size, gamma))

        return cls(
            schema_version=schema_version,
            max_speculative_tokens=max_gamma,
            default_acceptance_rate=acceptance_rate,
            initial_gamma=initial_gamma,
            target=ForwardLatencyModel.from_mapping(target_values, "target"),
            draft=ForwardLatencyModel.from_mapping(draft_values, "draft"),
            draft_parallel=_boolean(
                values.get("draft_parallel", False),
                "draft_parallel",
            ),
            per_gamma_overhead_ms=tuple(sorted(overhead)),
            batch_gamma_policy=tuple(sorted(policy)),
        )

    def overhead_ms(self, gamma: int) -> float:
        return dict(self.per_gamma_overhead_ms).get(gamma, 0.0)

    def preferred_gamma(self, batch_size: int) -> int | None:
        """Return the calibrated gamma for the smallest matching batch bucket."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for maximum_batch_size, gamma in self.batch_gamma_policy:
            if batch_size <= maximum_batch_size:
                return gamma
        if self.batch_gamma_policy:
            return self.batch_gamma_policy[-1][1]
        return None

    def predict_step_latency_ms(
        self,
        gamma: int,
        batch_size: int,
        context_tokens: int,
    ) -> float:
        if gamma < 0 or gamma > self.max_speculative_tokens:
            raise ValueError(f"gamma must be in [0, {self.max_speculative_tokens}]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if gamma == 0:
            draft_latency = 0.0
        elif self.draft_parallel:
            draft_latency = self.draft.predict(
                context_tokens,
                batch_size * (gamma + 1),
            )
        else:
            draft_latency = sum(
                self.draft.predict(
                    context_tokens + draft_index * batch_size,
                    batch_size,
                )
                for draft_index in range(gamma)
            )
        target_latency = self.target.predict(
            context_tokens,
            batch_size * (gamma + 1),
        )
        return draft_latency + target_latency + self.overhead_ms(gamma)
