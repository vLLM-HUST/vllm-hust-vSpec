"""Centralized environment contract for the vSpec plugin."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

ENV_ENABLED = "HUST_VSPEC_ENABLED"
ENV_METHOD = "HUST_VSPEC_METHOD"
ENV_ASSUME_SHARED_TOKENIZER = "HUST_VSPEC_SHARED_TOKENIZER_PADDING"
ENV_USE_MERGED_FULL = "HUST_VSPEC_USE_MERGED_FULL"
ENV_MERGED_FULL_MAX_BATCH = "HUST_VSPEC_MERGED_FULL_MAX_BATCH"
ENV_MAX_NUM_SEQS = "HUST_VSPEC_MAX_NUM_SEQS"
ENV_DRAFT_ACTIVE_VOCAB = "HUST_VSPEC_DRAFT_ACTIVE_VOCAB"
ENV_DRAFT_TARGET_ACTIVE_VOCAB = "HUST_VSPEC_DRAFT_TARGET_ACTIVE_VOCAB"
ENV_EAGLE_TREE_WIDTH = "HUST_VSPEC_EAGLE_TREE_WIDTH"
ENV_EAGLE_DRAFT_ACTIVE_VOCAB = "HUST_VSPEC_EAGLE_DRAFT_ACTIVE_VOCAB"
ENV_EAGLE_TARGET_ACTIVE_VOCAB = "HUST_VSPEC_EAGLE_TARGET_ACTIVE_VOCAB"
ENV_EAGLE_RELAXED_ACCEPT_TOPK = "HUST_VSPEC_EAGLE_RELAXED_ACCEPT_TOPK"
ENV_CONFIDENCE_ACCEPT_MARGIN = "HUST_VSPEC_CONFIDENCE_ACCEPT_MARGIN"
ENV_CONFIDENCE_ACCEPT_FROM_POSITION = "HUST_VSPEC_CONFIDENCE_ACCEPT_FROM_POSITION"
ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS = "HUST_VSPEC_CONFIDENCE_ACCEPT_AFTER_TOKENS"
ENV_CONFIDENCE_PROTECTED_TOKEN_IDS = "HUST_VSPEC_CONFIDENCE_PROTECTED_TOKEN_IDS"
ENV_ADAPTIVE_SPECULATION = "HUST_VSPEC_ADAPTIVE_SPECULATION"
ENV_ADAPTIVE_POLICY = "HUST_VSPEC_ADAPTIVE_POLICY"
ENV_ADAPTIVE_PROFILE_PATH = "HUST_VSPEC_ADAPTIVE_PROFILE_PATH"
ENV_ADAPTIVE_MAX_GAMMA = "HUST_VSPEC_ADAPTIVE_MAX_GAMMA"
ENV_ADAPTIVE_MIN_GAMMA = "HUST_VSPEC_ADAPTIVE_MIN_GAMMA"
ENV_ADAPTIVE_EWMA_WEIGHT = "HUST_VSPEC_ADAPTIVE_EWMA_WEIGHT"
ENV_ADAPTIVE_CONTROL_INTERVAL = "HUST_VSPEC_ADAPTIVE_CONTROL_INTERVAL"
ENV_ADAPTIVE_HYSTERESIS = "HUST_VSPEC_ADAPTIVE_HYSTERESIS"
ENV_ADAPTIVE_MIN_OBSERVATIONS = "HUST_VSPEC_ADAPTIVE_MIN_OBSERVATIONS"
ENV_ADAPTIVE_MAX_GAMMA_STEP = "HUST_VSPEC_ADAPTIVE_MAX_GAMMA_STEP"
ENV_ADAPTIVE_LATENCY_CALIBRATION = "HUST_VSPEC_ADAPTIVE_LATENCY_CALIBRATION"
ENV_ADAPTIVE_LATENCY_EWMA_WEIGHT = "HUST_VSPEC_ADAPTIVE_LATENCY_EWMA_WEIGHT"
ENV_ADAPTIVE_GAMMA0_MODE = "HUST_VSPEC_ADAPTIVE_GAMMA0_MODE"
ENV_ADAPTIVE_TRACE = "HUST_VSPEC_ADAPTIVE_TRACE"
ENV_ADAPTIVE_FULL_GRAPH = "HUST_VSPEC_ADAPTIVE_FULL_GRAPH"
ENV_ADAPTIVE_ASYNC = "HUST_VSPEC_ADAPTIVE_ASYNC"
ENV_ADAPTIVE_ONLINE_WINDOW = "HUST_VSPEC_ADAPTIVE_ONLINE_WINDOW"
ENV_ADAPTIVE_ONLINE_EXPLORATION = "HUST_VSPEC_ADAPTIVE_ONLINE_EXPLORATION"
ENV_ADAPTIVE_ONLINE_WARMUP_SAMPLES = "HUST_VSPEC_ADAPTIVE_ONLINE_WARMUP_SAMPLES"
ENV_ADAPTIVE_ONLINE_WARMUP_RETURN = "HUST_VSPEC_ADAPTIVE_ONLINE_WARMUP_RETURN"
ENV_ADAPTIVE_REFILL_BATCH = "HUST_VSPEC_ADAPTIVE_REFILL_BATCH"
ENV_ADAPTIVE_ENTROPY_STOP = "HUST_VSPEC_ADAPTIVE_ENTROPY_STOP"
ENV_ADAPTIVE_ENTROPY_TOPK = "HUST_VSPEC_ADAPTIVE_ENTROPY_TOPK"
ENV_ADAPTIVE_ENTROPY_THRESHOLD = "HUST_VSPEC_ADAPTIVE_ENTROPY_THRESHOLD"
ENV_ADAPTIVE_ENTROPY_SCALE = "HUST_VSPEC_ADAPTIVE_ENTROPY_SCALE"
ENV_PREFIX = "HUST_VSPEC_"
LEGACY_ENV_PREFIXES = (
    "HUST_SPECSLO_",
    "HUST_SPECASCEND_",
)

METHOD_ALIASES = {
    "draft": "draft_model",
    "draft_model": "draft_model",
    "eagle": "eagle",
    "eagle3": "eagle3",
    "dflash": "dflash",
}

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})
ADAPTIVE_GAMMA0_MODES = frozenset({"sticky", "sync"})
ADAPTIVE_POLICIES = frozenset({"profile", "online"})
ADAPTIVE_ONLINE_WARMUP_RETURNS = frozenset({"best", "incumbent"})


def _read_raw(environment: Mapping[str, str], name: str) -> str | None:
    if name in environment:
        return environment[name]
    for legacy_prefix in LEGACY_ENV_PREFIXES:
        legacy_name = name.replace(ENV_PREFIX, legacy_prefix, 1)
        if legacy_name in environment:
            return environment[legacy_name]
    return None


def _read_bool(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = _read_raw(environment, name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of 0/1, true/false, yes/no, or on/off")


def _read_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
) -> int:
    raw_value = _read_raw(environment, name)
    value = default if raw_value is None else int(raw_value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _read_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    raw_value = _read_raw(environment, name)
    value = default if raw_value is None else float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    below_minimum = value < minimum if minimum_inclusive else value <= minimum
    if below_minimum or (maximum is not None and value > maximum):
        interval = "[" if minimum_inclusive else "("
        upper = "inf" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be in {interval}{minimum}, {upper}], got {value}")
    return value


def _read_optional_float(
    environment: Mapping[str, str],
    name: str,
    minimum: float,
) -> float | None:
    raw_value = _read_raw(environment, name)
    if raw_value is None or not raw_value.strip():
        return None
    value = float(raw_value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value}")
    return value


def _read_int_tuple(
    environment: Mapping[str, str],
    name: str,
) -> tuple[int, ...]:
    raw_value = _read_raw(environment, name)
    if raw_value is None or not raw_value.strip():
        return ()
    values = tuple(int(value.strip()) for value in raw_value.split(","))
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")
    return values


def _read_choice(
    environment: Mapping[str, str],
    name: str,
    default: str,
    choices: frozenset[str],
) -> str:
    value = (_read_raw(environment, name) or default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")
    return value


@dataclass(frozen=True)
class PluginSettings:
    enabled: bool = False
    method: str = "draft_model"
    assume_shared_tokenizer: bool = True
    use_merged_full: bool = True
    merged_full_max_batch: int = 0
    max_num_seqs: int = 128
    draft_active_vocab: bool = False
    draft_target_active_vocab: bool = False
    eagle_tree_width: int = 1
    eagle_draft_active_vocab: bool = False
    eagle_target_active_vocab: bool = False
    eagle_relaxed_accept_topk: int = 1
    confidence_accept_margin: float | None = None
    confidence_accept_from_position: int = 0
    confidence_accept_after_tokens: int = 0
    confidence_protected_token_ids: tuple[int, ...] = ()
    adaptive_speculation: bool = False
    adaptive_policy: str = "profile"
    adaptive_profile_path: str = ""
    adaptive_max_gamma: int = 1
    adaptive_min_gamma: int = 1
    adaptive_ewma_weight: float = 0.1
    adaptive_control_interval: int = 4
    adaptive_hysteresis: float = 0.05
    adaptive_min_observations: int = 32
    adaptive_max_gamma_step: int = 1
    adaptive_latency_calibration: bool = True
    adaptive_latency_ewma_weight: float = 0.2
    adaptive_gamma0_mode: str = "sticky"
    adaptive_trace: bool = False
    adaptive_full_graph: bool = False
    adaptive_async: bool = False
    adaptive_online_window: int = 32
    adaptive_online_exploration: float = 0.01
    adaptive_online_warmup_samples: int = 1
    adaptive_online_warmup_return: str = "best"
    adaptive_refill_batch: int = 0
    adaptive_entropy_stop: bool = False
    adaptive_entropy_topk: int = 2
    adaptive_entropy_threshold: float = 0.3
    adaptive_entropy_scale: float = 0.15

    def __post_init__(self) -> None:
        if self.confidence_accept_margin is not None and (
            not math.isfinite(self.confidence_accept_margin) or self.confidence_accept_margin < 0
        ):
            raise ValueError(f"{ENV_CONFIDENCE_ACCEPT_MARGIN} must be finite and nonnegative")
        if self.confidence_accept_from_position < 0:
            raise ValueError(f"{ENV_CONFIDENCE_ACCEPT_FROM_POSITION} must be nonnegative")
        if self.confidence_accept_after_tokens < 0:
            raise ValueError(f"{ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS} must be nonnegative")
        if any(token_id < 0 for token_id in self.confidence_protected_token_ids):
            raise ValueError(f"{ENV_CONFIDENCE_PROTECTED_TOKEN_IDS} must be nonnegative")
        if self.adaptive_refill_batch < 0:
            raise ValueError(f"{ENV_ADAPTIVE_REFILL_BATCH} must be nonnegative")
        if self.adaptive_policy not in ADAPTIVE_POLICIES:
            raise ValueError(
                f"{ENV_ADAPTIVE_POLICY} must be one of {', '.join(sorted(ADAPTIVE_POLICIES))}"
            )
        if (
            self.adaptive_speculation
            and self.adaptive_policy == "profile"
            and not self.adaptive_profile_path
        ):
            raise ValueError(
                f"{ENV_ADAPTIVE_PROFILE_PATH} is required when {ENV_ADAPTIVE_POLICY}=profile"
            )
        if self.adaptive_entropy_stop and self.adaptive_policy != "online":
            raise ValueError(f"{ENV_ADAPTIVE_ENTROPY_STOP}=1 requires {ENV_ADAPTIVE_POLICY}=online")
        if (
            self.adaptive_speculation
            and self.adaptive_policy == "online"
            and not self.adaptive_latency_calibration
        ):
            raise ValueError(
                f"{ENV_ADAPTIVE_LATENCY_CALIBRATION}=1 is required when "
                f"{ENV_ADAPTIVE_POLICY}=online"
            )
        if self.adaptive_min_gamma > self.adaptive_max_gamma:
            raise ValueError(f"{ENV_ADAPTIVE_MIN_GAMMA} cannot exceed {ENV_ADAPTIVE_MAX_GAMMA}")
        if self.adaptive_gamma0_mode not in ADAPTIVE_GAMMA0_MODES:
            raise ValueError(
                f"{ENV_ADAPTIVE_GAMMA0_MODE} must be one of "
                f"{', '.join(sorted(ADAPTIVE_GAMMA0_MODES))}"
            )
        if self.adaptive_online_warmup_return not in ADAPTIVE_ONLINE_WARMUP_RETURNS:
            raise ValueError(
                f"{ENV_ADAPTIVE_ONLINE_WARMUP_RETURN} must be one of "
                f"{', '.join(sorted(ADAPTIVE_ONLINE_WARMUP_RETURNS))}"
            )
        if self.adaptive_entropy_topk < 2:
            raise ValueError(f"{ENV_ADAPTIVE_ENTROPY_TOPK} must be at least 2")
        if (
            not math.isfinite(self.adaptive_entropy_threshold)
            or self.adaptive_entropy_threshold < 0
        ):
            raise ValueError(f"{ENV_ADAPTIVE_ENTROPY_THRESHOLD} must be finite and nonnegative")
        if not math.isfinite(self.adaptive_entropy_scale) or self.adaptive_entropy_scale <= 0:
            raise ValueError(f"{ENV_ADAPTIVE_ENTROPY_SCALE} must be finite and positive")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> PluginSettings:
        values = os.environ if environment is None else environment
        raw_method = (_read_raw(values, ENV_METHOD) or "draft_model").strip().lower()
        if raw_method not in METHOD_ALIASES:
            choices = ", ".join(sorted(METHOD_ALIASES))
            raise ValueError(f"{ENV_METHOD} must be one of {choices}")
        return cls(
            enabled=_read_bool(values, ENV_ENABLED, False),
            method=METHOD_ALIASES[raw_method],
            assume_shared_tokenizer=_read_bool(
                values,
                ENV_ASSUME_SHARED_TOKENIZER,
                True,
            ),
            use_merged_full=_read_bool(values, ENV_USE_MERGED_FULL, True),
            merged_full_max_batch=_read_int(
                values,
                ENV_MERGED_FULL_MAX_BATCH,
                0,
                0,
            ),
            max_num_seqs=_read_int(values, ENV_MAX_NUM_SEQS, 128, 1),
            draft_active_vocab=_read_bool(
                values,
                ENV_DRAFT_ACTIVE_VOCAB,
                False,
            ),
            draft_target_active_vocab=_read_bool(
                values,
                ENV_DRAFT_TARGET_ACTIVE_VOCAB,
                False,
            ),
            eagle_tree_width=_read_int(
                values,
                ENV_EAGLE_TREE_WIDTH,
                1,
                1,
            ),
            eagle_draft_active_vocab=_read_bool(
                values,
                ENV_EAGLE_DRAFT_ACTIVE_VOCAB,
                False,
            ),
            eagle_target_active_vocab=_read_bool(
                values,
                ENV_EAGLE_TARGET_ACTIVE_VOCAB,
                False,
            ),
            eagle_relaxed_accept_topk=_read_int(
                values,
                ENV_EAGLE_RELAXED_ACCEPT_TOPK,
                1,
                1,
            ),
            confidence_accept_margin=_read_optional_float(
                values,
                ENV_CONFIDENCE_ACCEPT_MARGIN,
                0.0,
            ),
            confidence_accept_from_position=_read_int(
                values,
                ENV_CONFIDENCE_ACCEPT_FROM_POSITION,
                0,
                0,
            ),
            confidence_accept_after_tokens=_read_int(
                values,
                ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS,
                0,
                0,
            ),
            confidence_protected_token_ids=_read_int_tuple(
                values,
                ENV_CONFIDENCE_PROTECTED_TOKEN_IDS,
            ),
            adaptive_speculation=_read_bool(
                values,
                ENV_ADAPTIVE_SPECULATION,
                False,
            ),
            adaptive_policy=_read_choice(
                values,
                ENV_ADAPTIVE_POLICY,
                "profile",
                ADAPTIVE_POLICIES,
            ),
            adaptive_profile_path=(_read_raw(values, ENV_ADAPTIVE_PROFILE_PATH) or ""),
            adaptive_max_gamma=_read_int(
                values,
                ENV_ADAPTIVE_MAX_GAMMA,
                1,
                1,
            ),
            adaptive_min_gamma=_read_int(
                values,
                ENV_ADAPTIVE_MIN_GAMMA,
                1,
                0,
            ),
            adaptive_ewma_weight=_read_float(
                values,
                ENV_ADAPTIVE_EWMA_WEIGHT,
                0.1,
                0.0,
                1.0,
                minimum_inclusive=False,
            ),
            adaptive_control_interval=_read_int(
                values,
                ENV_ADAPTIVE_CONTROL_INTERVAL,
                4,
                1,
            ),
            adaptive_hysteresis=_read_float(
                values,
                ENV_ADAPTIVE_HYSTERESIS,
                0.05,
                0.0,
            ),
            adaptive_min_observations=_read_int(
                values,
                ENV_ADAPTIVE_MIN_OBSERVATIONS,
                32,
                1,
            ),
            adaptive_max_gamma_step=_read_int(
                values,
                ENV_ADAPTIVE_MAX_GAMMA_STEP,
                1,
                1,
            ),
            adaptive_latency_calibration=_read_bool(
                values,
                ENV_ADAPTIVE_LATENCY_CALIBRATION,
                True,
            ),
            adaptive_latency_ewma_weight=_read_float(
                values,
                ENV_ADAPTIVE_LATENCY_EWMA_WEIGHT,
                0.2,
                0.0,
                1.0,
                minimum_inclusive=False,
            ),
            adaptive_gamma0_mode=_read_choice(
                values,
                ENV_ADAPTIVE_GAMMA0_MODE,
                "sticky",
                ADAPTIVE_GAMMA0_MODES,
            ),
            adaptive_trace=_read_bool(
                values,
                ENV_ADAPTIVE_TRACE,
                False,
            ),
            adaptive_full_graph=_read_bool(
                values,
                ENV_ADAPTIVE_FULL_GRAPH,
                False,
            ),
            adaptive_async=_read_bool(
                values,
                ENV_ADAPTIVE_ASYNC,
                False,
            ),
            adaptive_online_window=_read_int(
                values,
                ENV_ADAPTIVE_ONLINE_WINDOW,
                32,
                1,
            ),
            adaptive_online_exploration=_read_float(
                values,
                ENV_ADAPTIVE_ONLINE_EXPLORATION,
                0.01,
                0.0,
            ),
            adaptive_online_warmup_samples=_read_int(
                values,
                ENV_ADAPTIVE_ONLINE_WARMUP_SAMPLES,
                1,
                1,
            ),
            adaptive_online_warmup_return=_read_choice(
                values,
                ENV_ADAPTIVE_ONLINE_WARMUP_RETURN,
                "best",
                ADAPTIVE_ONLINE_WARMUP_RETURNS,
            ),
            adaptive_refill_batch=_read_int(
                values,
                ENV_ADAPTIVE_REFILL_BATCH,
                0,
                0,
            ),
            adaptive_entropy_stop=_read_bool(
                values,
                ENV_ADAPTIVE_ENTROPY_STOP,
                False,
            ),
            adaptive_entropy_topk=_read_int(
                values,
                ENV_ADAPTIVE_ENTROPY_TOPK,
                2,
                2,
            ),
            adaptive_entropy_threshold=_read_float(
                values,
                ENV_ADAPTIVE_ENTROPY_THRESHOLD,
                0.3,
                0.0,
            ),
            adaptive_entropy_scale=_read_float(
                values,
                ENV_ADAPTIVE_ENTROPY_SCALE,
                0.15,
                0.0,
                minimum_inclusive=False,
            ),
        )

    def as_environment(self) -> dict[str, str]:
        environment = {
            ENV_ENABLED: "1" if self.enabled else "0",
            ENV_METHOD: self.method,
            ENV_ASSUME_SHARED_TOKENIZER: ("1" if self.assume_shared_tokenizer else "0"),
            ENV_USE_MERGED_FULL: "1" if self.use_merged_full else "0",
            ENV_MERGED_FULL_MAX_BATCH: str(self.merged_full_max_batch),
            ENV_MAX_NUM_SEQS: str(self.max_num_seqs),
            ENV_DRAFT_ACTIVE_VOCAB: ("1" if self.draft_active_vocab else "0"),
            ENV_DRAFT_TARGET_ACTIVE_VOCAB: ("1" if self.draft_target_active_vocab else "0"),
            ENV_EAGLE_TREE_WIDTH: str(self.eagle_tree_width),
            ENV_EAGLE_DRAFT_ACTIVE_VOCAB: ("1" if self.eagle_draft_active_vocab else "0"),
            ENV_EAGLE_TARGET_ACTIVE_VOCAB: ("1" if self.eagle_target_active_vocab else "0"),
            ENV_EAGLE_RELAXED_ACCEPT_TOPK: str(self.eagle_relaxed_accept_topk),
            ENV_CONFIDENCE_ACCEPT_MARGIN: (
                "" if self.confidence_accept_margin is None else str(self.confidence_accept_margin)
            ),
            ENV_CONFIDENCE_ACCEPT_FROM_POSITION: str(self.confidence_accept_from_position),
            ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS: str(self.confidence_accept_after_tokens),
            ENV_CONFIDENCE_PROTECTED_TOKEN_IDS: ",".join(
                map(str, self.confidence_protected_token_ids)
            ),
            ENV_ADAPTIVE_SPECULATION: ("1" if self.adaptive_speculation else "0"),
            ENV_ADAPTIVE_POLICY: self.adaptive_policy,
            ENV_ADAPTIVE_PROFILE_PATH: self.adaptive_profile_path,
            ENV_ADAPTIVE_MAX_GAMMA: str(self.adaptive_max_gamma),
            ENV_ADAPTIVE_MIN_GAMMA: str(self.adaptive_min_gamma),
            ENV_ADAPTIVE_EWMA_WEIGHT: str(self.adaptive_ewma_weight),
            ENV_ADAPTIVE_CONTROL_INTERVAL: str(self.adaptive_control_interval),
            ENV_ADAPTIVE_HYSTERESIS: str(self.adaptive_hysteresis),
            ENV_ADAPTIVE_MIN_OBSERVATIONS: str(self.adaptive_min_observations),
            ENV_ADAPTIVE_MAX_GAMMA_STEP: str(self.adaptive_max_gamma_step),
            ENV_ADAPTIVE_LATENCY_CALIBRATION: ("1" if self.adaptive_latency_calibration else "0"),
            ENV_ADAPTIVE_LATENCY_EWMA_WEIGHT: str(self.adaptive_latency_ewma_weight),
            ENV_ADAPTIVE_GAMMA0_MODE: self.adaptive_gamma0_mode,
            ENV_ADAPTIVE_TRACE: "1" if self.adaptive_trace else "0",
            ENV_ADAPTIVE_FULL_GRAPH: ("1" if self.adaptive_full_graph else "0"),
            ENV_ADAPTIVE_ASYNC: "1" if self.adaptive_async else "0",
            ENV_ADAPTIVE_ONLINE_WINDOW: str(self.adaptive_online_window),
            ENV_ADAPTIVE_ONLINE_EXPLORATION: str(self.adaptive_online_exploration),
            ENV_ADAPTIVE_ONLINE_WARMUP_SAMPLES: str(self.adaptive_online_warmup_samples),
            ENV_ADAPTIVE_ONLINE_WARMUP_RETURN: (self.adaptive_online_warmup_return),
            ENV_ADAPTIVE_REFILL_BATCH: str(self.adaptive_refill_batch),
            ENV_ADAPTIVE_ENTROPY_STOP: ("1" if self.adaptive_entropy_stop else "0"),
            ENV_ADAPTIVE_ENTROPY_TOPK: str(self.adaptive_entropy_topk),
            ENV_ADAPTIVE_ENTROPY_THRESHOLD: str(self.adaptive_entropy_threshold),
            ENV_ADAPTIVE_ENTROPY_SCALE: str(self.adaptive_entropy_scale),
        }
        return environment
