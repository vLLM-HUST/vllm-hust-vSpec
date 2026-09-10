"""Fit vSpec Adaptive linear latency profiles from measured samples."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .profile import AdaptiveProfile, ForwardLatencyModel

SAMPLE_KEYS = frozenset({"context_tokens", "batched_tokens", "latency_ms"})


def fit_forward_latency(
    samples: Sequence[Mapping[str, Any]],
    component: str,
) -> ForwardLatencyModel:
    """Fit a nonnegative three-coefficient linear model by active sets."""
    if len(samples) < 3:
        raise ValueError(f"{component}_samples requires at least three samples")

    rows: list[list[float]] = []
    latencies: list[float] = []
    for index, sample in enumerate(samples):
        unknown = sorted(set(sample) - SAMPLE_KEYS)
        missing = sorted(SAMPLE_KEYS - set(sample))
        if unknown or missing:
            raise ValueError(
                f"invalid {component}_samples[{index}]: unknown={unknown}, missing={missing}"
            )
        context_tokens = int(sample["context_tokens"])
        batched_tokens = int(sample["batched_tokens"])
        latency_ms = float(sample["latency_ms"])
        if (
            context_tokens < 0
            or batched_tokens <= 0
            or not math.isfinite(latency_ms)
            or latency_ms <= 0
        ):
            raise ValueError(f"invalid nonpositive value in {component}_samples[{index}]")
        rows.append([float(context_tokens), float(batched_tokens), 1.0])
        latencies.append(latency_ms)

    design = np.asarray(rows, dtype=np.float64)
    observed = np.asarray(latencies, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError(f"{component}_samples must vary context_tokens and batched_tokens")

    # With three coefficients, enumerating active variable subsets gives the
    # exact nonnegative least-squares solution without a SciPy dependency.
    best_coefficients: np.ndarray | None = None
    best_residual = float("inf")
    for mask in range(1, 1 << 3):
        active = [index for index in range(3) if mask & (1 << index)]
        active_design = design[:, active]
        fitted, *_ = np.linalg.lstsq(active_design, observed, rcond=None)
        if np.any(fitted < -1e-12):
            continue
        coefficients = np.zeros(3, dtype=np.float64)
        coefficients[active] = np.maximum(fitted, 0)
        residual = float(np.square(design @ coefficients - observed).sum())
        if residual < best_residual:
            best_coefficients = coefficients
            best_residual = residual

    if best_coefficients is None:
        raise ValueError(f"cannot fit a nonnegative {component} latency model")
    model = ForwardLatencyModel(
        alpha_ms_per_context_token=float(best_coefficients[0]),
        gamma_ms_per_batched_token=float(best_coefficients[1]),
        delta_ms=float(best_coefficients[2]),
    )
    if model.predict(1, 1) <= 0:
        raise ValueError(f"fitted {component} latency model is not positive")
    return model


def fit_profile(document: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "max_speculative_tokens",
        "initial_gamma",
        "default_acceptance_rate",
        "target_samples",
        "draft_samples",
        "draft_parallel",
        "per_gamma_overhead_ms",
        "batch_gamma_policy",
    }
    unknown = sorted(set(document) - allowed_keys)
    if unknown:
        raise ValueError(f"unknown fitting input field(s): {', '.join(unknown)}")
    if "max_speculative_tokens" not in document:
        raise ValueError("fitting input requires max_speculative_tokens")
    target_samples = document.get("target_samples")
    draft_samples = document.get("draft_samples")
    if not isinstance(target_samples, list) or not isinstance(draft_samples, list):
        raise ValueError("fitting input requires target_samples and draft_samples arrays")

    target = fit_forward_latency(target_samples, "target")
    draft = fit_forward_latency(draft_samples, "draft")
    profile = {
        "schema_version": 1,
        "max_speculative_tokens": int(document["max_speculative_tokens"]),
        "initial_gamma": int(document.get("initial_gamma", document["max_speculative_tokens"])),
        "default_acceptance_rate": float(document.get("default_acceptance_rate", 0.7)),
        "target": {
            "alpha_ms_per_context_token": target.alpha_ms_per_context_token,
            "gamma_ms_per_batched_token": target.gamma_ms_per_batched_token,
            "delta_ms": target.delta_ms,
        },
        "draft": {
            "alpha_ms_per_context_token": draft.alpha_ms_per_context_token,
            "gamma_ms_per_batched_token": draft.gamma_ms_per_batched_token,
            "delta_ms": draft.delta_ms,
        },
        "draft_parallel": document.get("draft_parallel", False),
        "per_gamma_overhead_ms": document.get("per_gamma_overhead_ms", {}),
        "batch_gamma_policy": document.get("batch_gamma_policy", {}),
    }
    AdaptiveProfile.from_mapping(profile)
    return profile


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-vspec-fit-profile",
        description="Fit a vSpec Adaptive latency profile from JSON samples.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args(argv)
    try:
        document = json.loads(options.input.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("fitting input root must be a JSON object")
        profile = fit_profile(document)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    options.output.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
