from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_hust_vspec.adaptive.controller import (
    GoodputController,
    expected_generated_tokens,
    expected_generated_tokens_by_position,
)
from vllm_hust_vspec.adaptive.entropy import (
    EntropyDraftStopper,
    normalized_topk_entropy,
)
from vllm_hust_vspec.adaptive.fitting import fit_profile
from vllm_hust_vspec.adaptive.online import OnlineGammaController
from vllm_hust_vspec.adaptive.profile import AdaptiveProfile
from vllm_hust_vspec.adaptive.profiling import (
    build_profile_from_measurements,
    measurement_to_samples,
)
from vllm_hust_vspec.adaptive.runtime import (
    _adaptive_scheduled_batch_size,
    _adaptive_target_query_lens,
    _add_adaptive_decode_graph_keys,
    _batch_descriptor_query_width,
    _copy_dynamic_draft_tokens,
    _current_eagle_confidence_accept_enabled,
    _current_target_is_target_only,
    _disable_fixed_width_eagle_state_kernel,
    _draft_entropy_matrix,
    _full_graph_batch_matches_capture,
    _initialize_adaptive_decode_only_graph_keys,
    _install_draft_entropy_probe,
    _nonuniform_batch_descriptor,
    _prepare_target_graph_params,
    _proposal_execution_gamma,
    _proposal_input_query_width,
    _run_with_runtime_gamma,
    _runner_query_width,
    _runtime_dynamic_eagle_state_kernel,
    _runtime_eager_dispatch,
    _runtime_eager_proposer,
    _runtime_runner_query_width,
    _runtime_target_query_width,
    _TargetOnlyLatch,
    _trim_placeholder_suffixes,
    _uniform_decode_fits_capture_bucket,
    _update_async_next_frame_gamma,
    _uses_width_isolated_target_graph_params,
)
from vllm_hust_vspec.cli import (
    build_environment,
    build_vllm_command,
    generate_capture_sizes,
    parse_args,
)
from vllm_hust_vspec.config import (
    ENV_ADAPTIVE_ENTROPY_STOP,
    ENV_ADAPTIVE_GAMMA0_MODE,
    ENV_ADAPTIVE_LATENCY_CALIBRATION,
    ENV_ADAPTIVE_LATENCY_EWMA_WEIGHT,
    ENV_ADAPTIVE_MAX_GAMMA,
    ENV_ADAPTIVE_ONLINE_EXPLORATION,
    ENV_ADAPTIVE_ONLINE_WARMUP_RETURN,
    ENV_ADAPTIVE_ONLINE_WARMUP_SAMPLES,
    ENV_ADAPTIVE_ONLINE_WINDOW,
    ENV_ADAPTIVE_POLICY,
    ENV_ADAPTIVE_PROFILE_PATH,
    ENV_ADAPTIVE_SPECULATION,
)


def profile_document(default_acceptance_rate: float = 0.9) -> dict[str, object]:
    return {
        "schema_version": 1,
        "max_speculative_tokens": 4,
        "initial_gamma": 4,
        "default_acceptance_rate": default_acceptance_rate,
        "target": {
            "alpha_ms_per_context_token": 0.0,
            "gamma_ms_per_batched_token": 0.01,
            "delta_ms": 100.0,
        },
        "draft": {
            "alpha_ms_per_context_token": 0.0,
            "gamma_ms_per_batched_token": 0.0,
            "delta_ms": 30.0,
        },
    }


class AdaptiveControllerTest(unittest.TestCase):
    def test_adaptive_batch_size_falls_back_to_scheduled_requests(self) -> None:
        scheduler = SimpleNamespace()
        self.assertEqual(_adaptive_scheduled_batch_size(scheduler, 17), 17)

    def test_adaptive_batch_size_uses_legacy_scheduler_hook(self) -> None:
        scheduler = SimpleNamespace(
            _get_dynamic_sd_batch_size=lambda num_requests: num_requests + 3,
        )
        self.assertEqual(_adaptive_scheduled_batch_size(scheduler, 17), 20)

    def test_adaptive_decode_graph_keys_cover_runtime_widths(self) -> None:
        from vllm.config import CUDAGraphMode

        class Dispatcher:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            uniform_decode_query_len = 5
            compilation_config = SimpleNamespace(
                cudagraph_capture_sizes=[4, 8, 12, 16],
            )
            vllm_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=8),
            )

            def __init__(self) -> None:
                self.keys = set()

            def _get_lora_cases(self):
                return [0]

            def _create_padded_batch_descriptor(
                self,
                size,
                uniform,
                has_lora,
                num_active_loras,
            ):
                return (size, size // self.uniform_decode_query_len, uniform)

            def add_cudagraph_key(self, mode, descriptor):
                self.keys.add((mode, descriptor))

        dispatcher = Dispatcher()
        _add_adaptive_decode_graph_keys(dispatcher, (1, 2, 4))

        descriptors = {descriptor for _, descriptor in dispatcher.keys}
        self.assertIn((8, 8, True), descriptors)
        self.assertIn((16, 8, True), descriptors)
        self.assertIn((16, 4, True), descriptors)
        self.assertEqual(dispatcher.uniform_decode_query_len, 5)

    def test_decode_only_initialization_filters_mixed_width_buckets(self) -> None:
        from vllm.config import CUDAGraphMode

        class Dispatcher:
            uniform_decode_query_len = 5
            compilation_config = SimpleNamespace(
                cudagraph_capture_sizes=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20],
                max_cudagraph_capture_size=20,
            )
            vllm_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=4),
            )

            def __init__(self) -> None:
                self.keys = set()
                self.keys_initialized = False

            def _compute_bs_to_padded_graph_size(self) -> None:
                maximum = self.compilation_config.max_cudagraph_capture_size
                sizes = self.compilation_config.cudagraph_capture_sizes
                self._bs_to_padded_graph_size = [0] * (maximum + 1)
                for value in range(maximum + 1):
                    self._bs_to_padded_graph_size[value] = next(
                        (size for size in sizes if size >= value),
                        maximum,
                    )

            def _get_lora_cases(self):
                return [0]

            def _create_padded_batch_descriptor(
                self,
                size,
                uniform,
                has_lora,
                num_active_loras,
            ):
                padded = self._bs_to_padded_graph_size[size]
                assert padded % self.uniform_decode_query_len == 0
                return (padded, padded // self.uniform_decode_query_len, uniform)

            def add_cudagraph_key(self, mode, descriptor):
                self.keys.add((mode, descriptor))

        dispatcher = Dispatcher()
        _initialize_adaptive_decode_only_graph_keys(
            dispatcher,
            CUDAGraphMode.FULL_DECODE_ONLY,
            (1, 2, 3, 4, 5),
        )

        self.assertTrue(dispatcher.keys_initialized)
        self.assertEqual(dispatcher.uniform_decode_query_len, 5)
        self.assertIn((CUDAGraphMode.FULL, (12, 4, True)), dispatcher.keys)
        self.assertIn((CUDAGraphMode.FULL, (20, 4, True)), dispatcher.keys)
        self.assertNotIn((CUDAGraphMode.FULL, (3, 0, True)), dispatcher.keys)

    def test_expected_generated_tokens(self) -> None:
        self.assertAlmostEqual(expected_generated_tokens(0.5, 3), 1.875)
        self.assertEqual(expected_generated_tokens(1.0, 4), 5.0)

    def test_expected_generated_tokens_uses_conditional_position_rates(self) -> None:
        self.assertEqual(
            expected_generated_tokens_by_position([1.0, 0.5, 0.0, 1.0], 4),
            2.5,
        )

    def test_profile_rejects_nonpositive_latency_model(self) -> None:
        document = profile_document()
        document["target"] = {
            "alpha_ms_per_context_token": 0,
            "gamma_ms_per_batched_token": 0,
            "delta_ms": 0,
        }
        with self.assertRaisesRegex(ValueError, "positive forward latency"):
            AdaptiveProfile.from_mapping(document)

    def test_profile_selects_gamma_by_batch_bucket(self) -> None:
        document = profile_document()
        document["batch_gamma_policy"] = {"8": 4, "64": 3, "128": 2}
        profile = AdaptiveProfile.from_mapping(document)
        self.assertEqual(profile.preferred_gamma(1), 4)
        self.assertEqual(profile.preferred_gamma(8), 4)
        self.assertEqual(profile.preferred_gamma(32), 3)
        self.assertEqual(profile.preferred_gamma(128), 2)
        self.assertEqual(profile.preferred_gamma(256), 2)

    def test_profile_rejects_policy_gamma_above_capacity(self) -> None:
        document = profile_document()
        document["batch_gamma_policy"] = {"8": 5}
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            AdaptiveProfile.from_mapping(document)

    def test_profile_accepts_target_only_policy(self) -> None:
        document = profile_document()
        document["initial_gamma"] = 0
        document["batch_gamma_policy"] = {"128": 0}
        profile = AdaptiveProfile.from_mapping(document)
        self.assertEqual(profile.initial_gamma, 0)
        self.assertEqual(profile.preferred_gamma(128), 0)

    def test_parallel_draft_profile_models_one_block_forward(self) -> None:
        document = profile_document()
        document["draft_parallel"] = True
        profile = AdaptiveProfile.from_mapping(document)
        self.assertTrue(profile.draft_parallel)
        expected = profile.target.predict(128, 8 * 3) + profile.draft.predict(
            128,
            8 * 3,
        )
        self.assertEqual(profile.predict_step_latency_ms(2, 8, 128), expected)

    def test_profile_rejects_nonboolean_parallel_flag(self) -> None:
        document = profile_document()
        document["draft_parallel"] = 1
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            AdaptiveProfile.from_mapping(document)

    def test_profile_fitting_recovers_linear_coefficients(self) -> None:
        def samples(alpha: float, gamma: float, delta: float) -> list[dict[str, float]]:
            points = ((100, 8), (100, 32), (500, 8), (500, 32))
            return [
                {
                    "context_tokens": context,
                    "batched_tokens": batch,
                    "latency_ms": alpha * context + gamma * batch + delta,
                }
                for context, batch in points
            ]

        profile = fit_profile(
            {
                "max_speculative_tokens": 4,
                "default_acceptance_rate": 0.8,
                "batch_gamma_policy": {"8": 4, "128": 2},
                "draft_parallel": True,
                "target_samples": samples(0.02, 0.5, 3.0),
                "draft_samples": samples(0.01, 0.2, 1.0),
            }
        )
        self.assertAlmostEqual(
            profile["target"]["alpha_ms_per_context_token"],
            0.02,
        )
        self.assertAlmostEqual(
            profile["target"]["gamma_ms_per_batched_token"],
            0.5,
        )
        self.assertAlmostEqual(profile["draft"]["delta_ms"], 1.0)
        self.assertEqual(profile["batch_gamma_policy"], {"8": 4, "128": 2})
        self.assertTrue(profile["draft_parallel"])

    def test_profile_fitting_rejects_nonboolean_parallel_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            fit_profile(
                {
                    "max_speculative_tokens": 2,
                    "draft_parallel": "false",
                    "target_samples": [
                        {
                            "context_tokens": context,
                            "batched_tokens": batch,
                            "latency_ms": context + batch,
                        }
                        for context, batch in (
                            (1, 1),
                            (1, 2),
                            (2, 1),
                        )
                    ],
                    "draft_samples": [
                        {
                            "context_tokens": context,
                            "batched_tokens": batch,
                            "latency_ms": context + batch,
                        }
                        for context, batch in (
                            (1, 1),
                            (1, 2),
                            (2, 1),
                        )
                    ],
                }
            )

    def test_component_stats_convert_to_forward_samples(self) -> None:
        benchmark = {
            "prompt_lengths": [100, 200],
            "output_lengths": [20, 40],
        }
        component_stats = {
            "target": {
                "spec": {
                    "calls": 10,
                    "reqs": 80,
                    "tokens": 240,
                    "forward_npu_ms": 100.0,
                    "logits_npu_ms": 10.0,
                    "sampler_npu_ms": 10.0,
                    "graph_replay_npu_ms": 140.0,
                }
            }
        }
        target, draft = measurement_to_samples(
            benchmark,
            component_stats,
            gamma=2,
        )
        self.assertEqual(target["context_tokens"], 1463)
        self.assertEqual(target["batched_tokens"], 24)
        self.assertAlmostEqual(float(target["latency_ms"]), 12.0)
        self.assertIsNotNone(draft)
        assert draft is not None
        self.assertEqual(draft["context_tokens"], 1467)
        self.assertEqual(draft["batched_tokens"], 8)
        self.assertAlmostEqual(float(draft["latency_ms"]), 2.0)

        _, parallel_draft = measurement_to_samples(
            benchmark,
            component_stats,
            gamma=2,
            parallel_drafting=True,
        )
        self.assertIsNotNone(parallel_draft)
        assert parallel_draft is not None
        self.assertEqual(parallel_draft["context_tokens"], 1463)
        self.assertEqual(parallel_draft["batched_tokens"], 24)
        self.assertAlmostEqual(float(parallel_draft["latency_ms"]), 4.0)

    def test_profile_suite_uses_measured_throughput_policy(self) -> None:
        def measurement(
            batch_size: int,
            gamma: int,
            throughput: float,
            prompt_length: int,
        ) -> dict[str, object]:
            calls = 10
            return {
                "batch_size": batch_size,
                "gamma": gamma,
                "benchmark": {
                    "prompt_lengths": [prompt_length],
                    "output_lengths": [32],
                    "output_tokens_per_s": throughput,
                },
                "component_stats": {
                    "target": {
                        "spec": {
                            "calls": calls,
                            "reqs": calls * batch_size,
                            "tokens": calls * batch_size * (gamma + 1),
                            "forward_npu_ms": calls * (5 + batch_size),
                            "logits_npu_ms": calls * 0.5,
                            "sampler_npu_ms": calls * 0.5,
                            "graph_replay_npu_ms": (calls * (5 + batch_size + gamma * 2)),
                        }
                    }
                },
            }

        profile = build_profile_from_measurements(
            [
                measurement(2, 1, 100.0, 100),
                measurement(2, 2, 120.0, 100),
                measurement(4, 1, 180.0, 200),
                measurement(4, 2, 170.0, 200),
            ],
            max_gamma=3,
            default_acceptance_rate=0.8,
        )
        self.assertEqual(profile["batch_gamma_policy"], {"2": 2, "4": 1})
        AdaptiveProfile.from_mapping(profile)

    def test_controller_moves_toward_low_gamma_after_rejections(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            ewma_weight=1.0,
            min_observations=1,
            max_gamma_step=4,
            hysteresis=0,
        )
        controller.observe(4, 0)
        decision = controller.choose(batch_size=8, context_tokens=1024)
        self.assertEqual(decision.gamma, 1)
        self.assertEqual(decision.acceptance_rate, 0.0)
        self.assertTrue(decision.changed)

    def test_controller_can_disable_speculation_after_rejections(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_gamma=0,
            ewma_weight=1.0,
            min_observations=1,
            max_gamma_step=4,
            hysteresis=0,
        )
        controller.observe(4, 0)
        decision = controller.choose(batch_size=8, context_tokens=1024)
        self.assertEqual(decision.gamma, 0)
        self.assertEqual(decision.reason, "argmax_goodput")

    def test_controller_filters_candidates_by_batch_token_budget(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_gamma=0,
            max_batched_tokens=24,
        )
        decision = controller.choose(batch_size=8, context_tokens=1024)
        self.assertEqual(decision.gamma, 2)
        self.assertEqual(decision.reason, "batch_token_budget")

    def test_step_limit_does_not_force_a_goodput_regression(self) -> None:
        document = profile_document()
        document["per_gamma_overhead_ms"] = {"3": 100.0}
        profile = AdaptiveProfile.from_mapping(document)
        controller = GoodputController(
            profile,
            max_gamma=4,
            ewma_weight=1.0,
            min_observations=1,
            max_gamma_step=1,
            hysteresis=0,
        )
        controller.observe(4, 0)
        decision = controller.choose(batch_size=8, context_tokens=1024)
        self.assertEqual(decision.gamma, 4)
        self.assertEqual(decision.reason, "step_limit")

    def test_controller_aggregates_prefix_acceptance_trials(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            ewma_weight=1.0,
            min_observations=1,
        )
        controller.observe(4, 2)
        controller.observe(4, 4)
        decision = controller.choose(batch_size=2, context_tokens=128)
        self.assertAlmostEqual(decision.acceptance_rate, 6 / 7)
        self.assertEqual(controller.total_observations, 7)

    def test_controller_tracks_conditional_acceptance_by_position(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            ewma_weight=1.0,
            min_observations=1,
        )
        controller.observe(4, 2)
        controller.choose(batch_size=2, context_tokens=128)
        self.assertEqual(
            controller.position_acceptance_rates,
            [1.0, 1.0, 0.0, profile.default_acceptance_rate],
        )

    def test_online_acceptance_can_override_batch_policy(self) -> None:
        document = profile_document()
        document["initial_gamma"] = 4
        document["batch_gamma_policy"] = {"8": 4}
        profile = AdaptiveProfile.from_mapping(document)
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_gamma=0,
            ewma_weight=1.0,
            min_observations=1,
            max_gamma_step=4,
            hysteresis=0,
        )
        controller.observe(4, 0)
        decision = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual(decision.gamma, 0)
        self.assertEqual(decision.reason, "online_policy_override")

    def test_controller_follows_batch_policy_with_step_limit(self) -> None:
        document = profile_document()
        document["initial_gamma"] = 2
        document["batch_gamma_policy"] = {"8": 4, "128": 2}
        profile = AdaptiveProfile.from_mapping(document)
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_observations=1,
            max_gamma_step=1,
        )
        controller.observe(2, 2)
        first = controller.choose(batch_size=8, context_tokens=128)
        second = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual((first.gamma, first.reason), (3, "batch_policy_step"))
        self.assertEqual((second.gamma, second.reason), (4, "batch_policy"))

    def test_warmup_policy_can_resume_from_target_only(self) -> None:
        document = profile_document()
        document["initial_gamma"] = 0
        document["batch_gamma_policy"] = {"1": 2, "2": 0}
        profile = AdaptiveProfile.from_mapping(document)
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_gamma=0,
            min_observations=32,
            max_gamma_step=1,
        )
        target_only = controller.choose(batch_size=2, context_tokens=128)
        first_resume = controller.choose(batch_size=1, context_tokens=128)
        second_resume = controller.choose(batch_size=1, context_tokens=128)
        self.assertEqual(target_only.gamma, 0)
        self.assertEqual(
            (first_resume.gamma, first_resume.reason),
            (1, "batch_policy_warmup_step"),
        )
        self.assertEqual(
            (second_resume.gamma, second_resume.reason),
            (2, "batch_policy_warmup"),
        )

    def test_positive_gamma_stays_stable_during_warmup(self) -> None:
        document = profile_document()
        document["initial_gamma"] = 2
        document["batch_gamma_policy"] = {"8": 4}
        controller = GoodputController(
            AdaptiveProfile.from_mapping(document),
            max_gamma=4,
            min_observations=32,
        )
        decision = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual((decision.gamma, decision.reason), (2, "warmup"))

    def test_online_latency_calibration_changes_goodput_ranking(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_gamma=0,
            ewma_weight=1.0,
            latency_ewma_weight=1.0,
            min_observations=1,
            max_gamma_step=4,
            hysteresis=0,
        )
        batch_size = 8
        context_tokens = 128
        predicted = profile.predict_step_latency_ms(
            4,
            batch_size,
            context_tokens,
        )
        controller.observe_step_latency(
            4,
            batch_size,
            context_tokens,
            predicted * 10,
        )
        controller.observe_step_latency(
            3,
            batch_size,
            context_tokens,
            profile.predict_step_latency_ms(3, batch_size, context_tokens),
        )
        controller.observe(4, 4)
        decision = controller.choose(batch_size, context_tokens)
        self.assertEqual(decision.gamma, 3)
        self.assertEqual(controller.latency_observations[4], 1)
        self.assertAlmostEqual(controller.latency_correction_factors[4], 10.0)

    def test_unobserved_gamma_uses_global_latency_correction(self) -> None:
        profile = AdaptiveProfile.from_mapping(profile_document())
        controller = GoodputController(
            profile,
            max_gamma=4,
            min_gamma=0,
            latency_ewma_weight=1.0,
        )
        predicted = profile.predict_step_latency_ms(2, 8, 128)
        controller.observe_step_latency(2, 8, 128, predicted * 2)
        self.assertAlmostEqual(controller.latency_correction_factor(2), 2.0)
        self.assertAlmostEqual(controller.latency_correction_factor(3), 2.0)

    def test_online_controller_learns_from_measured_goodput(self) -> None:
        controller = OnlineGammaController(
            max_gamma=3,
            min_gamma=1,
            max_gamma_step=1,
            exploration=0,
            hysteresis=0,
            warmup_samples=1,
            burn_in_steps=0,
        )
        first = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual(
            (first.gamma, first.reason),
            (3, "online_ucb_warmup_window"),
        )

        controller.begin_step_feedback()
        for _ in range(8):
            controller.observe(3, 0)
        controller.complete_step(
            gamma=3,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        second = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual((second.gamma, second.reason), (2, "online_ucb_warmup"))

        controller.begin_step_feedback()
        for _ in range(8):
            controller.observe(2, 2)
        controller.complete_step(
            gamma=2,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        third = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual((third.gamma, third.reason), (1, "online_ucb_warmup"))
        controller.begin_step_feedback()
        for _ in range(8):
            controller.observe(1, 0)
        controller.complete_step(
            gamma=1,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        learned = controller.choose(batch_size=8, context_tokens=128)
        self.assertEqual(
            (learned.gamma, learned.reason),
            (2, "online_ucb_warmup_return"),
        )
        self.assertGreater(
            controller.predicted_goodput(2, 8, 128),
            controller.predicted_goodput(3, 8, 128),
        )

    def test_online_ucb_samples_both_arms(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            min_gamma=1,
            burn_in_steps=0,
        )
        decision = controller.choose(128, 128)
        self.assertEqual(decision.gamma, 2)
        controller.complete_step(
            gamma=2,
            batch_size=128,
            context_tokens=128,
            latency_ms=10,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 2)

    def test_online_ucb_warmup_visits_every_arm(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            exploration=0,
            hysteresis=0,
            warmup_samples=1,
            burn_in_steps=0,
        )
        visited = []
        for expected in (4, 3, 2, 1):
            decision = controller.choose(8, 128)
            visited.append(decision.gamma)
            self.assertEqual(decision.gamma, expected)
            controller.complete_step(
                gamma=expected,
                batch_size=8,
                context_tokens=128,
                latency_ms=10,
            )
        bucket = controller._bucket(8)
        self.assertEqual(visited, [4, 3, 2, 1])
        self.assertTrue(all(bucket.arms[gamma].observations == 1 for gamma in visited))

    def test_online_ucb_prefers_higher_gamma_inside_deadband(self) -> None:
        controller = OnlineGammaController(
            max_gamma=3,
            exploration=0,
            hysteresis=0.03,
            warmup_samples=1,
            burn_in_steps=0,
        )
        for gamma, latency in ((3, 10), (2, 9.8), (1, 20)):
            self.assertEqual(controller.choose(8, 128).gamma, gamma)
            controller.complete_step(
                gamma=gamma,
                batch_size=8,
                context_tokens=128,
                latency_ms=latency,
            )
        decision = controller.choose(8, 128)
        self.assertEqual(
            (decision.gamma, decision.reason),
            (2, "online_ucb_warmup_return"),
        )

    def test_online_ucb_returns_to_best_arm_after_initial_sweep(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            min_gamma=1,
            control_interval=1,
            exploration=0,
            hysteresis=0.2,
            warmup_samples=1,
            burn_in_steps=0,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 2)
        controller.complete_step(
            gamma=2,
            batch_size=128,
            context_tokens=128,
            latency_ms=10,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 1)
        controller.complete_step(
            gamma=1,
            batch_size=128,
            context_tokens=128,
            latency_ms=11,
        )

        returned = controller.choose(128, 128)

        self.assertEqual(
            (returned.gamma, returned.reason),
            (2, "online_ucb_warmup_return"),
        )

    def test_online_ucb_warmup_respects_incumbent_deadband(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            min_gamma=1,
            control_interval=1,
            exploration=0,
            hysteresis=0.2,
            warmup_samples=1,
            burn_in_steps=0,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 2)
        controller.complete_step(
            gamma=2,
            batch_size=128,
            context_tokens=128,
            latency_ms=10,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 1)
        controller.complete_step(
            gamma=1,
            batch_size=128,
            context_tokens=128,
            latency_ms=9.5,
        )

        returned = controller.choose(128, 128)

        self.assertEqual(
            (returned.gamma, returned.reason),
            (2, "online_ucb_warmup_return"),
        )

    def test_online_ucb_can_return_to_configured_incumbent(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            min_gamma=1,
            control_interval=1,
            exploration=0,
            hysteresis=0,
            warmup_samples=1,
            warmup_return="incumbent",
            burn_in_steps=0,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 2)
        controller.complete_step(
            gamma=2,
            batch_size=128,
            context_tokens=128,
            latency_ms=20,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 1)
        controller.complete_step(
            gamma=1,
            batch_size=128,
            context_tokens=128,
            latency_ms=5,
        )

        returned = controller.choose(128, 128)

        self.assertEqual(
            (returned.gamma, returned.reason),
            (2, "online_ucb_warmup_return"),
        )
        self.assertEqual(controller.summary()["warmup_return"], "incumbent")

    def test_online_ucb_uses_symmetric_deadband_when_recovering_upward(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            exploration=0,
            hysteresis=0.1,
            warmup_samples=1,
            burn_in_steps=0,
        )
        bucket = controller._bucket(8)
        bucket.launched_arms.update((1, 2))
        bucket.warmup_return_completed = True
        bucket.arms[1].record(100, 100)
        bucket.arms[2].record(105, 100)
        bucket.selected_gamma = 1
        bucket.scheduled_since_selection = 1
        controller.current_gamma = 1
        decision = controller.choose(8, 128)
        self.assertEqual(
            (decision.gamma, decision.reason),
            (1, "online_ucb_hysteresis"),
        )

    def test_online_ucb_explores_under_sampled_arm(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            exploration=1,
            hysteresis=0,
            warmup_samples=1,
            burn_in_steps=0,
        )
        for gamma in (2, 1):
            self.assertEqual(controller.choose(8, 128).gamma, gamma)
            controller.complete_step(
                gamma=gamma,
                batch_size=8,
                context_tokens=128,
                latency_ms=10,
            )
        self.assertEqual(controller.choose(8, 128).gamma, 2)
        for _ in range(3):
            controller.complete_step(
                gamma=2,
                batch_size=8,
                context_tokens=128,
                latency_ms=10,
            )
        self.assertEqual(controller.choose(8, 128).gamma, 1)

    def test_online_controller_isolates_batch_buckets(self) -> None:
        controller = OnlineGammaController(max_gamma=2, burn_in_steps=0)
        controller.begin_step_feedback()
        controller.observe(2, 2)
        controller.complete_step(
            gamma=2,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        self.assertGreater(controller.predicted_goodput(2, 8, 128), 0)
        self.assertEqual(controller._bucket(16).arms[2].observations, 0)

    def test_online_controller_summary_records_decisions_and_rewards(self) -> None:
        controller = OnlineGammaController(max_gamma=2, burn_in_steps=0)
        controller.choose(batch_size=8, context_tokens=128)
        controller.begin_step_feedback()
        controller.observe(2, 1)
        controller.complete_step(
            gamma=2,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        summary = controller.summary()
        self.assertEqual(summary["selection_counts"], {"2": 1})
        bucket = summary["batch_buckets"]["8"]
        self.assertEqual(bucket["arms"]["2"]["observations"], 1)

    def test_online_controller_tracks_request_second_token_transitions(self) -> None:
        controller = OnlineGammaController(max_gamma=2, burn_in_steps=0)
        controller.observe(2, 2, request_id="accepted")
        controller.observe(2, 2, request_id="accepted")
        controller.observe(2, 1, request_id="accepted")
        controller.observe(2, 1, request_id="rejected")
        controller.observe(2, 2, request_id="rejected")

        transition = controller.summary()["second_token_transition"]
        self.assertEqual(transition["counts"], [[0, 1], [1, 1]])
        self.assertEqual(transition["accept_after_reject"], 1.0)
        self.assertEqual(transition["accept_after_accept"], 0.5)

    def test_online_reward_uses_total_tokens_over_total_latency(self) -> None:
        controller = OnlineGammaController(max_gamma=1, burn_in_steps=0)
        for useful_tokens, latency_ms in ((10, 1), (1, 9)):
            controller.begin_step_feedback()
            for _ in range(useful_tokens):
                controller.observe(0, 0)
            controller.complete_step(
                gamma=1,
                batch_size=8,
                context_tokens=128,
                latency_ms=latency_ms,
            )
        self.assertAlmostEqual(controller.predicted_goodput(1, 8, 128), 1.1)

    def test_online_ucb_initializes_each_batch_bucket(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            exploration=0,
            warmup_samples=1,
            burn_in_steps=0,
        )
        controller.choose(128, 128)
        controller.begin_step_feedback()
        controller.observe(4, 4)
        controller.complete_step(
            gamma=4,
            batch_size=128,
            context_tokens=128,
            latency_ms=10,
        )
        self.assertEqual(controller.choose(128, 128).gamma, 3)
        decision = controller.choose(64, 128)
        self.assertEqual(
            (decision.gamma, decision.reason),
            (4, "online_ucb_bucket_transfer"),
        )
        self.assertEqual(controller._bucket(64).selected_gamma, 4)
        self.assertTrue(controller._bucket(64).launched_arms)
        self.assertEqual(controller._bucket(64).arms[4].observations, 0)

    def test_online_bucket_transfer_uses_near_best_neighbor_arm(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            min_gamma=0,
            exploration=0,
            hysteresis=0.2,
            warmup_samples=1,
            burn_in_steps=0,
        )
        source = controller._bucket(128)
        for gamma, reward in enumerate((1.3, 3.1, 3.0, 2.7, 2.3)):
            source.arms[gamma].record(int(reward * 100), 100)
            source.launched_arms.add(gamma)
        source.warmup_return_completed = True
        controller.current_gamma = 4

        first = controller.choose(64, 128)
        second = controller.choose(64, 128)

        self.assertEqual(
            (first.gamma, first.reason),
            (3, "online_ucb_bucket_transfer"),
        )
        self.assertEqual(
            (second.gamma, second.reason),
            (2, "online_ucb_bucket_transfer"),
        )

    def test_online_warmup_uses_midpoint_for_near_tied_rewards(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            min_gamma=0,
            exploration=0,
            hysteresis=0.2,
            warmup_samples=1,
            burn_in_steps=0,
        )
        bucket = controller._bucket(128)
        for gamma, reward in enumerate((1.3, 2.4, 2.62, 2.65, 2.3)):
            bucket.arms[gamma].record(int(reward * 100), 100)

        self.assertEqual(
            controller._preferred_measured_gamma(
                bucket,
                list(range(5)),
            ),
            2,
        )

    def test_online_wide_range_defers_exploration_until_safe_burn_in(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            min_gamma=0,
            window_size=1,
            warmup_samples=1,
            burn_in_steps=0,
        )
        for _ in range(8):
            decision = controller.choose(128, 128)
            self.assertEqual(
                (decision.gamma, decision.reason),
                (2, "online_ucb_safe_burn_in"),
            )
            controller.complete_step(
                gamma=2,
                batch_size=128,
                context_tokens=128,
                latency_ms=10,
            )
        self.assertEqual(controller.choose(128, 128).gamma, 3)

    def test_eagle_confidence_accept_is_strict_above_gamma_two(self) -> None:
        self.assertTrue(_current_eagle_confidence_accept_enabled(3, 4))
        self.assertFalse(_current_eagle_confidence_accept_enabled(4, 1))
        self.assertFalse(_current_eagle_confidence_accept_enabled(5, 0))
        self.assertTrue(_current_eagle_confidence_accept_enabled(None, 2))
        self.assertFalse(_current_eagle_confidence_accept_enabled(None, 3))

    def test_online_observation_count_outlives_reward_window(self) -> None:
        controller = OnlineGammaController(
            max_gamma=1,
            window_size=2,
            burn_in_steps=0,
        )
        for _ in range(4):
            controller.complete_step(
                gamma=1,
                batch_size=8,
                context_tokens=128,
                latency_ms=10,
            )
        arm = controller._bucket(8).arms[1]
        self.assertEqual(arm.observations, 4)
        self.assertEqual(len(arm.latency_ms), 2)

    def test_online_ucb_resets_dwell_after_returning_to_a_bucket(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            control_interval=2,
            burn_in_steps=0,
        )
        controller.choose(128, 128)
        controller.current_gamma = 3
        decision = controller.choose(128, 128)
        bucket = controller._bucket(128)
        self.assertEqual(
            (decision.gamma, decision.reason),
            (3, "online_ucb_warmup_window"),
        )
        self.assertEqual(bucket.selected_gamma, 3)
        self.assertEqual(bucket.selected_at_observations, 0)

    def test_online_ucb_uses_minimum_warmup_sample_window(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            control_interval=2,
            exploration=0,
            warmup_samples=1,
            burn_in_steps=0,
        )
        self.assertEqual(controller.choose(8, 128).gamma, 2)
        controller.complete_step(
            gamma=2,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        explored = controller.choose(8, 128)
        self.assertEqual(
            (explored.gamma, explored.reason),
            (1, "online_ucb_warmup"),
        )

    def test_online_ucb_bounds_last_warmup_arm_without_feedback(self) -> None:
        controller = OnlineGammaController(
            max_gamma=4,
            min_gamma=1,
            control_interval=2,
            warmup_samples=1,
            burn_in_steps=0,
            inflight_warmup_padding=1,
        )
        decisions = [controller.choose(128, 128).gamma for _ in range(16)]
        self.assertEqual(
            decisions[:9],
            [4, 4, 3, 3, 2, 2, 1, 1, 2],
        )
        self.assertTrue(all(gamma == 2 for gamma in decisions[8:]))

    def test_online_reward_uses_nonoverlapping_completion_cadence(self) -> None:
        controller = OnlineGammaController(max_gamma=1, burn_in_steps=0)
        controller.begin_step_feedback()
        controller.observe(1, 1)
        controller.complete_step(
            gamma=1,
            batch_size=8,
            context_tokens=128,
            completed_at_ms=100,
        )
        self.assertEqual(controller.latency_observations[1], 0)

        controller.begin_step_feedback()
        for _ in range(8):
            controller.observe(1, 1)
        controller.complete_step(
            gamma=1,
            batch_size=8,
            context_tokens=128,
            completed_at_ms=110,
        )
        self.assertEqual(controller.latency_observations[1], 1)
        self.assertAlmostEqual(controller.predicted_goodput(1, 8, 128), 1.6)

    def test_online_controller_discards_nonstable_step_reward(self) -> None:
        controller = OnlineGammaController(max_gamma=2, ewma_weight=1)
        controller.begin_step_feedback()
        controller.observe(2, 1)
        controller.discard_step_feedback()
        self.assertEqual(controller.total_observations, 2)
        self.assertEqual(controller.latency_observations, [0, 0, 0])

    def test_online_controller_excludes_initial_stable_steps(self) -> None:
        controller = OnlineGammaController(
            max_gamma=2,
            warmup_samples=2,
        )
        for _ in range(2):
            controller.begin_step_feedback()
            controller.observe(2, 2)
            controller.complete_step(
                gamma=2,
                batch_size=8,
                context_tokens=128,
                latency_ms=100,
            )
        self.assertEqual(controller.latency_observations[2], 0)
        controller.complete_step(
            gamma=2,
            batch_size=8,
            context_tokens=128,
            latency_ms=10,
        )
        self.assertEqual(controller.latency_observations[2], 1)

    def test_entropy_stopper_masks_each_request_in_current_round(self) -> None:
        stopper = EntropyDraftStopper(
            max_gamma=4,
            min_gamma=1,
            threshold=0.3,
            scale=0.15,
        )
        entropies = torch.tensor(
            [
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
            ]
        )
        tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        masked, lengths = stopper.mask_draft_tokens(tokens, entropies)
        self.assertEqual(lengths.tolist(), [2, 4, 1])
        self.assertEqual(
            masked.tolist(),
            [[10, 11, -1, -1], [20, 21, 22, 23], [30, -1, -1, -1]],
        )

    def test_normalized_topk_entropy_bounds(self) -> None:
        uniform = normalized_topk_entropy(torch.zeros(2, 8), topk=8)
        peaked = normalized_topk_entropy(
            torch.tensor([[20.0, 0.0, 0.0, 0.0]]),
            topk=4,
        )
        self.assertTrue(torch.allclose(uniform, torch.ones_like(uniform)))
        self.assertLess(float(peaked.item()), 0.01)

    def test_entropy_probe_collects_logits_without_changing_tokens(self) -> None:
        class Model:
            def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return hidden_states

        proposer = SimpleNamespace(model=Model(), _vspec_entropy_topk=4)
        _install_draft_entropy_probe(proposer)
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        returned = proposer.model.compute_logits(logits)
        self.assertIs(returned, logits)
        entropy = _draft_entropy_matrix(proposer, 1, batch_size=2, topk=4)
        assert entropy is not None
        self.assertEqual(tuple(entropy.shape), (1, 2))
        self.assertGreater(float(entropy[0, 1]), 0)

    def test_entropy_probe_can_skip_low_gamma_graphs(self) -> None:
        class Model:
            def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return hidden_states

        proposer = SimpleNamespace(model=Model(), _vspec_entropy_topk=4)
        _install_draft_entropy_probe(proposer)
        proposer._vspec_entropy_measure_enabled = False
        logits = torch.zeros(2, 4)
        self.assertIs(proposer.model.compute_logits(logits), logits)
        self.assertIsNone(_draft_entropy_matrix(proposer, 1, batch_size=2, topk=4))

    def test_scheduler_draft_rows_trim_placeholder_suffix(self) -> None:
        draft = SimpleNamespace(draft_token_ids=[[1, 2, -1, -1], [3, 4, 5]])
        self.assertIs(_trim_placeholder_suffixes(draft), draft)
        self.assertEqual(draft.draft_token_ids, [[1, 2], [3, 4, 5]])

    def test_runtime_gamma_is_visible_only_during_proposal(self) -> None:
        proposer = SimpleNamespace(
            num_speculative_tokens=4,
            num_draft_steps=4,
            decode_threshold=5,
        )

        def proposal(current: SimpleNamespace) -> tuple[int, int, int]:
            return (
                current.num_speculative_tokens,
                current.num_draft_steps,
                current.decode_threshold,
            )

        self.assertEqual(
            _run_with_runtime_gamma(proposer, 2, proposal),
            (2, 2, 3),
        )
        self.assertEqual(proposer.num_speculative_tokens, 4)
        self.assertEqual(proposer.num_draft_steps, 4)
        self.assertEqual(proposer.decode_threshold, 5)

    def test_eagle_downshift_executes_at_previous_width(self) -> None:
        self.assertEqual(_proposal_execution_gamma("eagle", 3, 4), 4)
        self.assertEqual(_proposal_execution_gamma("eagle", 2, 3), 3)
        self.assertEqual(_proposal_execution_gamma("eagle", 4, 3), 4)
        self.assertEqual(_proposal_execution_gamma("draft", 3, 4), 3)
        self.assertEqual(_proposal_execution_gamma("eagle", 0, 1), 0)

    def test_dynamic_eagle_disables_fixed_width_state_kernel(self) -> None:
        environment = {"VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL": "1"}
        self.assertTrue(_disable_fixed_width_eagle_state_kernel("eagle", 0, 4, environment))
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"],
            "0",
        )
        self.assertEqual(
            environment["HUST_VSPEC_EAGLE_DYNAMIC_UNIFORM_STATE_KERNEL"],
            "1",
        )

    def test_dynamic_eagle_state_kernel_uses_stable_anchor_gamma(self) -> None:
        runner = SimpleNamespace(num_spec_tokens=2)
        environment = {
            "VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL": "0",
            "HUST_VSPEC_EAGLE_DYNAMIC_UNIFORM_STATE_KERNEL": "1",
        }
        with _runtime_dynamic_eagle_state_kernel(
            runner,
            stable_decode=True,
            gamma=2,
            environ=environment,
        ):
            self.assertEqual(runner.num_spec_tokens, 2)
            self.assertEqual(
                environment["VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"],
                "1",
            )
        self.assertEqual(runner.num_spec_tokens, 2)
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"],
            "0",
        )

    def test_dynamic_eagle_state_kernel_stays_off_for_transition(self) -> None:
        runner = SimpleNamespace(num_spec_tokens=2)
        environment = {
            "VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL": "0",
            "HUST_VSPEC_EAGLE_DYNAMIC_UNIFORM_STATE_KERNEL": "1",
        }
        with _runtime_dynamic_eagle_state_kernel(
            runner,
            stable_decode=False,
            gamma=2,
            environ=environment,
        ):
            self.assertEqual(runner.num_spec_tokens, 2)
            self.assertEqual(
                environment["VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"],
                "0",
            )

    def test_fixed_eagle_keeps_fixed_width_state_kernel(self) -> None:
        environment = {"VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL": "1"}
        self.assertFalse(_disable_fixed_width_eagle_state_kernel("eagle", 4, 4, environment))
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"],
            "1",
        )

    def test_parallel_runtime_gamma_updates_slot_widths_temporarily(self) -> None:
        proposer = SimpleNamespace(
            num_speculative_tokens=4,
            num_draft_steps=4,
            decode_threshold=5,
            parallel_drafting=True,
            pass_hidden_states_to_model=True,
            method="dflash",
            extra_slots_per_request=4,
            net_num_new_slots_per_request=4,
            needs_extra_input_slots=True,
        )

        def proposal(current: SimpleNamespace) -> tuple[int, int, int, bool]:
            return (
                current.num_speculative_tokens,
                current.extra_slots_per_request,
                current.net_num_new_slots_per_request,
                current.needs_extra_input_slots,
            )

        self.assertEqual(
            _run_with_runtime_gamma(proposer, 2, proposal),
            (2, 2, 2, True),
        )
        self.assertEqual(
            (
                proposer.num_speculative_tokens,
                proposer.extra_slots_per_request,
                proposer.net_num_new_slots_per_request,
            ),
            (4, 4, 4),
        )

    def test_parallel_eagle_runtime_accounts_for_reused_hidden_slot(self) -> None:
        proposer = SimpleNamespace(
            num_speculative_tokens=4,
            num_draft_steps=4,
            decode_threshold=5,
            parallel_drafting=True,
            pass_hidden_states_to_model=True,
            method="eagle3",
            extra_slots_per_request=4,
            net_num_new_slots_per_request=3,
            needs_extra_input_slots=True,
        )
        observed = _run_with_runtime_gamma(
            proposer,
            1,
            lambda current: (
                current.extra_slots_per_request,
                current.net_num_new_slots_per_request,
                current.needs_extra_input_slots,
            ),
        )
        self.assertEqual(observed, (1, 0, False))
        self.assertEqual(proposer.net_num_new_slots_per_request, 3)

    def test_dynamic_draft_copy_preserves_capacity_and_clears_tail(self) -> None:
        destination = torch.full((3, 4), 99, dtype=torch.int32)
        source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        _copy_dynamic_draft_tokens(destination, source)
        self.assertEqual(
            destination.tolist(),
            [[1, 2, -1, -1], [3, 4, -1, -1], [99, 99, 99, 99]],
        )

    def test_target_only_draft_copy_clears_active_rows(self) -> None:
        destination = torch.full((3, 4), 99, dtype=torch.int32)
        source = torch.empty((2, 0), dtype=torch.int32)
        _copy_dynamic_draft_tokens(destination, source)
        self.assertEqual(
            destination.tolist(),
            [[-1, -1, -1, -1], [-1, -1, -1, -1], [99, 99, 99, 99]],
        )

    def test_target_only_latch_absorbs_new_requests_until_cohort_drains(self) -> None:
        latch = _TargetOnlyLatch()
        latch.enter({"old-1", "old-2"})
        self.assertTrue(latch.refresh({"old-2", "new"}))
        self.assertEqual(latch.request_ids, {"old-2", "new"})
        self.assertTrue(latch.refresh({"new"}))
        self.assertFalse(latch.refresh({"next-cohort"}))
        self.assertFalse(latch.active)

    def test_dynamic_capture_sizes_cover_each_verification_width(self) -> None:
        self.assertEqual(
            generate_capture_sizes(8, 3, dynamic_widths=True),
            [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        )

    def test_adaptive_target_graph_registers_every_query_width(self) -> None:
        self.assertEqual(
            _adaptive_target_query_lens((5,), max_gamma=4),
            (1, 2, 3, 4, 5),
        )

    def test_runtime_target_query_width_uses_current_draft_frame(self) -> None:
        output = SimpleNamespace(
            num_scheduled_tokens={"a": 4, "b": 4},
            scheduled_spec_decode_tokens={
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            },
        )
        self.assertEqual(_runtime_target_query_width(output), 4)
        output.scheduled_spec_decode_tokens["b"] = []
        self.assertIsNone(_runtime_target_query_width(output))

    def test_async_target_only_dispatch_uses_current_verification_width(self) -> None:
        self.assertFalse(_current_target_is_target_only(2, proposal_gamma=0))
        self.assertTrue(_current_target_is_target_only(1, proposal_gamma=1))
        self.assertTrue(_current_target_is_target_only(1, proposal_gamma=0))
        self.assertTrue(_current_target_is_target_only(None, proposal_gamma=0))
        self.assertFalse(_current_target_is_target_only(None, proposal_gamma=1))

    def test_batch_descriptor_query_width_requires_uniform_layout(self) -> None:
        self.assertEqual(
            _batch_descriptor_query_width(SimpleNamespace(num_tokens=12, num_reqs=3, uniform=True)),
            4,
        )
        self.assertIsNone(
            _batch_descriptor_query_width(SimpleNamespace(num_tokens=12, num_reqs=3, uniform=False))
        )
        self.assertIsNone(
            _batch_descriptor_query_width(SimpleNamespace(num_tokens=10, num_reqs=3, uniform=True))
        )

    def test_proposal_input_width_prefers_attention_metadata(self) -> None:
        metadata = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32))
        nonuniform_descriptor = SimpleNamespace(
            num_tokens=8,
            num_reqs=None,
            uniform=False,
        )
        self.assertEqual(
            _proposal_input_query_width(metadata, nonuniform_descriptor),
            4,
        )

    def test_proposal_input_width_rejects_mixed_metadata(self) -> None:
        metadata = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 3, 8], dtype=torch.int32))
        descriptor = SimpleNamespace(
            num_tokens=8,
            num_reqs=2,
            uniform=True,
        )
        self.assertEqual(
            _proposal_input_query_width(metadata, descriptor),
            4,
        )

    def test_eager_proposal_disables_uniform_graph_dispatch(self) -> None:
        @dataclass(frozen=True)
        class Descriptor:
            num_tokens: int
            num_reqs: int
            uniform: bool

        descriptor = Descriptor(8, 2, True)
        eager_descriptor = _nonuniform_batch_descriptor(descriptor)
        self.assertIsNot(eager_descriptor, descriptor)
        self.assertFalse(eager_descriptor.uniform)
        self.assertEqual(eager_descriptor.num_tokens, 8)

    def test_runtime_runner_query_width_is_scoped_to_proposal(self) -> None:
        dispatcher = SimpleNamespace(uniform_decode_query_len=5)
        runner = SimpleNamespace(
            uniform_decode_query_len=5,
            cudagraph_dispatcher=dispatcher,
        )
        proposer = SimpleNamespace(runner=runner)

        with _runtime_runner_query_width(proposer, 4):
            self.assertEqual(runner.uniform_decode_query_len, 4)
            self.assertEqual(dispatcher.uniform_decode_query_len, 4)

        self.assertEqual(runner.uniform_decode_query_len, 5)
        self.assertEqual(dispatcher.uniform_decode_query_len, 5)

    def test_runner_query_width_is_scoped_for_graph_capture(self) -> None:
        dispatcher = SimpleNamespace(uniform_decode_query_len=5)
        runner = SimpleNamespace(
            uniform_decode_query_len=5,
            cudagraph_dispatcher=dispatcher,
        )

        with _runner_query_width(runner, 4):
            self.assertEqual(runner.uniform_decode_query_len, 4)
            self.assertEqual(dispatcher.uniform_decode_query_len, 4)

        self.assertEqual(runner.uniform_decode_query_len, 5)
        self.assertEqual(dispatcher.uniform_decode_query_len, 5)

    def test_eager_dispatch_disables_uniform_decode_temporarily(self) -> None:
        calls: list[bool] = []

        class Dispatcher:
            def dispatch(self, *, uniform_decode: bool) -> bool:
                calls.append(uniform_decode)
                return uniform_decode

        dispatcher = Dispatcher()
        original_dispatch = dispatcher.dispatch
        proposer = SimpleNamespace(runner=SimpleNamespace(cudagraph_dispatcher=dispatcher))

        with _runtime_eager_dispatch(proposer, True):
            self.assertFalse(dispatcher.dispatch(uniform_decode=True))

        self.assertEqual(calls, [False])
        self.assertEqual(dispatcher.dispatch, original_dispatch)

    def test_eager_proposer_disables_graph_temporarily(self) -> None:
        proposer = SimpleNamespace(use_cuda_graph=True)
        with _runtime_eager_proposer(proposer, True):
            self.assertFalse(proposer.use_cuda_graph)
        self.assertTrue(proposer.use_cuda_graph)

    def test_dynamic_uniform_width_rejects_incompatible_bucket(self) -> None:
        dispatcher = SimpleNamespace(_bs_to_padded_graph_size=[0, 1, 2, 3, 4, 5, 6, 8, 8])
        self.assertTrue(_uniform_decode_fits_capture_bucket(dispatcher, 8, 4))
        self.assertFalse(_uniform_decode_fits_capture_bucket(dispatcher, 8, 5))

    def test_target_graph_params_are_isolated_by_query_width(self) -> None:
        @dataclass
        class GraphParams:
            events: dict
            workspaces: dict
            handles: dict
            attn_params: dict
            conv1d_params: dict
            conv1d_handles: dict
            conv1d_events: dict

        base = GraphParams(*({8: [], 16: []} for _ in range(7)))
        runner = SimpleNamespace()
        tables = _prepare_target_graph_params(
            runner,
            base,
            (1, 2, 3, 4, 5),
        )
        assert tables is not None
        self.assertIs(tables[5], base)
        self.assertIsNot(tables[1], base)
        self.assertEqual(tuple(tables[1].events), (8, 16))
        self.assertIsNone(tables[1].workspaces[8])
        self.assertIsNone(tables[1].workspaces[16])
        self.assertIsNot(tables[1].events[8], tables[1].events[16])
        tables[1].events[8].append("event")
        self.assertEqual(tables[1].events[16], [])
        self.assertIs(
            _prepare_target_graph_params(runner, base, (1, 2, 3, 4, 5)),
            tables,
        )

    def test_only_full_decode_uses_width_isolated_target_graph_params(
        self,
    ) -> None:
        self.assertTrue(
            _uses_width_isolated_target_graph_params(SimpleNamespace(name="FULL_DECODE_ONLY"))
        )
        self.assertFalse(_uses_width_isolated_target_graph_params(SimpleNamespace(name="FULL")))

    def test_full_graph_batch_guard_rejects_bucket_collision(self) -> None:
        metadata = SimpleNamespace(batch_size=lambda: 2, num_actual_tokens=6)
        exact_descriptor = SimpleNamespace(num_tokens=6, num_reqs=2)
        padded_metadata = SimpleNamespace(
            batch_size=lambda: 2,
            num_actual_tokens=7,
        )
        safely_padded_descriptor = SimpleNamespace(num_tokens=9, num_reqs=8)
        collided_descriptor = SimpleNamespace(num_tokens=10, num_reqs=2)
        self.assertTrue(
            _full_graph_batch_matches_capture(
                metadata,
                exact_descriptor,
                gamma=2,
                capture_sizes=[6, 9, 10],
            )
        )
        self.assertTrue(
            _full_graph_batch_matches_capture(
                padded_metadata,
                safely_padded_descriptor,
                gamma=2,
                capture_sizes=[6, 9, 10],
            )
        )
        self.assertFalse(
            _full_graph_batch_matches_capture(
                metadata,
                collided_descriptor,
                gamma=2,
                capture_sizes=[6, 9, 10],
            )
        )

    def test_async_next_frame_uses_adaptive_gamma(self) -> None:
        active_request = SimpleNamespace(
            is_prefill_chunk=False,
            spec_token_ids=[-1] * 4,
        )
        prefill_request = SimpleNamespace(
            is_prefill_chunk=True,
            spec_token_ids=[-1] * 4,
        )
        scheduler = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=True),
            requests={
                "active": active_request,
                "prefill": prefill_request,
            },
            _spec_token_placeholders=[-1] * 4,
        )
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={"active": 3, "prefill": 16},
        )

        _update_async_next_frame_gamma(scheduler, scheduler_output, gamma=2)

        self.assertEqual(scheduler._spec_token_placeholders, [-1, -1])
        self.assertEqual(active_request.spec_token_ids, [-1, -1])
        self.assertEqual(prefill_request.spec_token_ids, [-1] * 4)

    def test_adaptive_launcher_uses_piecewise_and_sync_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps(profile_document()),
                encoding="utf-8",
            )
            options, configured_environment = parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/draft",
                    "--gamma",
                    "4",
                    "--adaptive-speculation",
                    "--adaptive-profile",
                    str(profile_path),
                    "--no-adaptive-full-graph",
                    "--no-adaptive-async",
                    "--graph-mode",
                    "full",
                    "--vllm-executable",
                    "/usr/bin/vllm",
                ]
            )
            self.assertEqual(options.graph_mode, "piecewise")
            self.assertFalse(options.async_scheduling)
            self.assertFalse(options.merged_full)

            command = build_vllm_command(options)
            compilation = command[command.index("--compilation-config") + 1]
            self.assertIn('"cudagraph_mode":"PIECEWISE"', compilation)
            self.assertIn("--no-async-scheduling", command)

            environment = build_environment(
                options,
                configured_environment,
                {},
            )
            self.assertEqual(environment[ENV_ADAPTIVE_SPECULATION], "1")
            self.assertEqual(environment[ENV_ADAPTIVE_MAX_GAMMA], "4")
            self.assertEqual(
                environment[ENV_ADAPTIVE_PROFILE_PATH],
                str(profile_path.resolve()),
            )

    def test_adaptive_full_graph_and_async_are_explicit_opt_ins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps(profile_document()),
                encoding="utf-8",
            )
            options, _ = parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/eagle",
                    "--method",
                    "eagle",
                    "--gamma",
                    "4",
                    "--adaptive-speculation",
                    "--adaptive-profile",
                    str(profile_path),
                    "--adaptive-full-graph",
                    "--adaptive-async",
                    "--graph-mode",
                    "full",
                    "--vllm-executable",
                    "/usr/bin/vllm",
                ]
            )
            self.assertEqual(options.graph_mode, "full")
            self.assertTrue(options.async_scheduling)
            command = build_vllm_command(options)
            compilation = command[command.index("--compilation-config") + 1]
            self.assertIn('"cudagraph_mode":"FULL"', compilation)
            self.assertIn("--async-scheduling", command)

    def test_dflash_adaptive_requires_parallel_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps(profile_document()),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target-model",
                        "/models/target",
                        "--draft-model",
                        "/models/dflash",
                        "--method",
                        "dflash",
                        "--gamma",
                        "4",
                        "--adaptive-speculation",
                        "--adaptive-profile",
                        str(profile_path),
                    ]
                )

            parallel_document = profile_document()
            parallel_document["draft_parallel"] = True
            profile_path.write_text(
                json.dumps(parallel_document),
                encoding="utf-8",
            )
            options, _ = parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/dflash",
                    "--method",
                    "dflash",
                    "--gamma",
                    "4",
                    "--adaptive-speculation",
                    "--adaptive-profile",
                    str(profile_path),
                ]
            )
            self.assertEqual(options.method, "dflash")

    def test_profile_policy_requires_profile(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/draft",
                    "--adaptive-speculation",
                    "--adaptive-policy",
                    "profile",
                ]
            )

    def test_online_launcher_needs_no_profile(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--gamma",
                "4",
                "--adaptive-speculation",
                "--adaptive-policy",
                "online",
                "--adaptive-online-window",
                "16",
                "--adaptive-online-exploration",
                "0.25",
                "--adaptive-online-warmup-samples",
                "2",
                "--adaptive-online-warmup-return",
                "incumbent",
                "--adaptive-entropy-stop",
            ]
        )
        self.assertIsNone(options.adaptive_profile)
        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_ADAPTIVE_POLICY], "online")
        self.assertEqual(environment[ENV_ADAPTIVE_PROFILE_PATH], "")
        self.assertEqual(environment[ENV_ADAPTIVE_ONLINE_WINDOW], "16")
        self.assertEqual(environment[ENV_ADAPTIVE_ONLINE_EXPLORATION], "0.25")
        self.assertEqual(environment[ENV_ADAPTIVE_ONLINE_WARMUP_SAMPLES], "2")
        self.assertEqual(
            environment[ENV_ADAPTIVE_ONLINE_WARMUP_RETURN],
            "incumbent",
        )
        self.assertEqual(environment[ENV_ADAPTIVE_ENTROPY_STOP], "1")

    def test_online_launcher_rejects_sticky_gamma_zero(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/draft",
                    "--adaptive-speculation",
                    "--adaptive-policy",
                    "online",
                    "--adaptive-min-gamma",
                    "0",
                ]
            )

    def test_adaptive_launcher_accepts_zero_min_gamma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps(profile_document()),
                encoding="utf-8",
            )
            options, configured_environment = parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/draft",
                    "--gamma",
                    "4",
                    "--adaptive-speculation",
                    "--adaptive-profile",
                    str(profile_path),
                    "--adaptive-min-gamma",
                    "0",
                    "--adaptive-gamma0-mode",
                    "sync",
                    "--no-adaptive-latency-calibration",
                    "--adaptive-latency-ewma-weight",
                    "0.35",
                ]
            )
            self.assertEqual(options.adaptive_min_gamma, 0)
            self.assertEqual(options.adaptive_gamma0_mode, "sync")
            self.assertFalse(options.adaptive_latency_calibration)
            self.assertEqual(options.adaptive_latency_ewma_weight, 0.35)
            environment = build_environment(
                options,
                configured_environment,
                {},
            )
            self.assertEqual(environment["HUST_VSPEC_ADAPTIVE_MIN_GAMMA"], "0")
            self.assertEqual(environment[ENV_ADAPTIVE_GAMMA0_MODE], "sync")
            self.assertEqual(
                environment[ENV_ADAPTIVE_LATENCY_CALIBRATION],
                "0",
            )
            self.assertEqual(
                environment[ENV_ADAPTIVE_LATENCY_EWMA_WEIGHT],
                "0.35",
            )


if __name__ == "__main__":
    unittest.main()
