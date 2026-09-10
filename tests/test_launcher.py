from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from vllm_hust_vspec.backends.draft import (
    _bind_host_attn_update,
    _call_host_query_padding,
    _install_inner_logits_processor_alias,
)
from vllm_hust_vspec.backends.draft_vocab import (
    _configure_serial_draft_active_vocab,
)
from vllm_hust_vspec.backends.eagle_draft import (
    ActiveVocabLogits,
    _configure_draft_active_vocab,
    quantize_active_lm_head_w8a16,
)
from vllm_hust_vspec.backends.eagle_graph import (
    _call_host_graph_init,
    _host_has_native_event_ordering,
)
from vllm_hust_vspec.backends.eagle_rejection import (
    _confidence_accept_inputs,
    _confidence_accept_margin,
    _forward_greedy_token_ids,
    _forward_linear_eagle,
)
from vllm_hust_vspec.backends.eagle_target import (
    _configure_target_active_vocab,
    _configure_target_active_vocab_for_method,
)
from vllm_hust_vspec.cli import (
    build_environment,
    build_vllm_command,
    generate_capture_sizes,
    load_config,
    parse_args,
)
from vllm_hust_vspec.compatibility import (
    inspect_host_compatibility,
)
from vllm_hust_vspec.config import (
    ENV_ADAPTIVE_POLICY,
    ENV_ADAPTIVE_REFILL_BATCH,
    ENV_ADAPTIVE_SPECULATION,
    ENV_ASSUME_SHARED_TOKENIZER,
    ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS,
    ENV_CONFIDENCE_ACCEPT_FROM_POSITION,
    ENV_CONFIDENCE_ACCEPT_MARGIN,
    ENV_CONFIDENCE_PROTECTED_TOKEN_IDS,
    ENV_DRAFT_ACTIVE_VOCAB,
    ENV_DRAFT_TARGET_ACTIVE_VOCAB,
    ENV_EAGLE_RELAXED_ACCEPT_TOPK,
    ENV_ENABLED,
    ENV_MAX_NUM_SEQS,
    ENV_METHOD,
    PluginSettings,
)


class LauncherTest(unittest.TestCase):
    def test_draft_reduce_sample_aliases_outer_logits_processor(self) -> None:
        inner_model = SimpleNamespace()
        logits_processor = object()
        model = SimpleNamespace(
            model=inner_model,
            logits_processor=logits_processor,
        )

        self.assertTrue(_install_inner_logits_processor_alias(model))
        self.assertIs(inner_model.logits_processor, logits_processor)
        self.assertFalse(_install_inner_logits_processor_alias(model))

    def test_draft_reduce_sample_preserves_native_inner_processor(self) -> None:
        native_processor = object()
        inner_model = SimpleNamespace(logits_processor=native_processor)
        model = SimpleNamespace(
            model=inner_model,
            logits_processor=object(),
        )

        self.assertFalse(_install_inner_logits_processor_alias(model))
        self.assertIs(inner_model.logits_processor, native_processor)

    def test_graph_init_filters_argument_removed_by_current_host(self) -> None:
        calls = []

        def current_init(instance, runnable, *, use_eagle=False):
            calls.append((instance, runnable, use_eagle))

        instance = object()
        runnable = object()
        _call_host_graph_init(
            current_init,
            instance,
            runnable,
            use_eagle=True,
            is_draft_model=True,
        )

        self.assertEqual(calls, [(instance, runnable, True)])

    def test_graph_init_preserves_argument_supported_by_legacy_host(self) -> None:
        calls = []

        def legacy_init(instance, runnable, *, is_draft_model=False):
            calls.append((instance, runnable, is_draft_model))

        instance = object()
        runnable = object()
        _call_host_graph_init(
            legacy_init,
            instance,
            runnable,
            is_draft_model=True,
        )

        self.assertEqual(calls, [(instance, runnable, True)])

    def test_query_padding_supports_current_host_signature(self) -> None:
        calls = []

        def host_method(
            runner,
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode=None,
            batch_desc_num_reqs=None,
        ):
            calls.append((cudagraph_runtime_mode, batch_desc_num_reqs))
            return num_reqs_padded

        result = _call_host_query_padding(
            host_method,
            object(),
            object(),
            32,
            8,
            6,
            "FULL",
            8,
            True,
        )

        self.assertEqual(result, 8)
        self.assertEqual(calls, [("FULL", 8)])

    def test_query_padding_supports_legacy_host_signature(self) -> None:
        seen_uniform = []

        def host_method(
            runner,
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode=None,
            batch_desc_num_reqs=None,
            batch_desc_uniform=None,
        ):
            seen_uniform.append(batch_desc_uniform)
            return num_reqs_padded

        _call_host_query_padding(
            host_method,
            object(),
            object(),
            32,
            8,
            6,
            "FULL",
            8,
            False,
        )

        self.assertEqual(seen_uniform, [False])

    def test_attn_update_binding_supports_current_and_legacy_hosts(self) -> None:
        def current(
            self,
            draft_index,
            old_common_metadata,
            batch_size,
            input_batch_size,
            used_update_positions,
            aclgraph_runtime_mode,
            attn_group=None,
        ):
            pass

        def legacy(
            self,
            draft_index,
            old_attn_metadata,
            old_common_metadata,
            batch_size,
            input_batch_size,
            used_update_positions,
            aclgraph_runtime_mode,
            attn_group=None,
        ):
            pass

        current_bound = _bind_host_attn_update(
            current,
            object(),
            1,
            ("common", 8, 8, "positions", "PIECEWISE"),
            {"attn_group": "group"},
        )
        legacy_bound = _bind_host_attn_update(
            legacy,
            object(),
            1,
            ("attention", "common", 8, 8, "positions", "PIECEWISE"),
            {"attn_group": "group"},
        )

        current_bound.arguments["input_batch_size"] = 16
        legacy_bound.arguments["input_batch_size"] = 16
        self.assertEqual(current_bound.arguments["old_common_metadata"], "common")
        self.assertEqual(legacy_bound.arguments["old_common_metadata"], "common")
        self.assertEqual(current_bound.arguments["input_batch_size"], 16)
        self.assertEqual(legacy_bound.arguments["input_batch_size"], 16)

    def test_bare_greedy_verification_uses_token_id_path(self) -> None:
        metadata = SimpleNamespace(max_spec_len=2)
        sampling_metadata = SimpleNamespace(
            all_greedy=True,
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            allowed_token_ids_mask=None,
            bad_words_token_ids=None,
            logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
            thinking_budget_state_holder=None,
        )
        logits = torch.tensor([[0.0, 3.0], [4.0, 1.0]])
        sentinel = object()
        with mock.patch(
            "vllm_hust_vspec.backends.eagle_rejection._forward_greedy_token_ids",
            return_value=sentinel,
        ) as fast_path:
            result = _forward_linear_eagle(
                SimpleNamespace(),
                metadata,
                None,
                logits,
                sampling_metadata,
            )
        self.assertIs(result, sentinel)
        self.assertEqual(fast_path.call_args.args[2].tolist(), [1, 0])

    def test_disabled_confidence_accept_uses_strict_token_id_path(self) -> None:
        metadata = SimpleNamespace(max_spec_len=2)
        sampling_metadata = SimpleNamespace(
            all_greedy=True,
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            allowed_token_ids_mask=None,
            bad_words_token_ids=None,
            logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
            thinking_budget_state_holder=None,
        )
        logits = torch.tensor([[0.0, 3.0], [4.0, 1.0]])
        sampler = SimpleNamespace(
            _vspec_eagle_confidence_accept_enabled=False,
        )
        sentinel = object()
        with (
            mock.patch.dict(
                os.environ,
                {"VSPEC_EAGLE_CONFIDENCE_ACCEPT_MARGIN": "4.0"},
            ),
            mock.patch(
                "vllm_hust_vspec.backends.eagle_rejection._forward_greedy_token_ids",
                return_value=sentinel,
            ) as fast_path,
        ):
            result = _forward_linear_eagle(
                sampler,
                metadata,
                None,
                logits,
                sampling_metadata,
            )
        self.assertIs(result, sentinel)
        self.assertNotIn("relaxed_draft_mask", fast_path.call_args.kwargs)

    def test_serial_draft_active_vocab_maps_full_token_ids(self) -> None:
        class TinyDraftModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 1200, bias=False)

        with tempfile.TemporaryDirectory() as directory:
            ids_path = Path(directory) / "active.json"
            active_ids = list(range(100, 1124))
            ids_path.write_text(str(active_ids), encoding="utf-8")
            model = TinyDraftModel()
            with torch.no_grad():
                model.lm_head.weight.zero_()
                model.lm_head.weight[1100].fill_(1)
            proposer = SimpleNamespace(
                method="draft_model",
                vllm_config=SimpleNamespace(
                    parallel_config=SimpleNamespace(tensor_parallel_size=1),
                    quant_config=None,
                ),
                model=model,
            )
            environment = {
                "VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_IDS_PATH": str(ids_path),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertTrue(_configure_serial_draft_active_vocab(proposer))

        compact_logits = torch.nn.functional.linear(
            torch.ones(2, 4),
            proposer._eagle_draft_active_lm_head_weight,
        )
        selected = proposer._eagle_draft_active_vocab_ids[compact_logits.argmax(dim=-1)]
        self.assertEqual(selected.tolist(), [1100, 1100])
        self.assertFalse(_configure_serial_draft_active_vocab(proposer))

    def test_w8a16_active_lm_head_quantization(self) -> None:
        weight = torch.tensor(
            [[-2.0, -1.0, 0.0, 1.0], [0.5, 1.0, 1.5, 2.0]],
            dtype=torch.float16,
        )
        quant_weight, scale = quantize_active_lm_head_w8a16(weight)
        reconstructed = quant_weight.transpose(0, 1).float() * scale.float()[:, None]
        self.assertEqual(quant_weight.shape, (4, 2))
        self.assertEqual(scale.shape, (2,))
        self.assertTrue(torch.allclose(reconstructed, weight.float(), atol=0.02))

    def test_current_host_abi_is_compatible(self) -> None:
        for method in ("draft", "eagle", "eagle3", "dflash"):
            with self.subTest(method=method):
                report = inspect_host_compatibility(method)
                self.assertTrue(report.compatible)

    def test_current_host_adaptive_abi_is_compatible(self) -> None:
        report = inspect_host_compatibility("eagle", adaptive=True)
        self.assertTrue(report.compatible)

    def test_native_graph_event_ordering_is_not_double_patched(self) -> None:
        class GraphWrapper:
            def wait_for_prior_replay_before_update(self):
                pass

            def __call__(self):
                return self.event_ordered_replay

        class ModelRunner:
            def _update_full_graph_params_if_needed(self):
                return self.model.wait_for_prior_replay_before_update()

        class Proposer:
            def _update_full_graph_params(self):
                return self._runnable.wait_for_prior_replay_before_update()

        self.assertTrue(
            _host_has_native_event_ordering(
                GraphWrapper,
                ModelRunner,
                Proposer,
            )
        )

    def test_draft_active_vocab_maps_full_token_ids(self) -> None:
        class LogitsProcessor:
            def _gather_logits(self, logits):
                return logits

        class TinyDraftModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 1200, bias=False)
                self.model = SimpleNamespace(logits_processor=LogitsProcessor())

            def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.lm_head(hidden_states)

        with tempfile.TemporaryDirectory() as directory:
            ids_path = Path(directory) / "active.json"
            active_ids = list(range(100, 1124))
            ids_path.write_text(str(active_ids), encoding="utf-8")
            model = TinyDraftModel()
            with torch.no_grad():
                model.lm_head.weight.zero_()
                model.lm_head.weight[1100].fill_(1)
            proposer = SimpleNamespace(
                method="eagle",
                vllm_config=SimpleNamespace(
                    parallel_config=SimpleNamespace(tensor_parallel_size=1),
                    quant_config=None,
                ),
                model=model,
            )
            environment = {
                "VLLM_ASCEND_EAGLE_DRAFT_ACTIVE_VOCAB_IDS_PATH": str(ids_path),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                _configure_draft_active_vocab(proposer)

        compact_logits = model.compute_logits(torch.ones(2, 4))
        self.assertIsInstance(compact_logits, ActiveVocabLogits)
        self.assertEqual(compact_logits.shape, (2, 1024))
        self.assertEqual(compact_logits.argmax(dim=-1).tolist(), [1100, 1100])

    def test_target_active_vocab_projection(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 1200, bias=False)

            def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.lm_head(hidden_states)

        with tempfile.TemporaryDirectory() as directory:
            ids_path = Path(directory) / "active.json"
            ids_path.write_text(
                "[" + ",".join(str(index) for index in range(1024)) + "]",
                encoding="utf-8",
            )
            model = TinyModel()
            runner = SimpleNamespace(
                speculative_config=SimpleNamespace(method="eagle"),
                parallel_config=SimpleNamespace(tensor_parallel_size=1),
                vllm_config=SimpleNamespace(quant_config=None),
                model=model,
                max_num_tokens=16,
            )
            environment = {
                "VLLM_ASCEND_EAGLE_TARGET_ACTIVE_VOCAB_IDS_PATH": str(ids_path),
                "VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_TOPK": "1",
                "VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_AFTER_TOKENS": "0",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                _configure_target_active_vocab(runner)
                active_weight = runner._eagle_target_active_lm_head_weight
                _configure_target_active_vocab(runner)

        logits = model.compute_logits(torch.ones(2, 4))
        self.assertEqual(logits.shape, (2, 1024))
        self.assertEqual(runner._eagle_target_active_vocab_ids.numel(), 1024)
        self.assertIs(
            runner._eagle_target_active_lm_head_weight,
            active_weight,
        )

    def test_serial_draft_target_active_vocab_projection(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 1200, bias=False)

            def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.lm_head(hidden_states)

        with tempfile.TemporaryDirectory() as directory:
            ids_path = Path(directory) / "active.json"
            ids_path.write_text(
                "[" + ",".join(str(index) for index in range(1024)) + "]",
                encoding="utf-8",
            )
            runner = SimpleNamespace(
                speculative_config=SimpleNamespace(method="draft_model"),
                parallel_config=SimpleNamespace(tensor_parallel_size=1),
                vllm_config=SimpleNamespace(quant_config=None),
                model=TinyModel(),
                max_num_tokens=16,
            )
            environment_name = "VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH"
            with mock.patch.dict(
                os.environ,
                {environment_name: str(ids_path)},
                clear=False,
            ):
                _configure_target_active_vocab_for_method(
                    runner,
                    active_ids_environment=environment_name,
                    required_method="draft_model",
                    feature_name="Draft Target",
                )

        logits = runner.model.compute_logits(torch.ones(2, 4))
        self.assertEqual(logits.shape, (2, 1024))
        self.assertEqual(runner._eagle_target_active_vocab_ids.numel(), 1024)

    def test_relaxed_eagle_token_id_verifier(self) -> None:
        from vllm_ascend.sample import rejection_sampler as ascend_rejection

        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1]),
            bonus_logits_indices=torch.tensor([2]),
            draft_token_ids=torch.tensor([10, 20], dtype=torch.int32),
            num_draft_tokens=[2],
            cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
            max_spec_len=2,
        )
        target_token_ids = torch.tensor([99, 98, 77], dtype=torch.int64)
        sampler = SimpleNamespace(last_verify_results="stale")
        sampling_metadata = SimpleNamespace(all_greedy=True)
        original_has_triton = ascend_rejection.HAS_TRITON
        ascend_rejection.HAS_TRITON = False
        try:
            output = _forward_greedy_token_ids(
                sampler,
                metadata,
                target_token_ids,
                sampling_metadata,
                relaxed_draft_mask=torch.tensor([True, False]),
            )
        finally:
            ascend_rejection.HAS_TRITON = original_has_triton

        self.assertEqual(output.sampled_token_ids.tolist(), [[10, 98, -1]])
        self.assertIsNone(sampler.last_verify_results)

    def test_topk_token_id_verifier_accepts_candidate(self) -> None:
        from vllm_ascend.sample import rejection_sampler as ascend_rejection

        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1]),
            bonus_logits_indices=torch.tensor([2]),
            draft_token_ids=torch.tensor([10, 20], dtype=torch.int32),
            num_draft_tokens=[2],
            cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
            max_spec_len=2,
        )
        candidates = torch.tensor([[99, 10], [98, 20], [77, 0]])
        sampler = SimpleNamespace(last_verify_results="stale")
        sampling_metadata = SimpleNamespace(all_greedy=True)
        original_has_triton = ascend_rejection.HAS_TRITON
        ascend_rejection.HAS_TRITON = False
        try:
            output = _forward_greedy_token_ids(
                sampler,
                metadata,
                candidates[:, 0],
                sampling_metadata,
                target_token_id_candidates=candidates,
            )
        finally:
            ascend_rejection.HAS_TRITON = original_has_triton

        self.assertEqual(output.sampled_token_ids.tolist(), [[10, 20, 77]])
        self.assertIsNone(sampler.last_verify_results)

    def test_eagle_confidence_accept_compares_draft_logit(self) -> None:
        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1]),
            draft_token_ids=torch.tensor([1, 0], dtype=torch.int32),
            num_draft_tokens=[2],
        )
        logits = torch.tensor(
            [
                [3.0, 2.7, 0.0],
                [2.0, 4.0, 0.0],
                [0.0, 1.0, 5.0],
            ]
        )

        target_ids, relaxed_mask = _confidence_accept_inputs(
            logits,
            metadata,
            0.5,
        )

        self.assertEqual(target_ids.tolist(), [0, 1, 2])
        self.assertEqual(relaxed_mask.tolist(), [True, False])

    def test_eagle_confidence_accept_can_preserve_first_position(self) -> None:
        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1, 2]),
            draft_token_ids=torch.tensor([1, 0, 1], dtype=torch.int32),
            num_draft_tokens=[2, 1],
        )
        logits = torch.tensor(
            [
                [3.0, 2.7],
                [2.0, 1.8],
                [1.0, 0.9],
            ]
        )

        _, relaxed_mask = _confidence_accept_inputs(
            logits,
            metadata,
            0.5,
            min_position=1,
        )

        self.assertEqual(relaxed_mask.tolist(), [False, True, False])

    def test_eagle_confidence_accept_protects_stop_tokens(self) -> None:
        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1, 2]),
            draft_token_ids=torch.tensor([1, 7, 0], dtype=torch.int32),
            num_draft_tokens=[3],
        )
        logits = torch.tensor(
            [
                [0.0, 2.8, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        _, relaxed_mask = _confidence_accept_inputs(
            logits,
            metadata,
            0.5,
            protected_token_ids=(0, 7),
        )

        self.assertEqual(relaxed_mask.tolist(), [True, False, False])

    def test_eagle_confidence_accept_preserves_generated_prefix(self) -> None:
        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1, 2, 3]),
            draft_token_ids=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            num_draft_tokens=[2, 2],
        )
        logits = torch.tensor(
            [
                [3.0, 2.8],
                [2.0, 1.8],
                [1.0, 0.9],
                [0.8, 1.0],
            ]
        )

        _, relaxed_mask = _confidence_accept_inputs(
            logits,
            metadata,
            0.5,
            min_generated_tokens=4,
            output_token_ids=[[1, 2, 3], [1, 2, 3, 4]],
        )

        self.assertEqual(relaxed_mask.tolist(), [False, False, True, True])

    def test_eagle_confidence_accept_keeps_padded_rows_strict(self) -> None:
        metadata = SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1]),
            draft_token_ids=torch.tensor([1, 0], dtype=torch.int32),
            num_draft_tokens=[1, 1],
        )
        logits = torch.tensor([[3.0, 2.8], [2.0, 1.8]])

        _, relaxed_mask = _confidence_accept_inputs(
            logits,
            metadata,
            0.5,
            min_generated_tokens=1,
            output_token_ids=[[1]],
        )

        self.assertEqual(relaxed_mask.tolist(), [True, False])

    def test_capture_sizes_cover_b128_gamma3(self) -> None:
        self.assertEqual(
            generate_capture_sizes(128, 3),
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        )

    def test_eagle_exact_capture_sizes_include_runtime_batches(self) -> None:
        sizes = generate_capture_sizes(32, 2, "eagle", "exact")
        self.assertEqual(
            sizes,
            [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 72, 96],
        )

    def test_adaptive_draft_auto_uses_exact_dynamic_capture_sizes(self) -> None:
        sizes = generate_capture_sizes(
            128,
            4,
            "draft_model",
            "auto",
            dynamic_widths=True,
        )
        self.assertEqual(len(sizes), 55)
        self.assertEqual(sizes[-1], 640)
        self.assertIn(520, sizes)

    def test_config_rejects_unknown_serve_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.toml"
            config_path.write_text("[serve]\nunknown = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown.*option"):
                load_config(config_path)

    def test_packaged_method_configs_build_commands(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected_methods = {
            "qwen25-14b-05b.toml": "draft_model",
            "qwen25-14b-05b-adaptive-b128.toml": "draft_model",
            "qwen25-14b-eagle.toml": "eagle",
            "qwen25-14b-eagle-relaxed.toml": "eagle",
            "qwen3-8b-eagle3.toml": "eagle3",
        }
        for filename, method in expected_methods.items():
            with self.subTest(filename=filename):
                with mock.patch(
                    "vllm_hust_vspec.cli.resolve_default_model",
                    return_value=Path("/models/installed-default"),
                ):
                    options, _ = parse_args(
                        [
                            "--config",
                            str(root / "configs" / filename),
                            "--vllm-executable",
                            "/usr/bin/vllm",
                        ]
                    )
                command = build_vllm_command(options)
                speculative = command[command.index("--speculative-config") + 1]
                self.assertIn(f'"method":"{method}"', speculative)
                generation_config = command[command.index("--generation-config") + 1]
                self.assertEqual(generation_config, "vllm")

    def test_cli_overrides_toml_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "valid.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[serve]",
                        'target_model = "/models/target"',
                        'draft_model = "/models/draft"',
                        "gamma = 4",
                        "max_num_seqs = 64",
                    ]
                ),
                encoding="utf-8",
            )
            options, _ = parse_args(["--config", str(config_path), "--gamma", "3"])
            self.assertEqual(options.gamma, 3)
            self.assertEqual(options.max_num_seqs, 64)

    def test_command_and_environment(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--gamma",
                "3",
                "--max-num-seqs",
                "128",
                "--vllm-executable",
                "/usr/bin/vllm",
                "--device",
                "2",
            ]
        )
        command = build_vllm_command(options)
        speculative_index = command.index("--speculative-config") + 1
        self.assertIn('"num_speculative_tokens":3', command[speculative_index])
        self.assertEqual(
            command[command.index("--generation-config") + 1],
            "vllm",
        )
        capture_index = command.index("--cudagraph-capture-sizes") + 1
        self.assertEqual(
            [int(value) for value in command[capture_index:]],
            generate_capture_sizes(128, 3, "draft_model", "auto", dynamic_widths=True),
        )

        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_ENABLED], "1")
        self.assertEqual(environment[ENV_METHOD], "draft_model")
        self.assertEqual(environment[ENV_ASSUME_SHARED_TOKENIZER], "1")
        self.assertEqual(environment[ENV_MAX_NUM_SEQS], "128")
        self.assertEqual(environment[ENV_ADAPTIVE_SPECULATION], "1")
        self.assertEqual(environment[ENV_ADAPTIVE_POLICY], "online")
        self.assertEqual(environment["ASCEND_RT_VISIBLE_DEVICES"], "2")

    def test_cli_resolves_installed_model_and_allows_explicit_override(self) -> None:
        with mock.patch(
            "vllm_hust_vspec.cli.resolve_default_model",
            return_value=Path("/models/installed-draft"),
        ) as resolver:
            options, _ = parse_args(["--target-model", "/models/target"])

        self.assertEqual(options.draft_model, "/models/installed-draft")
        resolver.assert_called_once()

        with mock.patch("vllm_hust_vspec.cli.resolve_default_model") as resolver:
            explicit, _ = parse_args(
                [
                    "--target-model",
                    "/models/target",
                    "--draft-model",
                    "/models/explicit",
                    "--method",
                    "eagle",
                ]
            )
        self.assertEqual(explicit.draft_model, "/models/explicit")
        resolver.assert_not_called()

    def test_adaptive_defaults_online_and_can_be_disabled(self) -> None:
        default, _ = parse_args(
            ["--target-model", "/models/target", "--draft-model", "/models/draft"]
        )
        self.assertTrue(default.adaptive_speculation)
        self.assertEqual(default.adaptive_policy, "online")
        self.assertEqual(default.gamma, 4)
        self.assertEqual(default.adaptive_min_gamma, 1)
        self.assertTrue(default.adaptive_full_graph)
        self.assertTrue(default.adaptive_async)

        disabled, _ = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--no-adaptive-speculation",
            ]
        )
        self.assertFalse(disabled.adaptive_speculation)

    def test_dflash_keeps_adaptive_disabled_unless_explicitly_requested(self) -> None:
        options, _ = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/dflash",
                "--method",
                "dflash",
            ]
        )
        self.assertFalse(options.adaptive_speculation)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target-model",
                        "/models/target",
                        "--draft-model",
                        "/models/dflash",
                        "--method",
                        "dflash",
                        "--adaptive-speculation",
                    ]
                )

    def test_draft_active_vocab_environment(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--draft-active-vocab-size",
                "64000",
                "--draft-lm-head-quantization",
                "w8a16",
                "--draft-target-active-vocab-ids",
                "/tmp/target-active.json",
            ]
        )
        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_DRAFT_ACTIVE_VOCAB], "1")
        self.assertEqual(
            environment["VLLM_ASCEND_DRAFT_ACTIVE_VOCAB_SIZE"],
            "64000",
        )
        self.assertEqual(environment["VLLM_ASCEND_DRAFT_LM_HEAD_W8A16"], "1")
        self.assertEqual(environment[ENV_DRAFT_TARGET_ACTIVE_VOCAB], "1")
        self.assertEqual(
            environment["VLLM_ASCEND_DRAFT_TARGET_ACTIVE_VOCAB_IDS_PATH"],
            "/tmp/target-active.json",
        )

    def test_eagle_command_and_optimized_environment(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/qwen25",
                "--draft-model",
                "/models/eagle",
                "--method",
                "eagle",
                "--gamma",
                "3",
                "--graph-mode",
                "full",
                "--eagle-relaxed-accept-topk",
                "3",
                "--eagle-relaxed-accept-after-tokens",
                "64",
                "--eagle-spec-metadata-cache",
                "--eagle-uniform-state-kernel",
                "--graph-event-ordering",
                "--eagle-zero-draft-kv-first-step",
                "--eagle-draft-trace",
                "--eagle-draft-io-trace-dir",
                "/tmp/eagle-draft-io",
                "--eagle-target-hidden-trace-dir",
                "/tmp/eagle-target-hidden",
                "--eagle-target-argmax-trace-path",
                "/tmp/eagle-target-argmax.txt",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )
        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertIn('"method":"eagle"', speculative)
        self.assertNotIn("use_heterogeneous_vocab", speculative)
        compilation = command[command.index("--compilation-config") + 1]
        self.assertIn('"cudagraph_mode":"FULL"', compilation)

        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_METHOD], "eagle")
        self.assertEqual(environment[ENV_ASSUME_SHARED_TOKENIZER], "0")
        self.assertEqual(environment[ENV_EAGLE_RELAXED_ACCEPT_TOPK], "3")
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_RELAXED_ACCEPT_AFTER_TOKENS"],
            "64",
        )
        self.assertEqual(environment["VLLM_ASCEND_EAGLE_SPEC_METADATA_CACHE"], "1")
        self.assertEqual(environment["VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL"], "1")
        self.assertEqual(environment["VLLM_ASCEND_GRAPH_EVENT_ORDERING"], "1")
        self.assertEqual(environment["VLLM_ASCEND_EAGLE_ZERO_DRAFT_KV_FIRST_STEP"], "1")
        self.assertEqual(environment["VLLM_ASCEND_EAGLE_DRAFT_TRACE"], "1")
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_DRAFT_IO_TRACE_DIR"],
            "/tmp/eagle-draft-io",
        )
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_TARGET_HIDDEN_TRACE_DIR"],
            "/tmp/eagle-target-hidden",
        )
        self.assertEqual(
            environment["VLLM_ASCEND_EAGLE_TARGET_ARGMAX_TRACE_PATH"],
            "/tmp/eagle-target-argmax.txt",
        )

    def test_dflash_command_uses_native_parallel_method(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/qwen3",
                "--draft-model",
                "/models/dflash",
                "--method",
                "dflash",
                "--gamma",
                "4",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )
        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertIn('"method":"dflash"', speculative)
        self.assertNotIn("use_heterogeneous_vocab", speculative)
        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_METHOD], "dflash")
        self.assertEqual(environment[ENV_ASSUME_SHARED_TOKENIZER], "0")

    def test_eagle_options_rejected_for_eagle3(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target-model",
                        "/models/qwen3",
                        "--draft-model",
                        "/models/eagle3",
                        "--method",
                        "eagle3",
                        "--eagle-relaxed-accept-topk",
                        "3",
                    ]
                )

    def test_strict_eagle3_does_not_export_experimental_flags(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/qwen3",
                "--draft-model",
                "/models/eagle3",
                "--method",
                "eagle3",
            ]
        )
        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_METHOD], "eagle3")
        self.assertFalse(any(name.startswith("VLLM_ASCEND_EAGLE") for name in environment))

    def test_strict_eagle3_clears_stale_eagle_environment(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/qwen3",
                "--draft-model",
                "/models/eagle3",
                "--method",
                "eagle3",
            ]
        )
        stale_environment = {
            "VLLM_ASCEND_EAGLE_DRAFT_TRACE": "1",
            "VLLM_ASCEND_EAGLE_ZERO_DRAFT_KV_FIRST_STEP": "1",
            "VLLM_ASCEND_EAGLE_TARGET_ARGMAX_TRACE_PATH": "/tmp/stale",
        }
        environment = build_environment(options, configured_environment, stale_environment)
        self.assertFalse(any(name.startswith("VLLM_ASCEND_EAGLE") for name in environment))

    def test_plugin_settings_reject_invalid_environment(self) -> None:
        with self.assertRaisesRegex(ValueError, ENV_ENABLED):
            PluginSettings.from_environment({ENV_ENABLED: "sometimes"})

        with self.assertRaisesRegex(ValueError, ENV_MAX_NUM_SEQS):
            PluginSettings.from_environment({ENV_MAX_NUM_SEQS: "0"})

        with self.assertRaisesRegex(ValueError, ENV_METHOD):
            PluginSettings.from_environment({ENV_METHOD: "unknown"})

        with self.assertRaisesRegex(ValueError, ENV_ADAPTIVE_REFILL_BATCH):
            PluginSettings.from_environment({ENV_ADAPTIVE_REFILL_BATCH: "-1"})

        with self.assertRaisesRegex(ValueError, ENV_CONFIDENCE_ACCEPT_MARGIN):
            PluginSettings(confidence_accept_margin=-0.1)

    def test_performance_tuning_settings_round_trip_through_environment(self) -> None:
        settings = PluginSettings(
            confidence_accept_margin=5.25,
            confidence_accept_from_position=1,
            confidence_accept_after_tokens=16,
            confidence_protected_token_ids=(1, 2),
            adaptive_refill_batch=8,
        )

        environment = settings.as_environment()
        self.assertEqual(environment[ENV_CONFIDENCE_ACCEPT_MARGIN], "5.25")
        self.assertEqual(environment[ENV_CONFIDENCE_ACCEPT_FROM_POSITION], "1")
        self.assertEqual(environment[ENV_CONFIDENCE_ACCEPT_AFTER_TOKENS], "16")
        self.assertEqual(environment[ENV_CONFIDENCE_PROTECTED_TOKEN_IDS], "1,2")
        self.assertEqual(environment[ENV_ADAPTIVE_REFILL_BATCH], "8")
        self.assertEqual(PluginSettings.from_environment(environment), settings)

    def test_draft_cli_exports_performance_tuning_settings(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--method",
                "draft",
                "--gamma",
                "4",
                "--adaptive-speculation",
                "--adaptive-policy",
                "online",
                "--adaptive-refill-batch",
                "8",
                "--confidence-accept-margin",
                "5.25",
            ]
        )

        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_ADAPTIVE_REFILL_BATCH], "8")
        self.assertEqual(environment[ENV_CONFIDENCE_ACCEPT_MARGIN], "5.25")

    def test_strict_cli_clears_stale_confidence_margin(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--method",
                "draft",
            ]
        )

        environment = build_environment(
            options,
            configured_environment,
            {ENV_CONFIDENCE_ACCEPT_MARGIN: "5.25"},
        )
        self.assertEqual(environment[ENV_CONFIDENCE_ACCEPT_MARGIN], "")

    def test_legacy_eagle_confidence_margin_is_supported(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VSPEC_EAGLE_CONFIDENCE_ACCEPT_MARGIN": "4.5"},
            clear=True,
        ):
            self.assertEqual(_confidence_accept_margin(), 4.5)

    def test_legacy_specascend_environment_is_supported(self) -> None:
        settings = PluginSettings.from_environment(
            {
                "HUST_SPECASCEND_ENABLED": "1",
                "HUST_SPECASCEND_METHOD": "eagle",
                "HUST_SPECASCEND_MAX_NUM_SEQS": "32",
            }
        )
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.method, "eagle")
        self.assertEqual(settings.max_num_seqs, 32)

        preferred = PluginSettings.from_environment(
            {
                "HUST_SPECASCEND_METHOD": "draft",
                ENV_METHOD: "eagle3",
            }
        )
        self.assertEqual(preferred.method, "eagle3")

    def test_legacy_specslo_environment_is_supported(self) -> None:
        settings = PluginSettings.from_environment(
            {
                "HUST_SPECSLO_ENABLED": "1",
                "HUST_SPECSLO_METHOD": "eagle",
                "HUST_SPECSLO_MAX_NUM_SEQS": "64",
            }
        )
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.method, "eagle")
        self.assertEqual(settings.max_num_seqs, 64)

        preferred = PluginSettings.from_environment(
            {
                "HUST_SPECSLO_METHOD": "draft",
                "HUST_SPECASCEND_METHOD": "eagle",
                ENV_METHOD: "eagle3",
            }
        )
        self.assertEqual(preferred.method, "eagle3")


if __name__ == "__main__":
    unittest.main()
