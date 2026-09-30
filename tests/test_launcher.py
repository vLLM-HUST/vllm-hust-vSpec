from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_hust_vspec.backends.draft import (
    _add_draft_continuation_graph_keys,
    _bind_host_attn_update,
    _call_host_query_padding,
    _draft_request_buckets,
    _enable_merged_graph_replay,
    _install_inner_logits_processor_alias,
    _select_draft_continuation_graph,
    _uniform_descriptor_request_count,
)
from vllm_hust_vspec.backends.draft_parallel_update import (
    _chunk_descriptors,
    _DenseFIAParam,
    _graph_plan_scope,
    _normalize_dense_fia_param,
    _UpdateDescriptor,
)
from vllm_hust_vspec.backends.draft_repetition import (
    _collect_unique_history,
    _row_layout,
    select_exact_repetition_topk,
)
from vllm_hust_vspec.backends.draft_vocab import (
    _configure_serial_draft_active_vocab,
)
from vllm_hust_vspec.backends.eagle_body_quant import _WeightOnlyLinearMethod
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
    _confidence_accept_selected_inputs,
    _forward_greedy_token_ids,
    _forward_linear_eagle,
    _has_active_non_argmax_processor,
    _sparse_repetition_greedy,
)
from vllm_hust_vspec.backends.eagle_target import (
    _configure_target_active_vocab,
    _configure_target_active_vocab_for_method,
)
from vllm_hust_vspec.backends.mtp import (
    _annotate_qwen35_mtp_kv_groups,
    _get_current_mamba_groups,
    _is_uniform_decode_fallback,
    _missing_ascend_mtp_ops,
    _normalize_mamba_state_copy_funcs,
    _sample_mtp_local_argmax,
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
    ENV_DRAFT_EXACT_REPETITION_SYNC_PROOF,
    ENV_DRAFT_EXACT_REPETITION_TOPK,
    ENV_DRAFT_EXACT_REPETITION_TRACE,
    ENV_DRAFT_PARALLEL_GRAPH_UPDATES,
    ENV_DRAFT_TARGET_ACTIVE_VOCAB,
    ENV_DRAFT_TARGET_PARALLEL_GRAPH_UPDATES,
    ENV_EAGLE_RELAXED_ACCEPT_TOPK,
    ENV_ENABLED,
    ENV_MAX_NUM_SEQS,
    ENV_METHOD,
    ENV_MTP_STRICT_GRAPH,
    PluginSettings,
)


class LauncherTest(unittest.TestCase):
    def test_weight_only_linear_aligns_bias_with_input_dtype(self) -> None:
        captured: dict[str, torch.Tensor | None] = {}

        def weight_quant_matmul(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
            *,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            captured["bias"] = bias
            return x

        layer = SimpleNamespace(
            _vspec_w8a16_weight=torch.empty((64, 64), dtype=torch.int8),
            _vspec_w8a16_scale=torch.ones(64, dtype=torch.float16),
            _vspec_w8a16_bias=torch.ones(64, dtype=torch.float32),
        )
        torch_npu = SimpleNamespace(
            npu_weight_quant_batchmatmul=weight_quant_matmul,
        )
        x = torch.ones((1, 64), dtype=torch.float16)

        with mock.patch.dict(sys.modules, {"torch_npu": torch_npu}):
            output = _WeightOnlyLinearMethod().apply(
                layer,
                x,
                bias=layer._vspec_w8a16_bias,
            )

        self.assertIs(output, x)
        self.assertEqual(captured["bias"].dtype, torch.float16)

    def test_draft_parallel_update_normalizes_current_dense_fia_abi(
        self,
    ) -> None:
        captured = tuple(range(21)) + ("model.layers.7.self_attn.attn",)

        normalized = _normalize_dense_fia_param(captured)

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized.layer_name, captured[21])
        self.assertEqual(normalized.sliding_window, captured[16])
        self.assertEqual(normalized.c8_k_scale, captured[17])
        self.assertEqual(normalized.c8_v_scale, captured[19])

    def test_draft_parallel_update_chunks_static_descriptors(self) -> None:
        param = _DenseFIAParam(*range(16), "layer", None)
        descriptors = [_UpdateDescriptor(param, index, index, 0, "layer", 3) for index in range(7)]

        chunks = _chunk_descriptors(descriptors, 3)

        self.assertEqual(
            [[descriptor.handle for descriptor in chunk] for chunk in chunks],
            [[0, 3, 6], [1, 4], [2, 5]],
        )

    def test_draft_parallel_update_plan_scope_isolates_graphs(self) -> None:
        first_table = object()
        second_table = object()

        first = _graph_plan_scope("draft", first_table, [1, 2])
        same = _graph_plan_scope("draft", first_table, [3, 4])
        grown = _graph_plan_scope("draft", first_table, [1, 2, 3])
        other = _graph_plan_scope("draft", second_table, [1, 2])

        self.assertEqual(first, same)
        self.assertNotEqual(first, grown)
        self.assertNotEqual(first, other)

    def test_inactive_builtin_logits_processor_does_not_block_fast_path(self) -> None:
        inactive = SimpleNamespace(biases={}, min_toks={})
        metadata = SimpleNamespace(logitsprocs=SimpleNamespace(non_argmax_invariant=[inactive]))
        self.assertFalse(_has_active_non_argmax_processor(metadata))

        active = SimpleNamespace(biases={"request": {1: 2.0}}, min_toks={})
        metadata.logitsprocs.non_argmax_invariant = [active]
        self.assertTrue(_has_active_non_argmax_processor(metadata))

        metadata.logitsprocs.non_argmax_invariant = [object()]
        self.assertTrue(_has_active_non_argmax_processor(metadata))

    def test_draft_repetition_history_is_unique_and_bounded(self) -> None:
        rows, truncated = _collect_unique_history(
            prompt_token_ids=torch.tensor(
                [[1, 2, 1, 99], [4, 5, 0, 0]],
                dtype=torch.int32,
            ),
            prompt_lengths=[3, 2],
            output_token_ids=[[3, 2, -1], [5, 6, 7]],
            num_rows=2,
            vocab_size=8,
            history_width=3,
        )

        self.assertEqual(rows, [[1, 2, 3], [5, 6, 7]])
        self.assertTrue(truncated)

    def test_fused_repetition_row_layout_matches_target_and_bonus_rows(self) -> None:
        repeat_indices, local_positions = _row_layout([2, 1])

        self.assertEqual(repeat_indices, [0, 0, 1, 0, 1])
        self.assertEqual(local_positions, [0, 1, 0, 2, 1])

    def test_exact_repetition_topk_proves_global_winner(self) -> None:
        winner_ids, proven = select_exact_repetition_topk(
            torch.tensor([[10.0, 9.0, 8.0], [10.0, 9.0, 8.0]]),
            torch.tensor([[4, 2, 1], [4, 2, 1]]),
            torch.tensor(
                [[True, False, False], [True, True, True]],
            ),
            torch.tensor([2.0, 2.0]),
            vocab_size=6,
            has_outside_candidates=True,
        )

        self.assertEqual(winner_ids.tolist(), [2, 4])
        self.assertEqual(proven.tolist(), [True, False])

    def test_proven_repetition_topk_matches_full_vocab_argmax(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            logits = torch.randn((8, 97), generator=generator)
            seen = torch.rand((8, 97), generator=generator) < 0.35
            penalties = 1.0 + torch.rand((8,), generator=generator)
            full_values = torch.where(
                seen,
                torch.where(
                    logits > 0,
                    logits / penalties.unsqueeze(-1),
                    logits * penalties.unsqueeze(-1),
                ),
                logits,
            )
            expected = full_values.argmax(dim=-1)
            values, ids = logits.topk(16, dim=-1, sorted=True)
            candidate_seen = seen.gather(1, ids)
            actual, proven = select_exact_repetition_topk(
                values,
                ids,
                candidate_seen,
                penalties,
                vocab_size=logits.shape[-1],
                has_outside_candidates=True,
            )
            self.assertTrue(torch.equal(actual[proven], expected[proven]))

    def test_draft_continuation_graph_keys_cover_gamma_plus_two(self) -> None:
        graph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.FULL: set(),
        }
        dispatcher = SimpleNamespace(
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(method="draft_model"),
                scheduler_config=SimpleNamespace(max_num_seqs=16),
            ),
            cudagraph_keys=graph_keys,
            _get_lora_cases=lambda: [0],
            add_cudagraph_key=lambda mode, descriptor: graph_keys[mode].add(descriptor),
        )

        added = _add_draft_continuation_graph_keys(
            dispatcher,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            6,
        )

        self.assertEqual(_draft_request_buckets(16), (1, 2, 4, 8, 16))
        self.assertEqual(added, 5)
        self.assertEqual(
            sorted(descriptor.num_tokens for descriptor in graph_keys[CUDAGraphMode.FULL]),
            [7, 14, 28, 56, 112],
        )

    def test_draft_request_buckets_include_configured_sizes(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VSPEC_DRAFT_REQUEST_BUCKETS": "3, 9,15"},
        ):
            self.assertEqual(
                _draft_request_buckets(16),
                (1, 2, 3, 4, 8, 9, 15, 16),
            )

    def test_draft_request_buckets_reject_out_of_range_sizes(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VSPEC_DRAFT_REQUEST_BUCKETS": "17"},
        ):
            with self.assertRaisesRegex(ValueError, r"\[1, 16\]"):
                _draft_request_buckets(16)

    def test_adaptive_draft_continuation_graph_keys_cover_each_gamma(self) -> None:
        graph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.FULL: set(),
        }
        dispatcher = SimpleNamespace(
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(method="draft_model"),
                scheduler_config=SimpleNamespace(max_num_seqs=16),
            ),
            cudagraph_keys=graph_keys,
            _get_lora_cases=lambda: [0],
            add_cudagraph_key=lambda mode, descriptor: graph_keys[mode].add(descriptor),
        )

        added = _add_draft_continuation_graph_keys(
            dispatcher,
            CUDAGraphMode.FULL_DECODE_ONLY,
            5,
            query_lens=(3, 4, 5, 6),
        )

        self.assertEqual(added, 20)
        self.assertEqual(
            dispatcher._vspec_draft_merged_query_lens,
            (3, 4, 5, 6),
        )
        self.assertEqual(
            sorted(
                (descriptor.num_reqs, descriptor.num_tokens)
                for descriptor in graph_keys[CUDAGraphMode.FULL]
            ),
            sorted(
                (num_reqs, num_reqs * query_len)
                for query_len in (3, 4, 5, 6)
                for num_reqs in (1, 2, 4, 8, 16)
            ),
        )

    def test_draft_continuation_dispatch_uses_request_bucket(self) -> None:
        descriptors = {
            BatchDescriptor(
                num_tokens=num_reqs * 7,
                num_reqs=num_reqs,
                uniform=True,
            )
            for num_reqs in (1, 2, 4, 8, 16)
        }
        dispatcher = SimpleNamespace(
            _vspec_active_num_reqs=10,
            _vspec_draft_merged_query_lens=(7,),
            _vspec_draft_continuation_query_len=7,
            uniform_decode_query_len=6,
            cudagraph_keys={CUDAGraphMode.FULL: descriptors},
        )

        descriptor = _select_draft_continuation_graph(
            dispatcher,
            70,
            True,
            False,
            0,
            None,
            None,
        )

        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        self.assertEqual(descriptor.num_tokens, 112)
        self.assertEqual(descriptor.num_reqs, 16)

    def test_compact_request_count_uses_runtime_descriptor(self) -> None:
        descriptor = BatchDescriptor(
            num_tokens=4,
            num_reqs=1,
            uniform=True,
        )

        self.assertEqual(
            _uniform_descriptor_request_count(descriptor, 4, 4),
            1,
        )
        self.assertIsNone(_uniform_descriptor_request_count(descriptor, 8, 4))

    def test_merged_draft_reuses_host_graph_without_replay_barrier(self) -> None:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        wrapper = ACLGraphWrapper.__new__(ACLGraphWrapper)
        wrapper.use_eagle = False

        self.assertTrue(_enable_merged_graph_replay(wrapper, True))
        self.assertTrue(wrapper.use_eagle)

    def test_disabled_merged_draft_keeps_host_graph_barrier(self) -> None:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        wrapper = ACLGraphWrapper.__new__(ACLGraphWrapper)
        wrapper.use_eagle = False

        self.assertFalse(_enable_merged_graph_replay(wrapper, False))
        self.assertFalse(wrapper.use_eagle)

    def test_draft_body_quantization_cli_environment(self) -> None:
        options, configured = parse_args(
            [
                "--target-model",
                "/target",
                "--draft-model",
                "/draft",
                "--method",
                "draft",
                "--draft-body-quantization",
                "w8a16",
            ]
        )

        environment = build_environment(options, configured, {})

        self.assertEqual(environment["VSPEC_DRAFT_BODY_W8A16"], "1")

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

    def test_penalized_greedy_verification_uses_processed_token_ids(self) -> None:
        @dataclass
        class Sampling:
            all_greedy: bool = True
            max_num_logprobs: int | None = None
            logprob_token_ids: list[int] | None = None
            no_penalties: bool = False
            output_token_ids: list[list[int]] | None = None

        metadata = SimpleNamespace(
            max_spec_len=1,
            target_logits_indices=torch.tensor([0]),
            bonus_logits_indices=torch.tensor([1]),
        )
        sampling_metadata = Sampling(output_token_ids=[[1]])
        logits = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
        bonus_output = SimpleNamespace(sampled_token_ids=torch.tensor([[1]]))
        sampler = SimpleNamespace(
            sampler=mock.Mock(return_value=bonus_output),
            apply_logits_processors=mock.Mock(return_value=torch.tensor([[1.0, 5.0]])),
        )
        sentinel = object()
        with mock.patch(
            "vllm_hust_vspec.backends.eagle_rejection._forward_greedy_token_ids",
            return_value=sentinel,
        ) as fast_path:
            result = _forward_linear_eagle(
                sampler,
                metadata,
                None,
                logits,
                sampling_metadata,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(fast_path.call_args.args[2].tolist(), [1])
        self.assertTrue(fast_path.call_args.kwargs["target_rows_selected"])
        self.assertEqual(
            fast_path.call_args.kwargs["bonus_token_ids"].tolist(),
            [[1]],
        )
        self.assertIsNone(sampler.sampler.call_args.kwargs["sampling_metadata"].max_num_logprobs)

    def test_sparse_repetition_greedy_penalizes_seen_candidates(self) -> None:
        metadata = SimpleNamespace(
            num_draft_tokens=[2],
            cu_num_draft_tokens=torch.tensor([2]),
            target_logits_indices=torch.tensor([0, 1]),
            bonus_logits_indices=torch.tensor([2]),
        )
        sampling_metadata = SimpleNamespace(
            _vspec_repetition_only=True,
            prompt_token_ids=torch.tensor([[0, 4]]),
            repetition_penalties=torch.tensor([2.0]),
            output_token_ids=[[3]],
            spec_token_ids=[[1, 2]],
            allowed_token_ids_mask=None,
            bad_words_token_ids=None,
            logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
            thinking_budget_state_holder=None,
        )
        logits = torch.tensor(
            [
                [10.0, 9.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 10.0, 9.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 10.0, 0.0, 0.0, 9.0],
            ]
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "HUST_VSPEC_METHOD": "draft_model",
                    "VSPEC_DRAFT_SPARSE_REPETITION_TOPK": "3",
                },
            ),
            mock.patch(
                "vllm_ascend.sample.rejection_sampler.expand_batch_to_tokens",
                return_value=torch.tensor([0, 0]),
            ),
        ):
            result = _sparse_repetition_greedy(
                metadata,
                logits,
                sampling_metadata,
            )

        self.assertIsNotNone(result)
        assert result is not None
        target_ids, bonus_ids = result
        self.assertEqual(target_ids.tolist(), [1, 2])
        self.assertEqual(bonus_ids.tolist(), [[5]])

    def test_serial_draft_active_vocab_maps_full_token_ids(self) -> None:
        class TinyDraftModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 1200, bias=False)

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

        compact_logits = model.compute_logits(torch.ones(2, 4))
        self.assertIsInstance(compact_logits, ActiveVocabLogits)
        self.assertEqual(compact_logits.shape, (2, 1024))
        self.assertEqual(compact_logits.argmax(dim=-1).tolist(), [1100, 1100])
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
        for method in ("draft", "eagle", "eagle3", "dflash", "mtp"):
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

    def test_eagle3_draft_lm_head_w8a16_preserves_trained_vocab_mapping(self) -> None:
        class TinyLMHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(64, 64))
                self.register_parameter("bias", None)
                self.quant_method = object()

        model = SimpleNamespace(
            lm_head=TinyLMHead(),
            draft_id_to_target_id=torch.arange(64),
        )
        proposer = SimpleNamespace(
            method="eagle3",
            speculative_config=SimpleNamespace(draft_tensor_parallel_size=1),
            vllm_config=SimpleNamespace(quant_config=None),
            model=model,
        )
        environment = {"VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_W8A16": "1"}
        with mock.patch.dict(os.environ, environment, clear=False):
            _configure_draft_active_vocab(proposer)

        self.assertIsInstance(model.lm_head.quant_method, _WeightOnlyLinearMethod)
        self.assertEqual(model.lm_head._vspec_w8a16_weight.shape, (64, 64))
        self.assertEqual(model.lm_head._vspec_w8a16_scale.shape, (64,))
        self.assertTrue(torch.equal(model.draft_id_to_target_id, torch.arange(64)))

    def test_eagle3_draft_lm_head_w8a16_environment(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/eagle3",
                "--method",
                "eagle3",
                "--eagle-draft-lm-head-quantization",
                "w8a16",
            ]
        )
        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment["VLLM_ASCEND_EAGLE_DRAFT_LM_HEAD_W8A16"], "1")

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

    def test_eagle3_auto_target_active_vocab_partitions_tp_shard(self) -> None:
        class TinyShardedHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(600, 4))
                self.bias = None
                self.org_vocab_size = 1200
                self.shard_indices = SimpleNamespace(
                    org_vocab_start_index=0,
                    org_vocab_end_index=600,
                )

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = SimpleNamespace(lm_head=TinyShardedHead())

            def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.linear(
                    hidden_states,
                    self.language_model.lm_head.weight,
                )

        runner = SimpleNamespace(
            speculative_config=SimpleNamespace(method="eagle3"),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            vllm_config=SimpleNamespace(quant_config=None),
            model=TinyModel(),
            drafter=SimpleNamespace(
                model=SimpleNamespace(
                    draft_id_to_target_id=torch.zeros(1024, dtype=torch.long),
                ),
            ),
            max_num_tokens=16,
        )
        environment_name = "VLLM_ASCEND_EAGLE_TARGET_ACTIVE_VOCAB_IDS_PATH"
        with mock.patch.dict(os.environ, {environment_name: "auto"}, clear=False):
            _configure_target_active_vocab_for_method(
                runner,
                active_ids_environment=environment_name,
                required_method="eagle3",
                feature_name="EAGLE3 Target",
            )

        logits = runner.model.compute_logits(torch.ones(2, 4))
        self.assertEqual(logits.shape, (2, 600))
        self.assertEqual(runner._eagle_target_active_vocab_ids.numel(), 1024)
        self.assertEqual(runner._eagle_target_local_active_vocab_ids.numel(), 600)
        self.assertEqual(runner._eagle_target_active_vocab_tp_size, 2)

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

    def test_selected_confidence_accept_compares_processed_rows(self) -> None:
        metadata = SimpleNamespace(
            draft_token_ids=torch.tensor([1, 0], dtype=torch.int32),
            num_draft_tokens=[2],
        )
        target_logits = torch.tensor(
            [
                [3.0, 2.7, 0.0],
                [2.0, 4.0, 0.0],
            ]
        )

        target_ids, relaxed_mask = _confidence_accept_selected_inputs(
            target_logits,
            metadata,
            0.5,
        )

        self.assertEqual(target_ids.tolist(), [0, 1])
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

    def test_eagle3_exact_capture_sizes_are_dense_at_b16(self) -> None:
        sizes = generate_capture_sizes(16, 5, "eagle3", "exact")
        self.assertEqual(
            sizes,
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                18,
                24,
                30,
                36,
                42,
                48,
                54,
                60,
                66,
                72,
                78,
                84,
                90,
                96,
            ],
        )

    def test_adaptive_mtp2_capture_sizes_cover_every_b16_tail(self) -> None:
        sizes = generate_capture_sizes(
            16,
            6,
            "mtp",
            "exact",
            dynamic_widths=True,
        )

        for query_width in (3, 5, 7):
            for batch_size in range(1, 17):
                self.assertIn(batch_size * query_width, sizes)
        self.assertIn(45, sizes)
        self.assertIn(60, sizes)

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
            "qwen25-14b-05b-arc-easy.toml": "draft_model",
            "qwen25-14b-eagle.toml": "eagle",
            "qwen25-14b-eagle-arc-easy.toml": "eagle",
            "qwen25-14b-eagle-relaxed.toml": "eagle",
            "qwen3-8b-eagle3.toml": "eagle3",
            "qwen35-35b-a3b-mtp2.toml": "mtp",
            "qwen35-35b-a3b-frontier-mtp2.toml": "mtp",
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
                expected_generation_config = "auto" if "arc-easy" in filename else "vllm"
                self.assertEqual(generation_config, expected_generation_config)
                if filename == "qwen35-35b-a3b-mtp2.toml":
                    self.assertIn("--enable-expert-parallel", command)
                    self.assertIn("multistream_overlap_shared_expert", command[-1])
                if filename == "qwen35-35b-a3b-frontier-mtp2.toml":
                    self.assertIn("--enable-expert-parallel", command)
                    self.assertNotIn("multistream_overlap_shared_expert", command[-1])
                    self.assertIn("--enable-prefix-caching", command)
                    self.assertIn("--async-scheduling", command)
                    self.assertIn("--language-model-only", command)

    def test_qwen35_frontier_protocol_is_fixed_mtp2_contract(self) -> None:
        options, _ = parse_args(
            [
                "--protocol",
                "qwen35-frontier-mtp2",
                "--target-model",
                "/models/Qwen3.5-35B-A3B",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )

        self.assertEqual(options.method, "mtp")
        self.assertEqual(options.gamma, 2)
        self.assertEqual(options.tensor_parallel_size, 2)
        self.assertEqual(options.max_model_len, 262144)
        self.assertEqual(options.graph_mode, "full-and-piecewise")
        self.assertTrue(options.prefix_caching)
        self.assertTrue(options.async_scheduling)
        self.assertTrue(options.chunked_prefill)
        self.assertTrue(options.language_model_only)
        self.assertTrue(options.mtp_strict_graph)
        self.assertFalse(options.mtp_local_argmax_reduction)
        self.assertFalse(options.adaptive_speculation)

        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertEqual(
            speculative,
            '{"method":"mtp","num_speculative_tokens":2,'
            '"enforce_eager":false,"use_local_argmax_reduction":false}',
        )
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--max-model-len") + 1], "262144")
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--async-scheduling", command)
        self.assertIn("--language-model-only", command)
        self.assertIn("--enable-expert-parallel", command)
        compilation = command[command.index("--compilation-config") + 1]
        self.assertEqual(
            compilation,
            '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}',
        )
        capture_start = command.index("--cudagraph-capture-sizes") + 1
        capture_end = command.index("--enable-expert-parallel")
        self.assertEqual(
            [int(value) for value in command[capture_start:capture_end]],
            [3, 6, 9, 12, 18, 24, 48],
        )

    def test_qwen35_frontier_b16_performance_preset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        options, environment = parse_args(
            [
                "--config",
                str(root / "configs/qwen35-35b-a3b-frontier-mtp2-b16.toml"),
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )

        self.assertEqual(options.method, "mtp")
        self.assertEqual(options.gamma, 4)
        self.assertEqual(options.max_num_seqs, 16)
        self.assertEqual(environment["HUST_VSPEC_MTP_COHORT_REFILL"], "1")
        self.assertEqual(environment["HUST_VSPEC_REFILL_MAX_HOLDS"], "8")

        command = build_vllm_command(options)
        capture_start = command.index("--cudagraph-capture-sizes") + 1
        capture_end = command.index("--enable-expert-parallel")
        self.assertEqual(
            [int(value) for value in command[capture_start:capture_end]],
            [5, 10, 20, 40, 80],
        )
        additional_config = command[command.index("--additional-config") + 1]
        self.assertEqual(additional_config, '{"enable_cpu_binding":true}')

    def test_qwen35_frontier_mtp2_adaptive_uses_even_gamma_candidates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        options, environment = parse_args(
            [
                "--config",
                str(root / "configs/qwen35-35b-a3b-frontier-mtp2-adaptive.toml"),
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )

        self.assertEqual(options.method, "mtp")
        self.assertEqual(options.gamma, 6)
        self.assertEqual(options.adaptive_min_gamma, 2)
        self.assertEqual(options.adaptive_max_gamma_step, 2)
        self.assertTrue(options.adaptive_speculation)
        self.assertEqual(environment["HUST_VSPEC_MTP_ASYNC_RUNTIME_SWITCH"], "0")

        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertIn('"num_speculative_tokens":6', speculative)
        capture_start = command.index("--cudagraph-capture-sizes") + 1
        capture_end = command.index("--enable-expert-parallel")
        capture_sizes = [int(value) for value in command[capture_start:capture_end]]
        self.assertIn(48, capture_sizes)
        self.assertIn(80, capture_sizes)
        self.assertIn(112, capture_sizes)

    def test_qwen35_eagle3_adaptive_captures_every_verification_width(self) -> None:
        root = Path(__file__).resolve().parents[1]
        options, _ = parse_args(
            [
                "--config",
                str(root / "configs/qwen35-35b-a3b-eagle3-adaptive.toml"),
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )

        self.assertEqual(options.method, "eagle3")
        self.assertEqual(options.gamma, 6)
        self.assertEqual(options.adaptive_min_gamma, 1)
        self.assertTrue(options.adaptive_speculation)

        command = build_vllm_command(options)
        capture_start = command.index("--cudagraph-capture-sizes") + 1
        capture_end = command.index("--enable-expert-parallel")
        capture_sizes = [int(value) for value in command[capture_start:capture_end]]
        for query_width in range(2, 8):
            for batch_size in range(1, 17):
                self.assertIn(batch_size * query_width, capture_sizes)
        self.assertIn(112, capture_sizes)

    def test_adaptive_mtp_rejects_non_even_candidate_range(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target-model",
                        "/models/Qwen3.5-35B-A3B",
                        "--method",
                        "mtp",
                        "--gamma",
                        "5",
                        "--adaptive-min-gamma",
                        "2",
                        "--adaptive-max-gamma-step",
                        "2",
                        "--adaptive-speculation",
                    ]
                )

    def test_qwen35_frontier_protocol_rejects_contract_overrides(self) -> None:
        overrides = (
            ["--method", "draft"],
            ["--gamma", "3"],
            ["--tensor-parallel-size", "1"],
            ["--max-model-len", "4096"],
            ["--no-prefix-caching"],
            ["--no-async-scheduling"],
            ["--graph-mode", "full"],
        )
        for override in overrides:
            with self.subTest(override=override):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(
                            [
                                "--protocol",
                                "qwen35-frontier-mtp2",
                                "--target-model",
                                "/models/Qwen3.5-35B-A3B",
                                *override,
                            ]
                        )

    def test_qwen35_frontier_supports_agentx_forced_acceptance(self) -> None:
        options, _ = parse_args(
            [
                "--protocol",
                "qwen35-frontier-mtp2",
                "--target-model",
                "/models/Qwen3.5-35B-A3B",
                "--synthetic-acceptance-length",
                "2.4",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )

        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertEqual(
            speculative,
            '{"method":"mtp","num_speculative_tokens":2,'
            '"enforce_eager":false,"rejection_sample_method":"synthetic",'
            '"synthetic_acceptance_length":2.4,'
            '"use_local_argmax_reduction":false}',
        )

    def test_synthetic_acceptance_requires_valid_fixed_gamma(self) -> None:
        invalid_cases = (
            [
                "--method",
                "mtp",
                "--target-model",
                "/models/Qwen3.5-35B-A3B",
                "--gamma",
                "2",
                "--synthetic-acceptance-length",
                "3.1",
            ],
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--synthetic-acceptance-length",
                "2.0",
            ],
        )
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments)

    def test_arc_easy_protocol_matches_baseline_serving_contract(self) -> None:
        with mock.patch(
            "vllm_hust_vspec.cli.resolve_default_model",
            return_value=Path("/model/Qwen2.5-0.5B-Instruct"),
        ):
            options, _ = parse_args(
                [
                    "--protocol",
                    "arc-easy",
                    "--method",
                    "draft",
                    "--vllm-executable",
                    "/usr/local/python3.11.14/bin/vllm",
                ]
            )

        self.assertEqual(options.target_model, "/model/Qwen2.5-14B-Instruct")
        self.assertEqual(options.served_model_name, "qwen2.5-14b-draft-vspec-fp16")
        self.assertEqual(options.port, 18180)
        self.assertEqual(options.max_num_seqs, 16)
        self.assertEqual(options.max_num_batched_tokens, 8192)
        self.assertEqual(options.max_model_len, 32768)
        self.assertEqual(options.dtype, "float16")
        self.assertEqual(options.graph_mode, "full-decode-only")
        self.assertEqual(options.generation_config, "auto")
        self.assertFalse(options.async_scheduling)
        self.assertFalse(options.prefix_caching)
        self.assertTrue(options.chunked_prefill)
        self.assertTrue(options.adaptive_speculation)
        self.assertEqual(options.adaptive_min_gamma, 1)
        self.assertEqual(options.gamma, 4)

        command = build_vllm_command(options)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertIn("--no-async-scheduling", command)
        self.assertIn("--no-enforce-eager", command)
        self.assertNotIn("--disable-log-stats", command)
        compilation = command[command.index("--compilation-config") + 1]
        self.assertEqual(compilation, '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}')
        capture_index = command.index("--cudagraph-capture-sizes") + 1
        capture_sizes = [int(value) for value in command[capture_index:]]
        self.assertEqual(capture_sizes[-1], 80)

    def test_arc_easy_protocol_selects_eagle_served_name(self) -> None:
        with mock.patch(
            "vllm_hust_vspec.cli.resolve_default_model",
            return_value=Path("/model/Eagle-Qwen2.5-14B-Instruct"),
        ):
            options, _ = parse_args(["--protocol", "arc-easy", "--method", "eagle"])

        self.assertEqual(options.method, "eagle")
        self.assertEqual(options.served_model_name, "qwen2.5-14b-eagle-vspec-fp16")

    def test_full_and_piecewise_graph_mode_is_forwarded(self) -> None:
        options, _ = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--no-adaptive-speculation",
                "--graph-mode",
                "full-and-piecewise",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )

        command = build_vllm_command(options)
        compilation = command[command.index("--compilation-config") + 1]
        self.assertEqual(
            compilation,
            '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}',
        )

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
        self.assertEqual(environment["VLLM_USE_AOT_COMPILE"], "0")
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

    def test_mtp_uses_target_checkpoint_and_defaults_to_mtp2_graph(self) -> None:
        with mock.patch("vllm_hust_vspec.cli.resolve_default_model") as resolver:
            options, _ = parse_args(
                [
                    "--target-model",
                    "/models/Qwen3.5-35B-A3B",
                    "--method",
                    "qwen3_next_mtp",
                    "--language-model-only",
                    "--max-num-seqs",
                    "16",
                    "--vllm-executable",
                    "/usr/bin/vllm",
                ]
            )

        resolver.assert_not_called()
        self.assertEqual(options.method, "mtp")
        self.assertEqual(options.gamma, 2)
        self.assertIsNone(options.draft_model)
        self.assertFalse(options.adaptive_speculation)

        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertEqual(
            speculative,
            '{"method":"mtp","num_speculative_tokens":2,"enforce_eager":false,'
            '"use_local_argmax_reduction":false}',
        )
        self.assertIn("--language-model-only", command)
        self.assertIn("--no-enforce-eager", command)
        compilation = command[command.index("--compilation-config") + 1]
        self.assertEqual(compilation, '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}')
        capture_index = command.index("--cudagraph-capture-sizes") + 1
        self.assertEqual(
            [int(value) for value in command[capture_index:]],
            [3, 6, 9, 12, 18, 24, 48],
        )
        environment = build_environment(options, {}, {})
        self.assertEqual(environment[ENV_MTP_STRICT_GRAPH], "1")

    def test_mtp_local_argmax_can_be_enabled(self) -> None:
        options, _ = parse_args(
            [
                "--target-model",
                "/models/Qwen3.5-35B-A3B",
                "--method",
                "mtp",
                "--mtp-local-argmax-reduction",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )
        command = build_vllm_command(options)
        speculative = command[command.index("--speculative-config") + 1]
        self.assertIn('"use_local_argmax_reduction":true', speculative)

    def test_mtp_local_argmax_uses_model_reduction(self) -> None:
        expected = torch.tensor([3, 7], dtype=torch.int64)
        model = SimpleNamespace(get_top_tokens=mock.Mock(return_value=expected))
        proposer = SimpleNamespace(model=model)
        hidden_states = torch.randn(2, 4)

        token_ids, probabilities = _sample_mtp_local_argmax(
            proposer,
            hidden_states,
        )

        self.assertIs(token_ids, expected)
        self.assertIsNone(probabilities)
        model.get_top_tokens.assert_called_once_with(hidden_states)

    def test_mtp_local_argmax_requires_model_support(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "get_top_tokens"):
            _sample_mtp_local_argmax(
                SimpleNamespace(model=SimpleNamespace()),
                torch.randn(1, 4),
            )

    def test_qwen35_mtp_marks_only_group_containing_draft_attention(self) -> None:
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp"),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_5_moe"),
                hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            ),
        )
        groups = [
            SimpleNamespace(
                layer_names=["model.layers.0.linear_attn"],
                is_eagle_group=False,
            ),
            SimpleNamespace(
                layer_names=[
                    "model.layers.3.self_attn.attn",
                    "mtp.layers.0.self_attn.attn",
                ],
                is_eagle_group=False,
            ),
        ]

        annotated = _annotate_qwen35_mtp_kv_groups(
            config,
            {
                "model.layers.0.linear_attn": object(),
                "model.layers.3.self_attn.attn": object(),
                "mtp.layers.0.self_attn.attn": object(),
            },
            groups,
        )

        self.assertEqual(annotated, (1,))
        self.assertFalse(groups[0].is_eagle_group)
        self.assertTrue(groups[1].is_eagle_group)

    def test_qwen35_mtp_rejects_missing_draft_kv_layer(self) -> None:
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp"),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_5_moe"),
                hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "could not identify"):
            _annotate_qwen35_mtp_kv_groups(
                config,
                {"model.layers.3.self_attn.attn": object()},
                [],
            )

    def test_qwen35_mtp_normalizes_current_mamba_group_api(self) -> None:
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

        first_spec = MambaSpec(
            block_size=128,
            shapes=((4, 8),),
            dtypes=(torch.float32,),
        )
        second_spec = MambaSpec(
            block_size=128,
            shapes=((8, 8),),
            dtypes=(torch.float32,),
        )
        groups = _get_current_mamba_groups(
            SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["model.layers.0.linear_attn"],
                        kv_cache_spec=first_spec,
                    ),
                    SimpleNamespace(
                        layer_names=[
                            "model.layers.1.linear_attn",
                            "model.layers.2.linear_attn",
                        ],
                        kv_cache_spec=UniformTypeKVCacheSpecs(
                            block_size=128,
                            kv_cache_specs={
                                "model.layers.1.linear_attn": first_spec,
                                "model.layers.2.linear_attn": second_spec,
                            },
                        ),
                    ),
                ]
            )
        )

        self.assertEqual(groups[first_spec], [0, 1])
        self.assertEqual(groups[second_spec], [1])

        legacy_copy_funcs = (object(), object())
        normalized = _normalize_mamba_state_copy_funcs(
            SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["model.layers.0.linear_attn"],
                        kv_cache_spec=first_spec,
                    )
                ]
            ),
            legacy_copy_funcs,
        )
        self.assertEqual(normalized, {first_spec.mamba_type: legacy_copy_funcs})
        self.assertIs(
            _normalize_mamba_state_copy_funcs(
                SimpleNamespace(kv_cache_groups=[]),
                normalized,
            ),
            normalized,
        )

    def test_mtp_forwards_configured_kv_cache_dtype(self) -> None:
        options, _ = parse_args(
            [
                "--target-model",
                "/models/Qwen3.5-35B-A3B",
                "--method",
                "mtp",
                "--kv-cache-dtype",
                "fp8",
                "--vllm-executable",
                "/usr/bin/vllm",
            ]
        )
        command = build_vllm_command(options)
        self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "fp8")

    def test_mtp_method_aliases_normalize_to_native_method(self) -> None:
        for method in ("mtp", "qwen3_5_mtp", "qwen3_next_mtp"):
            with self.subTest(method=method):
                options, _ = parse_args(
                    [
                        "--target-model",
                        "/models/Qwen3.5-35B-A3B",
                        "--method",
                        method,
                    ]
                )
                self.assertEqual(options.method, "mtp")
                self.assertEqual(options.gamma, 2)

    def test_mtp_strict_graph_detects_only_uniform_decode_fallbacks(self) -> None:
        dispatcher = SimpleNamespace(
            keys_initialized=True,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        self.assertTrue(
            _is_uniform_decode_fallback(
                dispatcher,
                True,
                CUDAGraphMode.NONE,
            )
        )
        self.assertFalse(
            _is_uniform_decode_fallback(
                dispatcher,
                False,
                CUDAGraphMode.NONE,
            )
        )
        self.assertFalse(
            _is_uniform_decode_fallback(
                dispatcher,
                True,
                CUDAGraphMode.FULL,
            )
        )
        dispatcher.cudagraph_mode = CUDAGraphMode.NONE
        self.assertTrue(
            _is_uniform_decode_fallback(
                dispatcher,
                True,
                CUDAGraphMode.NONE,
            )
        )

    def test_mtp_native_op_validation_lists_missing_ops(self) -> None:
        namespace = SimpleNamespace(
            npu_gemma_rms_norm=object(),
            moe_gating_top_k=object(),
        )
        self.assertEqual(
            _missing_ascend_mtp_ops(namespace),
            ("npu_causal_conv1d_custom",),
        )

    def test_mtp_explicit_gamma_overrides_default_and_rejects_draft_model(self) -> None:
        options, _ = parse_args(
            [
                "--target-model",
                "/models/Qwen3.5-35B-A3B",
                "--method",
                "mtp",
                "--gamma",
                "3",
            ]
        )
        self.assertEqual(options.gamma, 3)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target-model",
                        "/models/Qwen3.5-35B-A3B",
                        "--method",
                        "mtp",
                        "--draft-model",
                        "/models/not-used",
                    ]
                )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target-model",
                        "/models/Qwen3.5-35B-A3B",
                        "--method",
                        "mtp",
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

    def test_draft_runtime_optimization_environment(self) -> None:
        options, configured_environment = parse_args(
            [
                "--target-model",
                "/models/target",
                "--draft-model",
                "/models/draft",
                "--draft-parallel-graph-updates",
                "4",
                "--draft-target-parallel-graph-updates",
                "3",
                "--draft-exact-repetition-topk",
                "64",
                "--draft-exact-repetition-trace",
                "--no-draft-exact-repetition-sync-proof",
            ]
        )
        environment = build_environment(options, configured_environment, {})
        self.assertEqual(environment[ENV_DRAFT_PARALLEL_GRAPH_UPDATES], "4")
        self.assertEqual(
            environment[ENV_DRAFT_TARGET_PARALLEL_GRAPH_UPDATES],
            "3",
        )
        self.assertEqual(environment[ENV_DRAFT_EXACT_REPETITION_TOPK], "64")
        self.assertEqual(environment[ENV_DRAFT_EXACT_REPETITION_TRACE], "1")
        self.assertEqual(
            environment[ENV_DRAFT_EXACT_REPETITION_SYNC_PROOF],
            "0",
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
