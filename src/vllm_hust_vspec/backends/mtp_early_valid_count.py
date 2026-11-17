"""Experimental, fail-closed early valid-count D2H for Qwen3.5 MTP.

This private prototype moves *the existing valid-count host copy* to just
after rejection sampling. It does not remove the CPU sequence-length wait.
It is opt-in, pinned to the exact Ascend source files, and restricted to
uniform pure decode. Never use its timings as a formal speed result without
independent token/count parity and adjacent native/candidate validation.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_RUNNER_SHA256 = "92610fb59add75e448f99c0e80421e9999168987e83eb2312c1c87f15eebbc72"
_PROPOSER_SHA256 = "51943285c4af8ab229e5b2db00334493cbcfd65278776f205c191ab757ccafcd"
_SAMPLER_SHA256 = "d62875a1c942255586b9bdd85abb820153be4731fa99406ebb0ab25b0d5c816b"
_DEVICE_OP_SHA256 = "c64dc4f693da5c537315fb66c9dd121b16be4ef9a7eaaae45390d1871760ba1d"
_ASCEND_FACTORY_SHA256 = "5056ddd4e3687d4b81314cef72a5d9a799be7f0f14a27e12e1525fd8e351edec"
_ASCEND_EAGLE_SHA256 = "aa5994f69f3050dd0a7cd374676857ddb787153083d7f6f9b3566cd3e4dc3474"
_SPEC_CONFIG_SHA256 = "4e3d0a9b93f54fc0f25897e8c7bba5412e3851dd819950964c201ed6376e08b7"
_VLLM_EAGLE_SHA256 = "b2b1f7d15117b43108be368756d9c6e9334d9415a070e46178e18bb7c6476378"
# This private v2 installs the opt-in GDN wrapper before early valid-count.
# The Eagle-init wrapper checked below is still defined by this exact mtp.py.
_LOCAL_MTP_SHA256 = "3077e7af576815acca469274f24dc667487df0907a396a78141c506ca05598f6"
_MARKER = "_vspec_q35_early_valid_count"


@dataclass
class _Pending:
    sampled_token_ids: torch.Tensor
    counts_gpu: torch.Tensor
    req_ids: tuple[str, ...]
    batch_size: int
    event: Any
    copy_stream: Any
    host_buffer: torch.Tensor
    input_batch: Any
    discard_tensor: torch.Tensor
    num_discarded_requests: int
    discard_indices_cpu: tuple[int, ...]
    vocab_size: int
    sampled_version: int | None
    prepared_count: torch.Tensor | None = None
    prepared_from_same_ids: bool = False


@dataclass
class _ValidatedCommit:
    req_ids: tuple[str, ...]
    counts_cpu: np.ndarray
    gamma: int


def _tensor_version_or_inference(tensor: torch.Tensor) -> int | None:
    """Inference-mode tensors have no version counter; never pretend they do."""
    try:
        return tensor._version
    except RuntimeError as exc:
        if "Inference tensors do not track version counter" in str(exc):
            return None
        raise


def _snapshot_unchanged(runner: Any, pending: _Pending) -> bool:
    try:
        n = pending.num_discarded_requests
        return all(
            (
                runner.input_batch is pending.input_batch,
                runner.input_batch.vocab_size == pending.vocab_size,
                runner.num_discarded_requests == n,
                runner.discard_request_indices.gpu is pending.discard_tensor,
                tuple(int(i) for i in runner.discard_request_indices.np[:n])
                == pending.discard_indices_cpu,
                _tensor_version_or_inference(pending.sampled_token_ids)
                == pending.sampled_version,
            )
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


def _validate_seq_len_correction(
    before: np.ndarray,
    after: np.ndarray,
    prev_positions: np.ndarray,
    prev_num_draft_tokens: np.ndarray,
    current_req_ids: tuple[str, ...],
    commit: _ValidatedCommit,
) -> int:
    """Independently check next-step row mapping and CPU correction."""
    n = len(current_req_ids)
    if len(before) != n or len(after) != n or len(prev_positions) < n:
        raise RuntimeError("early valid-count seq-len diagnostic shape mismatch")
    expected = before.copy()
    participating_rows = 0
    for row in range(n):
        prev_row = int(prev_positions[row])
        if prev_row < 0:
            continue
        if prev_row >= len(commit.req_ids) or prev_row >= len(prev_num_draft_tokens):
            raise RuntimeError("early valid-count prev_positions out of previous batch")
        drafts = int(prev_num_draft_tokens[prev_row])
        if drafts <= 0:
            continue
        if current_req_ids[row] != commit.req_ids[prev_row]:
            raise RuntimeError("early valid-count request row mapping mismatch")
        count = int(commit.counts_cpu[prev_row])
        if not 0 <= count <= drafts + 1:
            raise RuntimeError("early valid-count count outside previous draft width")
        expected[row] -= drafts + 1 - count
        participating_rows += 1
    if not np.array_equal(expected, after):
        raise RuntimeError("early valid-count next-step CPU seq-len correction mismatch")
    return participating_rows


def _source_is_pinned(path: str | Path, expected_sha256: str) -> bool:
    """Reject unknown host code before installing any monkeypatch."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_sha256
    except OSError:
        return False


def _method_tree(path: str | Path, class_name: str, method_name: str) -> ast.FunctionDef | None:
    """Return a named method from a source file without importing the host."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    return method
    return None


def _ast_contract_holds(runner_path: str | Path, proposer_path: str | Path, sampler_path: str | Path) -> bool:
    """Pin the exact sampler-to-proposer count and event publication ABI."""
    sample = _method_tree(runner_path, "NPUModelRunner", "sample_tokens")
    bookkeeping = _method_tree(runner_path, "NPUModelRunner", "_bookkeeping_sync")
    propose = _method_tree(runner_path, "NPUModelRunner", "propose_draft_token_ids")
    copy = _method_tree(runner_path, "NPUModelRunner", "_copy_valid_sampled_token_count")
    correct = _method_tree(runner_path, "NPUModelRunner", "_correct_optimistic_seq_lens_cpu")
    prepare = _method_tree(proposer_path, "AscendSpecDecodeBaseProposer", "prepare_next_token_ids_padded")
    reject = _method_tree(sampler_path, "AscendRejectionSampler", "forward")
    if any(method is None for method in (sample, bookkeeping, propose, copy, correct, prepare, reject)):
        return False
    sample_text = ast.unparse(sample)
    bookkeeping_text = ast.unparse(bookkeeping)
    propose_text = ast.unparse(propose)
    copy_text = ast.unparse(copy)
    correct_text = ast.unparse(correct)
    prepare_text = ast.unparse(prepare)
    reject_text = ast.unparse(reject)
    return all(
        (
            "sampler_output = self._sample(logits, spec_decode_metadata)" in sample_text,
            "propose_draft_token_ids(sampler_output.sampled_token_ids)" in sample_text,
            "sampled_token_ids = sampler_output.sampled_token_ids" in bookkeeping_text,
            "self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)" in propose_text,
            "self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)" in copy_text,
            "counts_cpu[:counts.shape[0]].copy_(counts, non_blocking=True)" in copy_text,
            "self.valid_sampled_token_count_event.record()" in copy_text,
            "self.valid_sampled_token_count_gpu = valid_sampled_tokens_count" in copy_text,
            "self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)" in copy_text,
            "self.valid_sampled_token_count_event.synchronize()" in correct_text,
            "correct_optimistic_seq_lens_cpu(" in correct_text,
            "valid_sampled_token_ids_gpu = sampled_token_ids.clone()" in prepare_text,
            "valid_sampled_token_ids_gpu = DeviceOperator.index_fill(" in prepare_text,
            "valid_mask = (valid_sampled_token_ids_gpu != -1) & (valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)" in prepare_text,
            "valid_sampled_tokens_count = valid_mask.sum(dim=1)" in prepare_text,
            "sampled_token_ids=output_token_ids" in reject_text,
        )
    )


def _eagle_factory_contract_holds(
    factory_path: str | Path, ascend_eagle_path: str | Path,
    speculative_path: str | Path, vllm_eagle_path: str | Path,
) -> bool:
    """Prove this Q35 MTP selects the exact Eagle proposer and base prepare."""
    try:
        factory_tree = ast.parse(Path(factory_path).read_text(encoding="utf-8"))
        eagle_tree = ast.parse(Path(ascend_eagle_path).read_text(encoding="utf-8"))
        vllm_eagle_tree = ast.parse(Path(vllm_eagle_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    factory = next((node for node in factory_tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_spec_decode_method"), None)
    eagle = next((node for node in eagle_tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendEagleProposer"), None)
    vllm_eagle = next((node for node in vllm_eagle_tree.body if isinstance(node, ast.ClassDef) and node.name == "EagleProposer"), None)
    use_step = _method_tree(speculative_path, "SpeculativeConfig", "use_step3p5_mtp")
    if any(node is None for node in (factory, eagle, vllm_eagle, use_step)):
        return False
    assert isinstance(factory, ast.FunctionDef)
    assert isinstance(eagle, ast.ClassDef)
    assert isinstance(vllm_eagle, ast.ClassDef)
    assert isinstance(use_step, ast.FunctionDef)
    factory_text = ast.unparse(factory)
    return all((
        "elif method in ('eagle', 'eagle3', 'mtp'):" in factory_text,
        "if speculative_config is not None and speculative_config.use_step3p5_mtp():" in factory_text,
        "return AscendStep3p5MTPProposer(vllm_config, device, runner)" in factory_text,
        "return AscendEagleProposer(vllm_config, device, runner)" in factory_text,
        [ast.unparse(base) for base in eagle.bases] == ["EagleProposer", "AscendSpecDecodeBaseProposer"],
        [ast.unparse(base) for base in vllm_eagle.bases] == ["SpecDecodeBaseProposer"],
        not any(isinstance(node, ast.FunctionDef) and node.name == "prepare_next_token_ids_padded" for node in eagle.body),
        not any(isinstance(node, ast.FunctionDef) and node.name == "prepare_next_token_ids_padded" for node in vllm_eagle.body),
        "== 'step3p5_mtp'" in ast.unparse(use_step),
    ))


def _eagle_init_contract_holds(function: Any, eagle_module: Any, metadata_enabled: bool) -> bool:
    """Allow only the pinned local Q35 metadata wrapper around Eagle init.

    vSpec installs that wrapper before this early-count patch. Reject any
    unknown wrapper, wrong origin, or a missing expected fusion wrapper.
    The prepare method remains independently required to be the exact pinned
    Ascend base method; this exception is only for Eagle construction.
    """
    local_mtp = Path(__file__).with_name("mtp.py").resolve()
    if not _source_is_pinned(local_mtp, _LOCAL_MTP_SHA256):
        return False

    def original_holds(candidate: Any) -> bool:
        return all((
            getattr(candidate, "__module__", None) == eagle_module.__name__,
            getattr(candidate, "__wrapped__", None) is None,
            Path(candidate.__code__.co_filename).resolve() == Path(eagle_module.__file__).resolve(),
        ))

    try:
        if not metadata_enabled:
            return original_holds(function)
        original = getattr(function, "__wrapped__", None)
        return all((
            getattr(function, "_vspec_q35_eagle_metadata_fusion", False) is True,
            function.__module__ == eagle_module.__name__,
            Path(function.__code__.co_filename).resolve() == local_mtp,
            original is not None and original_holds(original),
        ))
    except (AttributeError, TypeError, ValueError, OSError):
        return False


def _early_counts(
    sampled_token_ids: torch.Tensor,
    vocab_size: int,
    discard_request_indices: torch.Tensor,
) -> torch.Tensor:
    """Match Ascend's mask-after-index-fill count without changing IDs.

    The current proposer clones ``sampled_token_ids``, fills discarded rows
    with -1, then sums ``(id != -1) & (id < vocab_size)`` per row. Filling the
    count of those same rows with zero is algebraically identical. The tensor
    stays on device; no ``item()``, ``tolist()`` or host synchronization.
    """
    counts = ((sampled_token_ids != -1) & (sampled_token_ids < vocab_size)).sum(dim=1)
    if discard_request_indices.numel():
        counts.index_fill_(0, discard_request_indices, 0)
    return counts


def _eligible(runner: Any, metadata: Any, sampled_token_ids: Any, proposer_cls: type) -> bool:
    """All checks are host-only and conservative; unknown means native path."""
    try:
        if not (runner.use_async_scheduling and runner.use_async_spec_decode):
            return False
        spec_config = runner.speculative_config
        if spec_config.method != "mtp" or spec_config.disable_padded_drafter_batch:
            return False
        if not (spec_config.use_eagle() or spec_config.uses_draft_model()):
            return False
        model_config = runner.model_config
        if not model_config.is_hybrid or model_config.hf_config.model_type != "qwen3_5_moe":
            return False
        if runner.cache_config.mamba_cache_mode != "align" or runner.dcp_size != 1:
            return False
        if runner.parallel_config.pipeline_parallel_size != 1 or runner.use_compress:
            return False
        if runner.with_prefill is not False or runner.attn_state.name != "SpecDecoding":
            return False
        batch = runner.input_batch
        n = batch.num_reqs
        if n <= 0 or len(batch.req_ids) != n:
            return False
        if not isinstance(batch.num_computed_tokens_cpu, np.ndarray) or not isinstance(
            batch.num_prompt_tokens, np.ndarray
        ):
            return False
        if not np.all(batch.num_computed_tokens_cpu[:n] >= batch.num_prompt_tokens[:n]):
            return False
        if type(runner.drafter) is not proposer_cls:
            return False
        if metadata is None or metadata.max_spec_len not in (4, 6):
            return False
        if type(metadata.num_draft_tokens) is not list or len(metadata.num_draft_tokens) != n:
            return False
        if any(type(k) is not int or k != metadata.max_spec_len for k in metadata.num_draft_tokens):
            return False
        if not isinstance(sampled_token_ids, torch.Tensor):
            return False
        if sampled_token_ids.device.type != "npu" or sampled_token_ids.ndim != 2:
            return False
        if sampled_token_ids.shape != (n, metadata.max_spec_len + 1):
            return False
        if sampled_token_ids.dtype not in (torch.int32, torch.int64):
            return False
        if not sampled_token_ids.is_contiguous():
            return False
        num_discarded = runner.num_discarded_requests
        if not isinstance(num_discarded, int) or not 0 <= num_discarded <= n:
            return False
        discard_cpu = runner.discard_request_indices.np[:num_discarded]
        if np.any(discard_cpu < 0) or np.any(discard_cpu >= n):
            return False
        if len(set(discard_cpu.tolist())) != num_discarded:
            return False
        event = runner.valid_sampled_token_count_event
        copy_stream = runner.valid_sampled_token_count_copy_stream
        host_buffer = runner.valid_sampled_token_count_cpu
        if event is None or copy_stream is None or host_buffer is None:
            return False
        if host_buffer.dtype != torch.int64 or host_buffer.numel() < n or not host_buffer.is_pinned():
            return False
        if runner.discard_request_indices.gpu.device.type != "npu":
            return False
        npu = getattr(torch, "npu", None)
        capture_probe = getattr(npu, "is_current_stream_capturing", None)
        if not callable(capture_probe) or capture_probe():
            return False
        return True
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


def _eligibility_fallback_reason(
    runner: Any, metadata: Any, sampled_token_ids: Any, proposer_cls: type,
) -> str:
    """Host-only explanation of the first failed guard; no tensor data read.

    This is diagnostic only and never overrides the original `_eligible`.
    Names are stable so the smoke can report both workers' actual fallback.
    """
    try:
        if not runner.use_async_scheduling or not runner.use_async_spec_decode:
            return "async_decode_disabled"
        spec_config = runner.speculative_config
        if spec_config.method != "mtp" or spec_config.disable_padded_drafter_batch:
            return "mtp_or_padded_batch_disabled"
        if not (spec_config.use_eagle() or spec_config.uses_draft_model()):
            return "unsupported_drafter_method"
        model_config = runner.model_config
        if not model_config.is_hybrid or model_config.hf_config.model_type != "qwen3_5_moe":
            return "not_q35_hybrid"
        if runner.cache_config.mamba_cache_mode != "align" or runner.dcp_size != 1:
            return "mamba_align_or_dcp_mismatch"
        if runner.parallel_config.pipeline_parallel_size != 1 or runner.use_compress:
            return "pipeline_or_compression_mismatch"
        if runner.with_prefill is not False or runner.attn_state.name != "SpecDecoding":
            return "not_pure_spec_decode"
        batch = runner.input_batch
        n = batch.num_reqs
        if n <= 0 or len(batch.req_ids) != n:
            return "empty_or_misaligned_batch"
        if not isinstance(batch.num_computed_tokens_cpu, np.ndarray) or not isinstance(batch.num_prompt_tokens, np.ndarray):
            return "prompt_count_metadata_type_mismatch"
        if not np.all(batch.num_computed_tokens_cpu[:n] >= batch.num_prompt_tokens[:n]):
            return "unfinished_prompt"
        if type(runner.drafter) is not proposer_cls:
            return "drafter_class_mismatch"
        if metadata is None or metadata.max_spec_len not in (4, 6):
            return "draft_metadata_or_gamma_mismatch"
        if type(metadata.num_draft_tokens) is not list or len(metadata.num_draft_tokens) != n:
            return "draft_width_list_mismatch"
        if any(type(k) is not int or k != metadata.max_spec_len for k in metadata.num_draft_tokens):
            return "nonuniform_draft_width"
        if not isinstance(sampled_token_ids, torch.Tensor):
            return "sampled_tensor_type_mismatch"
        if sampled_token_ids.device.type != "npu" or sampled_token_ids.ndim != 2:
            return "sampled_tensor_device_or_rank_mismatch"
        if sampled_token_ids.shape != (n, metadata.max_spec_len + 1):
            return "sampled_tensor_shape_mismatch"
        if sampled_token_ids.dtype not in (torch.int32, torch.int64):
            return "sampled_tensor_dtype_mismatch"
        if not sampled_token_ids.is_contiguous():
            return "sampled_tensor_not_contiguous"
        num_discarded = runner.num_discarded_requests
        if not isinstance(num_discarded, int) or not 0 <= num_discarded <= n:
            return "discard_count_invalid"
        discard_cpu = runner.discard_request_indices.np[:num_discarded]
        if np.any(discard_cpu < 0) or np.any(discard_cpu >= n):
            return "discard_index_out_of_bounds"
        if len(set(discard_cpu.tolist())) != num_discarded:
            return "duplicate_discard_index"
        event = runner.valid_sampled_token_count_event
        copy_stream = runner.valid_sampled_token_count_copy_stream
        host_buffer = runner.valid_sampled_token_count_cpu
        if event is None or copy_stream is None or host_buffer is None:
            return "copy_event_stream_or_host_buffer_missing"
        if host_buffer.dtype != torch.int64 or host_buffer.numel() < n or not host_buffer.is_pinned():
            return "host_count_buffer_not_pinned_int64"
        if runner.discard_request_indices.gpu.device.type != "npu":
            return "discard_tensor_not_npu"
        npu = getattr(torch, "npu", None)
        capture_probe = getattr(npu, "is_current_stream_capturing", None)
        if not callable(capture_probe) or capture_probe():
            return "capture_active_or_probe_missing"
        return "all_guard_checks_passed_on_reinspection"
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        return "guard_exception_" + type(exc).__name__


def _trace_fallback(runner: Any, metadata: Any, ids: Any, proposer_cls: type) -> None:
    if os.getenv("HUST_ISSUE18_Q35_EARLY_VALID_GUARD_TRACE", "0") != "1":
        return
    reason = _eligibility_fallback_reason(runner, metadata, ids, proposer_cls)
    counts = getattr(runner, "_vspec_early_valid_fallback_reasons", None)
    if counts is None:
        counts = {}
        runner._vspec_early_valid_fallback_reasons = counts
    hits = counts.get(reason, 0) + 1
    counts[reason] = hits
    if hits <= 3 or hits % 128 == 0:
        logger.warning(
            "Q35 early valid-count guard fallback: pid=%d, reason=%s, seen=%d",
            os.getpid(), reason, hits,
        )


def _stage_copy(runner: Any, counts_gpu: torch.Tensor, n: int) -> None:
    """Queue D2H early, but do not publish the event before the original site."""
    copy_stream = runner.valid_sampled_token_count_copy_stream
    host_buffer = runner.valid_sampled_token_count_cpu
    default_stream = torch.npu.current_stream()
    with torch.npu.stream(copy_stream):
        copy_stream.wait_stream(default_stream)
        # The producer tensor may otherwise be released after the proposer
        # returns while this side-stream D2H still reads it.
        counts_gpu.record_stream(copy_stream)
        host_buffer[:n].copy_(counts_gpu, non_blocking=True)


def _wrap_runner_and_proposer(runner_cls: type, proposer_base_cls: type, proposer_cls: type) -> bool:
    """Install one producer, one identity witness and one consumer wrapper."""
    original_sample = runner_cls._sample
    if getattr(original_sample, _MARKER, False):
        return False
    original_copy = runner_cls._copy_valid_sampled_token_count
    original_prepare = proposer_base_cls.prepare_next_token_ids_padded

    @wraps(original_sample)
    def sample(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        # A staged D2H without its original-site commit is an unknown state.
        # Never silently replace it with a later step's count.
        if getattr(self, "_vspec_early_valid_pending", None) is not None:
            raise RuntimeError("uncommitted early valid-count D2H from previous sample")
        # The previous step's correction has already run in execute_model.
        # A fallback step must never leave a stale diagnostic snapshot.
        self._vspec_early_valid_last_commit = None
        output = original_sample(self, logits, spec_decode_metadata)
        ids = getattr(output, "sampled_token_ids", None)
        if not _eligible(self, spec_decode_metadata, ids, proposer_cls):
            _trace_fallback(self, spec_decode_metadata, ids, proposer_cls)
            return output
        n = self.input_batch.num_reqs
        discard = self.discard_request_indices.gpu[: self.num_discarded_requests]
        discard_cpu = tuple(
            int(i) for i in self.discard_request_indices.np[: self.num_discarded_requests]
        )
        sampled_version = _tensor_version_or_inference(ids)
        counts = _early_counts(ids, self.input_batch.vocab_size, discard)
        _stage_copy(self, counts, n)
        self._vspec_early_valid_pending = _Pending(
            sampled_token_ids=ids,
            counts_gpu=counts,
            req_ids=tuple(self.input_batch.req_ids),
            batch_size=n,
            event=self.valid_sampled_token_count_event,
            copy_stream=self.valid_sampled_token_count_copy_stream,
            host_buffer=self.valid_sampled_token_count_cpu,
            input_batch=self.input_batch,
            discard_tensor=self.discard_request_indices.gpu,
            num_discarded_requests=self.num_discarded_requests,
            discard_indices_cpu=discard_cpu,
            vocab_size=self.input_batch.vocab_size,
            sampled_version=sampled_version,
        )
        return output

    @wraps(original_prepare)
    def prepare_next_token_ids_padded(self: Any, sampled_token_ids: Any, *args: Any, **kwargs: Any) -> Any:
        runner = getattr(self, "runner", None)
        pending = getattr(runner, "_vspec_early_valid_pending", None)
        same_ids = pending is not None and sampled_token_ids is pending.sampled_token_ids
        if pending is not None and not (
            same_ids
            and len(args) == 4
            and not kwargs
            and args[0] is runner.requests
            and args[1] is pending.input_batch
            and args[2] is pending.discard_tensor
            and args[3] == pending.num_discarded_requests
            and _snapshot_unchanged(runner, pending)
        ):
            raise RuntimeError("early valid-count proposer input witness mismatch")
        result = original_prepare(self, sampled_token_ids, *args, **kwargs)
        if pending is not None and not _snapshot_unchanged(runner, pending):
            raise RuntimeError("early valid-count sampler/discard state mutated in proposer")
        if same_ids and isinstance(result, tuple) and len(result) == 2:
            pending.prepared_count = result[1]
            pending.prepared_from_same_ids = True
        elif pending is not None:
            raise RuntimeError("early valid-count proposer return witness mismatch")
        return result

    @wraps(original_copy)
    def copy_valid_sampled_token_count(self: Any, next_token_ids: Any, valid_count: Any) -> Any:
        pending = getattr(self, "_vspec_early_valid_pending", None)
        self._vspec_early_valid_pending = None
        if pending is None:
            # All unknown/pre-stage paths retain the pinned native copy.
            return original_copy(self, next_token_ids, valid_count)
        if (
            pending.prepared_from_same_ids
            and valid_count is pending.prepared_count
            and isinstance(valid_count, torch.Tensor)
            and valid_count.shape == pending.counts_gpu.shape
            and self.input_batch.num_reqs == pending.batch_size
            and tuple(self.input_batch.req_ids) == pending.req_ids
            and self.valid_sampled_token_count_event is pending.event
            and self.valid_sampled_token_count_copy_stream is pending.copy_stream
            and self.valid_sampled_token_count_cpu is pending.host_buffer
            and _snapshot_unchanged(self, pending)
        ):
            # Diagnostic only: synchronize the device to prove equality in a
            # small smoke. Never enable during throughput measurement.
            validate = os.getenv("HUST_ISSUE18_Q35_EARLY_VALID_COUNT_VALIDATE", "0") == "1"
            if validate:
                if not torch.equal(pending.counts_gpu, valid_count):
                    raise RuntimeError("early valid-count parity mismatch")
            # Publish readiness only where the original implementation would
            # record its event. The D2H itself has already been queued on this
            # same stream, so the record remains ordered after that copy.
            with torch.npu.stream(pending.copy_stream):
                pending.event.record()
            if validate:
                pending.event.synchronize()
                expected_cpu = valid_count.cpu()
                if not torch.equal(
                    pending.host_buffer[: pending.batch_size], expected_cpu
                ):
                    raise RuntimeError("early valid-count host-copy parity mismatch")
                self._vspec_early_valid_last_commit = _ValidatedCommit(
                    req_ids=pending.req_ids,
                    counts_cpu=expected_cpu.numpy().copy(),
                    gamma=pending.sampled_token_ids.shape[1] - 1,
                )
            # Preserve the original GPU-side correction source and next-token
            # bookkeeping. Only the duplicate D2H/event record is skipped.
            if self.use_async_spec_decode:
                self.valid_sampled_token_count_gpu = valid_count
            self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)
            hits = getattr(self, "_vspec_early_valid_count_hits", 0) + 1
            self._vspec_early_valid_count_hits = hits
            gamma = pending.sampled_token_ids.shape[1] - 1
            hits_by_gamma = getattr(self, "_vspec_early_valid_hits_by_gamma", {})
            hits_by_gamma[gamma] = hits_by_gamma.get(gamma, 0) + 1
            self._vspec_early_valid_hits_by_gamma = hits_by_gamma
            if hits_by_gamma[gamma] == 1:
                logger.warning(
                    "Q35 early valid-count first eligible step: pid=%d, batch=%d, gamma=%d, validation=%s, sampled_version_available=%s",
                    os.getpid(),
                    pending.batch_size,
                    gamma,
                    validate,
                    pending.sampled_version is not None,
                )
            return None
        # After staging, a late identity/batch mismatch is an invariant
        # violation, not a safe fallback: a second write could race a reader
        # of the shared pinned buffer. Abort rather than return wrong output.
        raise RuntimeError("early valid-count D2H lost sampler/proposer identity witness")

    setattr(sample, _MARKER, True)
    setattr(prepare_next_token_ids_padded, _MARKER, True)
    setattr(copy_valid_sampled_token_count, _MARKER, True)
    proposer_base_cls.prepare_next_token_ids_padded = prepare_next_token_ids_padded
    runner_cls._copy_valid_sampled_token_count = copy_valid_sampled_token_count
    runner_cls._sample = sample
    return True


def _install_correction_validator(runner_cls: type) -> None:
    """Diagnostic-only next-step check; no change to throughput path."""
    original = runner_cls._correct_optimistic_seq_lens_cpu

    @wraps(original)
    def corrected(self: Any, num_reqs: int) -> Any:
        commit = getattr(self, "_vspec_early_valid_last_commit", None)
        before = (
            self.optimistic_seq_lens_cpu.numpy()[:num_reqs].copy()
            if commit is not None else None
        )
        result = original(self, num_reqs)
        if commit is None:
            return result
        self._vspec_early_valid_last_commit = None
        if not np.array_equal(
            self.valid_sampled_token_count_cpu.numpy()[: len(commit.req_ids)],
            commit.counts_cpu,
        ):
            raise RuntimeError("early valid-count previous host buffer changed before correction")
        rows = _validate_seq_len_correction(
            before=before,
            after=self.optimistic_seq_lens_cpu.numpy()[:num_reqs].copy(),
            prev_positions=self.prev_positions.np,
            prev_num_draft_tokens=self.prev_num_draft_tokens.np,
            current_req_ids=tuple(self.input_batch.req_ids[:num_reqs]),
            commit=commit,
        )
        if rows:
            mapping_hits = getattr(self, "_vspec_early_valid_mapping_hits_by_gamma", {})
            mapping_hits[commit.gamma] = mapping_hits.get(commit.gamma, 0) + 1
            self._vspec_early_valid_mapping_hits_by_gamma = mapping_hits
            if mapping_hits[commit.gamma] == 1:
                logger.warning(
                    "Q35 early valid-count first next-step mapping parity: pid=%d, gamma=%d, rows=%d",
                    os.getpid(),
                    commit.gamma,
                    rows,
                )
        return result

    setattr(corrected, _MARKER, True)
    runner_cls._correct_optimistic_seq_lens_cpu = corrected


def apply_q35_early_valid_count_patch() -> bool:
    """Install only on the pinned host source and expected Q35 proposer MRO."""
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
    from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.config.speculative import SpeculativeConfig
    from vllm.v1.spec_decode.eagle import EagleProposer
    import vllm_ascend.spec_decode as factory_module
    import vllm_ascend.spec_decode.eagle_proposer as eagle_module
    import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
    import vllm_ascend.sample.rejection_sampler as sampler_module
    import vllm_ascend.worker.model_runner_v1 as runner_module
    import vllm_ascend.device.device_op as device_op_module
    import vllm.config.speculative as speculative_module
    import vllm.v1.spec_decode.eagle as vllm_eagle_module

    pinned = (
        _source_is_pinned(runner_module.__file__, _RUNNER_SHA256)
        and _source_is_pinned(proposer_module.__file__, _PROPOSER_SHA256)
        and _source_is_pinned(sampler_module.__file__, _SAMPLER_SHA256)
        and _source_is_pinned(device_op_module.__file__, _DEVICE_OP_SHA256)
        and _source_is_pinned(factory_module.__file__, _ASCEND_FACTORY_SHA256)
        and _source_is_pinned(eagle_module.__file__, _ASCEND_EAGLE_SHA256)
        and _source_is_pinned(speculative_module.__file__, _SPEC_CONFIG_SHA256)
        and _source_is_pinned(vllm_eagle_module.__file__, _VLLM_EAGLE_SHA256)
        and _source_is_pinned(Path(__file__).with_name("mtp.py"), _LOCAL_MTP_SHA256)
        and _ast_contract_holds(
            runner_module.__file__, proposer_module.__file__, sampler_module.__file__
        )
        and _eagle_factory_contract_holds(
            factory_module.__file__, eagle_module.__file__,
            speculative_module.__file__, vllm_eagle_module.__file__,
        )
    )
    if not pinned:
        logger.warning("Q35 early valid-count D2H disabled: pinned Ascend source hash mismatch")
        return False
    host_functions = (
        (NPUModelRunner._sample, runner_module),
        (NPUModelRunner._bookkeeping_sync, runner_module),
        (NPUModelRunner._copy_valid_sampled_token_count, runner_module),
        (NPUModelRunner._correct_optimistic_seq_lens_cpu, runner_module),
        (AscendSpecDecodeBaseProposer.prepare_next_token_ids_padded, proposer_module),
        (EagleProposer.__init__, vllm_eagle_module),
        (SpeculativeConfig.use_step3p5_mtp, speculative_module),
        (factory_module.get_spec_decode_method, factory_module),
        (AscendRejectionSampler.forward, sampler_module),
    )
    if (
        AscendEagleProposer.prepare_next_token_ids_padded
        is not AscendSpecDecodeBaseProposer.prepare_next_token_ids_padded
        or not _eagle_init_contract_holds(
            AscendEagleProposer.__init__, eagle_module,
            os.getenv("VSPEC_DRAFT_Q35_METADATA_FUSION", "0") == "1",
        )
        or any(
            function.__module__ != module.__name__
            or getattr(function, "__wrapped__", None) is not None
            or Path(function.__code__.co_filename).resolve() != Path(module.__file__).resolve()
            for function, module in host_functions
        )
    ):
        logger.warning("Q35 early valid-count D2H disabled: unknown sampler/proposer/runner wrapper")
        return False
    installed = _wrap_runner_and_proposer(
        NPUModelRunner,
        AscendSpecDecodeBaseProposer,
        AscendEagleProposer,
    )
    if installed:
        if os.getenv("HUST_ISSUE18_Q35_EARLY_VALID_COUNT_VALIDATE", "0") == "1":
            _install_correction_validator(NPUModelRunner)
        logger.warning("Q35 early valid-count D2H prototype enabled: pid=%d (pure-decode fail-closed path)", os.getpid())
    return installed
