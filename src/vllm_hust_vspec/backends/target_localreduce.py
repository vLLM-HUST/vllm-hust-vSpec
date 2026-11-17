"""Q35 TP2 greedy target reduction with exact shard-local stop masking.

Static model admission is voted once. Per-step admission uses the sampling
and scheduler metadata replicated by the TP executor, never a device value.
Unexpected rank-local tensor layouts abort instead of choosing a different
collective. A late replicated grammar request gathers both shards normally.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from functools import wraps
from numbers import Integral
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_TARGET_EXECUTE_RUNNER: ContextVar[Any | None] = ContextVar(
    "vspec_target_localreduce_execute_runner", default=None
)
_LOCAL_TARGET_SAMPLING: ContextVar[bool] = ContextVar(
    "vspec_target_localreduce_sampling", default=False
)
_LOCAL_TARGET_SHARD: ContextVar[tuple[int, int] | None] = ContextVar(
    "vspec_target_localreduce_shard", default=None
)
_MIN_TOKENS_CLASS: type | None = None


def _packed_greedy(logits: torch.Tensor) -> torch.Tensor:
    """One collective carries FP32 max values and exactly represented IDs."""
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    n, local_vocab = logits.shape
    if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16) or local_vocab * group.world_size >= 2**24:
        raise RuntimeError('Packed greedy reduction requires FP32/FP16/BF16 logits and vocabulary below2**24')
    values, local_ids = logits.max(dim=-1)
    global_ids = local_ids.to(torch.float32) + group.rank_in_group * local_vocab
    packed = torch.stack((values.float(), global_ids), dim=-1)
    gathered = group.all_gather(packed, dim=-1).reshape(n, group.world_size, 2)
    # Ranks are in vocabulary order, so argmax tie-breaking is the same as
    # an argmax over the full gathered vocabulary, including within-rank ties.
    winner = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(1, winner).squeeze(1).to(torch.int64)


def _combined_stop_mask(processor, draft_counts, rank, local_vocab):
    """Map native bonus/draft MinTokens masks onto the unsplit logit rows."""
    signature_entries = []
    for row, (minimum, outputs, stops, structured) in processor.min_toks.items():
        if structured:
            raise RuntimeError('Structured MinTokens cannot use local greedy sampling')
        masked_drafts = min(max(minimum - len(outputs), 0), draft_counts[row])
        signature_entries.append((row, masked_drafts, tuple(sorted(stops))))
    signature = (tuple(draft_counts), rank, local_vocab, tuple(signature_entries))
    cached = getattr(processor, '_vspec_combined_stop_mask', None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    starts = [0]
    for count in draft_counts:
        starts.append(starts[-1] + count + 1)
    rows, tokens = [], []
    offset = rank * local_vocab
    for request, masked_drafts, stops in signature_entries:
        positions = [starts[request + 1] - 1, *range(starts[request], starts[request] + masked_drafts)]
        for stop in stops:
            if offset <= stop < offset + local_vocab:
                rows.extend(positions)
                tokens.extend([stop - offset] * len(positions))
    mask = None
    if rows:
        mask = (processor._device_tensor(rows, torch.int64), processor._device_tensor(tokens, torch.int64))
    processor._vspec_combined_stop_mask = (signature, mask)
    return mask


class _GreedyLocalConfig:
    enable_reduce_sample = True

    def __init__(self, source: Any) -> None:
        self._source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


def _safe_min_tokens_processor(processor: Any, batch_size: int) -> bool:
    """Admit only non-structured, in-vocabulary MinTokens state."""
    if _MIN_TOKENS_CLASS is None or type(processor) is not _MIN_TOKENS_CLASS:
        return False
    entries = getattr(processor, "min_toks", None)
    if not isinstance(entries, dict):
        return False
    restore_slice = getattr(processor, "restore_logits_slice", None)
    if (
        restore_slice is None
        or len(restore_slice) != 2
        or restore_slice[0].numel() != 0
        or restore_slice[1].numel() != 0
    ):
        return False
    for row, entry in entries.items():
        if not isinstance(row, Integral) or not 0 <= row < batch_size:
            return False
        if not isinstance(entry, tuple) or len(entry) != 4:
            return False
        min_toks, output_ids, stop_ids, structured = entry
        if (
            not isinstance(min_toks, Integral)
            or min_toks < 0
            or not hasattr(output_ids, "__len__")
            or not isinstance(stop_ids, set)
            or structured
        ):
            return False
        if any(
            not isinstance(tok, Integral) or not 0 <= tok < 248320
            for tok in stop_ids
        ):
            return False
    return True


def _plain_greedy_metadata(metadata: Any, batch_size: int | None = None) -> bool:
    """Reject all global-logit features except exact MinTokens stop masking."""
    if metadata is None or not getattr(metadata, "all_greedy", False):
        return False
    if getattr(metadata, "all_random", True):
        return False
    if getattr(metadata, "max_num_logprobs", None) is not None:
        return False
    if getattr(metadata, "logprob_token_ids", None):
        return False
    if not getattr(metadata, "no_penalties", False):
        return False
    if getattr(metadata, "allowed_token_ids_mask", None) is not None:
        return False
    if getattr(metadata, "bad_words_token_ids", None):
        return False
    if getattr(metadata, "top_k", None) is not None:
        return False
    if getattr(metadata, "top_p", None) is not None:
        return False
    processors = getattr(metadata, "logitsprocs", None)
    if processors is None:
        return False
    non_invariant = getattr(processors, "non_argmax_invariant", None)
    if (
        not isinstance(non_invariant, list)
        or len(non_invariant) != 1
        or getattr(processors, "argmax_invariant", None) != []
    ):
        return False
    output_ids = getattr(metadata, "output_token_ids", None)
    # With no penalties/bad words, vLLM can omit its redundant output-ID
    # lists even while MinTokens tracks request histories in its own state.
    # The runner's replicated batch count is authoritative for row bounds.
    if batch_size is None and isinstance(output_ids, list):
        batch_size = len(output_ids)
    if not isinstance(output_ids, list) or not _safe_min_tokens_processor(
        non_invariant[0], batch_size
    ):
        return False
    # Even an untracked holder is rejected: future request-state changes must
    # not silently turn the local path into a constrained-vocabulary sampler.
    if getattr(metadata, "thinking_budget_state_holder", None) is not None:
        return False
    return True


def _plain_greedy_runner(runner: Any) -> bool:
    if getattr(getattr(runner, "speculative_config", None), "method", None) != "mtp":
        return False
    if getattr(getattr(runner, "ascend_config", None), "enable_reduce_sample", False):
        return False
    if getattr(runner, "lora_config", None) is not None:
        return False
    if not getattr(runner, "use_async_scheduling", False):
        return False
    if getattr(runner, "num_prompt_logprobs", None):
        return False
    if getattr(runner, "_vspec_target_structured", True):
        return False
    parallel = getattr(runner, "parallel_config", None)
    if (
        getattr(parallel, "tensor_parallel_size", None) != 2
        or getattr(parallel, "pipeline_parallel_size", None) != 1
        or getattr(parallel, "decode_context_parallel_size", None) != 1
    ):
        return False
    model = getattr(runner, "model", None)
    if model is None or type(model).__name__ != "ACLGraphWrapper":
        return False
    if type(model.unwrap()).__name__ != "Qwen3_5MoeForConditionalGeneration":
        return False
    if not _plain_greedy_metadata(
        getattr(getattr(runner, "input_batch", None), "sampling_metadata", None),
        getattr(getattr(runner, "input_batch", None), "num_reqs", None),
    ):
        return False
    rejection_sampler = getattr(runner, "rejection_sampler", None)
    sampler = getattr(runner, "sampler", None)
    if (
        rejection_sampler is None
        or type(rejection_sampler).__name__ != "AscendRejectionSampler"
        or sampler is None
        or type(sampler).__name__ != "AscendSampler"
        or getattr(rejection_sampler, "synthetic_mode", False)
    ):
        return False
    # Block/entropy verification can inspect target distributions even at
    # temperature zero; they are outside this token-parity argument.
    ascend_config = getattr(runner, "ascend_config", None)
    rejection_config = getattr(ascend_config, "rejection_sampler_config", None)
    if rejection_config is None or (
        getattr(rejection_config, "enable_block_verify", False)
        or getattr(rejection_config, "enable_entropy_verify", False)
    ):
        return False
    return True


def _target_local_head(model: Any, tp_size: int) -> tuple[Any, int] | None:
    """Return a validated, unpadded local LM head or disable the experiment."""
    language_model = getattr(model, "language_model", None)
    processor = getattr(language_model, "logits_processor", None)
    lm_head = getattr(language_model, "lm_head", None)
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(getattr(config, "text_config", None), "vocab_size", None)
    if (
        processor is None
        or type(processor).__name__ != "AscendLogitsProcessor"
        or lm_head is None
        or not callable(getattr(model, "compute_logits_local", None))
        or not getattr(processor, "use_all_gather", False)
        or not isinstance(vocab_size, int)
        or vocab_size != 248320
        or getattr(processor, "org_vocab_size", None) != vocab_size
        or tp_size != 2
        or vocab_size % tp_size != 0
    ):
        return None
    local_vocab = vocab_size // tp_size
    if getattr(lm_head, "num_org_embeddings_per_partition", None) != local_vocab:
        return None
    # Q35's 124160-token shard has no padding. Requiring this avoids the
    # rank*V_local offset and dummy-vocabulary corner cases in greedy_sample.
    if getattr(lm_head, "num_embeddings_per_partition", None) != local_vocab:
        return None
    return processor, local_vocab


def _full_logits_from_local(model: Any, local_logits: torch.Tensor) -> torch.Tensor:
    language_model = model.language_model
    processor = language_model.logits_processor
    full_logits = processor._gather_logits(local_logits)
    if full_logits is None:
        raise RuntimeError("Local target fallback gather returned no logits")
    return full_logits[..., : processor.org_vocab_size]


def _tp_vote_count(*flags: bool) -> tuple[int, ...]:
    """One Gloo CPU collective for all branch predicates; no NPU sync."""
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    if tp_group.world_size != 2 or tp_group.cpu_group is None:
        raise RuntimeError("Target local-reduce requires a TP2 CPU process group")
    ballot = torch.tensor([int(flag) for flag in flags], dtype=torch.int32)
    dist.all_reduce(ballot, op=dist.ReduceOp.SUM, group=tp_group.cpu_group)
    return tuple(int(value) for value in ballot.tolist())


def _spec_min_tokens_rows_safe(processor: Any, spec_metadata: Any) -> bool:
    if spec_metadata is None:
        return True
    counts = getattr(spec_metadata, "num_draft_tokens", None)
    if not isinstance(counts, (list, tuple)) or any(
        not isinstance(count, Integral) or count < 0 for count in counts
    ):
        return False
    return all(0 <= row < len(counts) for row in processor.min_toks)


def _install_min_tokens_local_masks(min_tokens_class: type) -> None:
    """Intercept only local target sampling; full-vocab calls are untouched."""
    if getattr(min_tokens_class.apply, "_vspec_local_minmask", False):
        return
    original_apply = min_tokens_class.apply
    original_spec_apply = min_tokens_class.apply_with_spec_decode

    @wraps(original_apply)
    def apply(self: Any, logits: torch.Tensor) -> torch.Tensor:
        shard = _LOCAL_TARGET_SHARD.get()
        if shard is None:
            return original_apply(self, logits)
        rank, local_vocab = shard
        if self.min_toks:
            source_slice = getattr(self, "logits_slice", None)
            cached = getattr(self, "_vspec_local_bonus_slice", None)
            if cached is None or cached[0] is not source_slice or cached[1] != shard:
                rows: list[int] = []
                toks: list[int] = []
                offset = rank * local_vocab
                for row, (_, _, stop_ids, structured) in self.min_toks.items():
                    if structured:
                        raise RuntimeError("Structured MinTokens restore is not local-safe")
                    for token_id in stop_ids:
                        if offset <= token_id < offset + local_vocab:
                            rows.append(row)
                            toks.append(token_id - offset)
                local_slice = (
                    self._device_tensor(rows, torch.int32),
                    self._device_tensor(toks, torch.int32),
                )
                self._vspec_local_bonus_slice = (source_slice, shard, local_slice)
            else:
                local_slice = cached[2]
            if local_slice[0].numel():
                logits.index_put_(local_slice, self.neg_inf_tensor)
        return logits

    @wraps(original_spec_apply)
    def apply_with_spec_decode(
        self: Any, logits: torch.Tensor, num_draft_tokens: list[int]
    ) -> torch.Tensor:
        shard = _LOCAL_TARGET_SHARD.get()
        if shard is None:
            return original_spec_apply(self, logits, num_draft_tokens)
        if not self.min_toks:
            return logits
        rank, local_vocab = shard
        offset = rank * local_vocab
        starts: list[int] = [0]
        for count in num_draft_tokens:
            starts.append(starts[-1] + count)
        rows: list[int] = []
        toks: list[int] = []
        for req_idx, (
            min_toks,
            output_ids,
            stop_ids,
            structured,
        ) in self.min_toks.items():
            if structured:
                raise RuntimeError("Structured MinTokens restore is not local-safe")
            n_mask = min(max(min_toks - len(output_ids), 0), num_draft_tokens[req_idx])
            local_stop_ids = [
                tok - offset
                for tok in stop_ids
                if offset <= tok < offset + local_vocab
            ]
            for row in range(starts[req_idx], starts[req_idx] + n_mask):
                rows.extend([row] * len(local_stop_ids))
                toks.extend(local_stop_ids)
        if rows:
            local_slice = (
                self._device_tensor(rows, torch.int64),
                self._device_tensor(toks, torch.int64),
            )
            logits.index_put_(local_slice, self.neg_inf_tensor)
        return logits

    apply._vspec_local_minmask = True  # type: ignore[attr-defined]
    min_tokens_class.apply = apply
    min_tokens_class.apply_with_spec_decode = apply_with_spec_decode


def install_target_localreduce() -> bool:
    """Install target-only local reduction without changing draft policy."""
    if os.getenv("HUST_ISSUE18_TARGET_GREEDY_LOCALREDUCE", "0") != "1":
        return False

    from vllm_ascend.sample import rejection_sampler as rejection_module
    from vllm_ascend.sample import sampler as sampler_module
    from vllm_ascend.utils import lmhead_tp_enable
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
    from vllm.distributed.parallel_state import get_tp_group

    global _MIN_TOKENS_CLASS
    _MIN_TOKENS_CLASS = MinTokensLogitsProcessor
    _install_min_tokens_local_masks(MinTokensLogitsProcessor)

    original_load_model = NPUModelRunner.load_model
    if getattr(original_load_model, "_vspec_target_localreduce", False):
        return False
    validate = os.getenv('HUST_ISSUE18_TARGET_GREEDY_LOCALREDUCE_VALIDATE') == '1'

    original_sampler_greedy = sampler_module.AscendSampler.greedy_sample
    original_rejection_greedy = rejection_module.greedy_sample
    original_rejection_forward = rejection_module.AscendRejectionSampler.forward

    def sampler_greedy(logits):
        return _packed_greedy(logits) if _LOCAL_TARGET_SAMPLING.get() else original_sampler_greedy(logits)

    def rejection_greedy(logits):
        return _packed_greedy(logits) if _LOCAL_TARGET_SAMPLING.get() else original_rejection_greedy(logits)

    sampler_module.AscendSampler.greedy_sample = staticmethod(sampler_greedy)
    rejection_module.greedy_sample = rejection_greedy

    @wraps(original_rejection_forward)
    def unified_greedy_forward(self, metadata, draft_probs, logits, sampling_metadata):
        if not _LOCAL_TARGET_SAMPLING.get() or not rejection_module.HAS_TRITON:
            return original_rejection_forward(self, metadata, draft_probs, logits, sampling_metadata)
        if (not sampling_metadata.all_greedy or sampling_metadata.max_num_logprobs is not None
                or self.synthetic_mode or draft_probs is not None):
            return original_rejection_forward(self, metadata, draft_probs, logits, sampling_metadata)
        processor = sampling_metadata.logitsprocs.non_argmax_invariant[0]
        rank, local_vocab = _LOCAL_TARGET_SHARD.get()
        mask = _combined_stop_mask(processor, metadata.num_draft_tokens, rank, local_vocab)
        if mask is not None:
            value = getattr(processor, '_vspec_mask_value', None)
            if value is None or value.dtype != logits.dtype:
                value = processor.neg_inf_tensor.to(logits.dtype)
                processor._vspec_mask_value = value
            logits.index_put_(mask, value)
        # Target and bonus rows share one exact global argmax collective.
        # The native strict prefix-rejection kernel consumes those same IDs.
        winners = _packed_greedy(logits)
        target_ids = winners[metadata.target_logits_indices].contiguous()
        bonus_ids = winners[metadata.bonus_logits_indices].unsqueeze(1).contiguous()
        n = len(metadata.num_draft_tokens)
        output = torch.full((n, metadata.max_spec_len + 1), rejection_module.PLACEHOLDER_TOKEN_ID,
                            dtype=torch.int32, device=logits.device)
        grid, block = rejection_module.cal_grid_and_block_size(n)
        rejection_module.rejection_greedy_sample_with_triton(
            output, metadata.num_draft_tokens, metadata.cu_num_draft_tokens,
            metadata.draft_token_ids, target_ids, bonus_ids, None, metadata.max_spec_len, grid, block,
        )
        return rejection_module.SamplerOutput(sampled_token_ids=output, logprobs_tensors=None)

    rejection_module.AscendRejectionSampler.forward = unified_greedy_forward

    # Shadow only the two greedy sampling modules. The global Ascend config,
    # target/draft graph config, and proposer config stay untouched.
    for module in (sampler_module, rejection_module):
        original_get_config = module.get_ascend_config

        @wraps(original_get_config)
        def local_get_config(_original: Any = original_get_config) -> Any:
            config = _original()
            return _GreedyLocalConfig(config) if _LOCAL_TARGET_SAMPLING.get() else config

        module.get_ascend_config = local_get_config

    @wraps(original_load_model)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        model = getattr(self, "model", None)
        parallel = getattr(self, "parallel_config", None)
        tp_size = getattr(parallel, "tensor_parallel_size", None)
        local_head = _target_local_head(model, tp_size) if model is not None else None
        static_eligible = local_head is not None and not lmhead_tp_enable()
        if _tp_vote_count(static_eligible) != (2,):
            logger.warning(
                "vSpec target local-reduce unavailable on at least one TP rank"
            )
            return result
        assert local_head is not None
        _, local_vocab = local_head
        self._vspec_target_local_vocab = local_vocab
        self._vspec_target_tp_rank = get_tp_group().rank_in_group
        original_compute_logits = model.compute_logits

        @wraps(original_compute_logits)
        def compute_logits(
            hidden_states: torch.Tensor, *logit_args: Any, **logit_kwargs: Any
        ) -> Any:
            if _TARGET_EXECUTE_RUNNER.get() is not self:
                return original_compute_logits(hidden_states, *logit_args, **logit_kwargs)
            if logit_args or logit_kwargs or not _plain_greedy_runner(self):
                if not getattr(self, '_vspec_target_fallback_logged', False):
                    sm = self.input_batch.sampling_metadata
                    print(f'[q35-target-fallback] pid={os.getpid()} n={self.input_batch.num_reqs} '
                          f'output_id_lists={len(sm.output_token_ids)} greedy={sm.all_greedy} '
                          f'logprobs={sm.max_num_logprobs} structured={self._vspec_target_structured}', flush=True)
                    self._vspec_target_fallback_logged = True
                return original_compute_logits(hidden_states, *logit_args, **logit_kwargs)
            local_logits = model.compute_logits_local(hidden_states)
            local_valid = bool(
                isinstance(local_logits, torch.Tensor)
                and local_logits.ndim == 2
                and local_logits.shape[0] == hidden_states.shape[0]
                and local_logits.shape[-1] == local_vocab
                and local_logits.device == hidden_states.device
                and local_logits.is_contiguous()
                and local_logits.dtype in (torch.float32, torch.bfloat16, torch.float16)
            )
            if not local_valid:
                raise RuntimeError(f'Unexpected Q35 target local-logit layout: '
                                   f'shape={getattr(local_logits, "shape", None)} '
                                   f'dtype={getattr(local_logits, "dtype", None)} '
                                   f'local_vocab={local_vocab}')
            self._vspec_target_localreduce_pending = True
            count = getattr(self, "_vspec_target_localreduce_count", 0) + 1
            self._vspec_target_localreduce_count = count
            if count in (1, 2, 128, 1024):
                print(f'[q35-target-local] pid={os.getpid()} calls={count} rows={local_logits.shape[0]}', flush=True)
            return local_logits

        model.compute_logits = compute_logits
        self._vspec_target_localreduce_model = model
        return result

    original_execute_model = NPUModelRunner.execute_model

    @wraps(original_execute_model)
    def execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._vspec_target_localreduce_pending = False
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        self._vspec_target_structured = bool(
            getattr(scheduler_output, "has_structured_output_requests", True)
        )
        token = _TARGET_EXECUTE_RUNNER.set(self)
        try:
            return original_execute_model(self, *args, **kwargs)
        finally:
            _TARGET_EXECUTE_RUNNER.reset(token)

    original_sample = NPUModelRunner._sample

    @wraps(original_sample)
    def sample(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        if not getattr(self, "_vspec_target_localreduce_pending", False):
            return original_sample(self, logits, spec_decode_metadata)
        snapshot = None
        if validate:
            gamma = getattr(spec_decode_metadata, 'max_spec_len', 0)
            key = (logits.shape[0], gamma)
            counts = getattr(self, '_vspec_target_parity_counts', None)
            if counts is None:
                counts = self._vspec_target_parity_counts = {}
            counts[key] = counts.get(key, 0) + 1
            if counts[key] in (1, 2, 128):
                # Fast sampling may mask its input in-place. The independent
                # native shadow must receive the original unmasked shard.
                snapshot = logits.clone()
        token = _LOCAL_TARGET_SAMPLING.set(True)
        shard_token = _LOCAL_TARGET_SHARD.set(
            (self._vspec_target_tp_rank, self._vspec_target_local_vocab)
        )
        try:
            result = original_sample(self, logits, spec_decode_metadata)
        finally:
            _LOCAL_TARGET_SHARD.reset(shard_token)
            _LOCAL_TARGET_SAMPLING.reset(token)
        if snapshot is not None:
            full = _full_logits_from_local(self._vspec_target_localreduce_model, snapshot)
            native = original_sample(self, full, spec_decode_metadata)
            assert torch.equal(result.sampled_token_ids, native.sampled_token_ids), 'Target greedy token parity'
            print(f'[q35-target-parity] pid={os.getpid()} gamma={gamma} rows={key[0]} checks={counts[key]}', flush=True)
            return native
        return result

    original_sample_tokens = NPUModelRunner.sample_tokens

    @wraps(original_sample_tokens)
    def sample_tokens(self: Any, grammar_output: Any) -> Any:
        if getattr(self, "_vspec_target_localreduce_model", None) is None:
            return original_sample_tokens(self, grammar_output)
        pending = getattr(self, "_vspec_target_localreduce_pending", False)
        state = getattr(self, "execute_model_state", None)
        model = getattr(self, "_vspec_target_localreduce_model", None)
        head = _target_local_head(model, 2) if model is not None else None
        local_logits = getattr(state, "logits", None)
        processor_list = getattr(
            getattr(getattr(self, "input_batch", None), "sampling_metadata", None),
            "logitsprocs", None
        )
        processor_list = getattr(processor_list, "non_argmax_invariant", None)
        processor = (
            processor_list[0]
            if isinstance(processor_list, list) and len(processor_list) == 1
            else None
        )
        local_layout = bool(
            head is not None
            and isinstance(local_logits, torch.Tensor)
            and local_logits.ndim == 2
            and isinstance(getattr(state, "sample_hidden_states", None), torch.Tensor)
            and local_logits.shape[0] == state.sample_hidden_states.shape[0]
            and local_logits.shape[-1] == head[1]
            and local_logits.device == state.sample_hidden_states.device
            and local_logits.is_contiguous()
        )
        if not pending:
            return original_sample_tokens(self, grammar_output)
        if not local_layout:
            raise RuntimeError('Pending Q35 local target logits lost their expected layout')
        use_local = bool(
            pending
            and local_layout
            and grammar_output is None
            and _plain_greedy_runner(self)
            and processor is not None
            and _spec_min_tokens_rows_safe(processor, state.spec_decode_metadata)
        )
        if use_local:
            try:
                return original_sample_tokens(self, grammar_output)
            finally:
                self._vspec_target_localreduce_pending = False
        assert model is not None
        full_logits = _full_logits_from_local(model, local_logits)
        self.execute_model_state = state._replace(logits=full_logits)
        self._vspec_target_localreduce_pending = False
        try:
            return original_sample_tokens(self, grammar_output)
        finally:
            self._vspec_target_localreduce_pending = False

    load_model._vspec_target_localreduce = True  # type: ignore[attr-defined]
    NPUModelRunner.load_model = load_model
    NPUModelRunner.execute_model = execute_model
    NPUModelRunner._sample = sample
    NPUModelRunner.sample_tokens = sample_tokens
    logger.warning(
        "vSpec installed target greedy-local reduction: static TP vote, replicated CPU admission, packed argmax"
    )
    return True
