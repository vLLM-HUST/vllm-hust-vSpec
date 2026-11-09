"""Compact BF16 proposal projection for Q35 MTP; target verification is full.

The private projection copies selected draft-head rows without changing the
shared original LM head. Missing-vocabulary proposals can be rejected by the
ordinary target verifier. This changes proposal quality/cost, not acceptance.
"""

import os
from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F


def active_ids_for_shard(vocab_size, active_size, start, end, tail_size=512):
    if not 0 < active_size < vocab_size or not 0 <= start < end <= vocab_size:
        raise ValueError('Invalid active-vocabulary geometry')
    tail = min(tail_size, active_size // 4)
    prefix_end = active_size - tail
    return [*range(start, min(end, prefix_end)),
            *range(max(start, vocab_size - tail), end)]


def balanced_active_ids(vocab_size, active_size, rank, world_size=2):
    """Preserve the exact selected ID set and ascending-ID tie order."""
    if world_size != 2 or rank not in (0, 1) or active_size % world_size:
        raise ValueError('Balanced proposal projection requires even TP2 geometry')
    ids = active_ids_for_shard(vocab_size, active_size, 0, vocab_size)
    rows = active_size // world_size
    return ids[rank * rows:(rank + 1) * rows]


class CompactProposalHead:
    def __init__(self, weight, bias, global_ids, shard_start, group, scale=1.0, soft_cap=None):
        if not global_ids:
            raise ValueError('Each proposal TP shard needs at least one candidate')
        self.group = group
        self.scale, self.soft_cap = scale, soft_cap
        self.ids = torch.tensor(global_ids, dtype=torch.int64, device=weight.device)
        local_ids = self.ids - shard_start
        self.weight = weight.index_select(0, local_ids).contiguous()
        self.bias = bias.index_select(0, local_ids).contiguous() if bias is not None else None

    def project(self, hidden):
        logits = F.linear(hidden, self.weight, self.bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1:
            logits = logits * self.scale
        values, indices = logits.max(dim=-1)
        ids = self.ids[indices]
        packed = torch.stack((values.float(), ids.float()), dim=-1)
        gathered = self.group.all_gather(packed, dim=-1).reshape(hidden.shape[0], self.group.world_size, 2)
        winner = gathered[:, :, 0].argmax(-1, keepdim=True)
        return gathered[:, :, 1].gather(1, winner).squeeze(1).to(torch.int64)


class BalancedCompactProposalHead(CompactProposalHead):
    """One-time row redistribution; no additional inference collective.

    All selected rows retain their BF16 values and global token IDs. Only
    private proposal copies move; the full shared target head is untouched.
    """
    def __init__(self, weight, bias, vocab_size, active_size, rank, group,
                 scale=1.0, soft_cap=None):
        if group.world_size != 2 or group.rank_in_group != rank:
            raise ValueError('Unexpected proposal TP group')
        self.group, self.scale, self.soft_cap = group, scale, soft_cap
        ids = balanced_active_ids(vocab_size, active_size, rank)
        self.ids = torch.tensor(ids, dtype=torch.int64, device=weight.device)
        original_rows = vocab_size // 2

        def redistribute(source):
            selected = []
            for destination in range(2):
                destination_ids = balanced_active_ids(vocab_size, active_size, destination)
                for owner in range(2):
                    wanted = [token for token in destination_ids
                              if owner * original_rows <= token < (owner + 1) * original_rows]
                    if not wanted:
                        continue
                    if owner == destination:
                        if rank == owner:
                            local = torch.tensor([token - owner * original_rows for token in wanted],
                                                 dtype=torch.int64, device=source.device)
                            selected.append(source.index_select(0, local))
                        continue
                    if rank == owner:
                        local = torch.tensor([token - owner * original_rows for token in wanted],
                                             dtype=torch.int64, device=source.device)
                        transfer = source.index_select(0, local).contiguous()
                        if transfer.device.type == 'npu':
                            # HCCL transmits storage bytes. Both endpoints
                            # must use ND, not mismatched ND/NZ layouts.
                            import torch_npu
                            if torch_npu.get_npu_format(transfer) != 2:
                                transfer = torch_npu.npu_format_cast(transfer, 2)
                    else:
                        transfer = torch.empty((len(wanted), *source.shape[1:]),
                                               dtype=source.dtype, device=source.device)
                    group.broadcast(transfer, src=owner)
                    if rank == destination:
                        selected.append(transfer)
            return torch.cat(selected, dim=0).contiguous() if len(selected) > 1 else selected[0].contiguous()

        self.weight = redistribute(weight)
        self.bias = redistribute(bias) if bias is not None else None
        if self.weight.shape[0] != len(ids):
            raise RuntimeError('Proposal row redistribution lost rows')

    def validate_against_full_target_rows(self, original_weight, original_bias, rank):
        """Independent full-head gather, validation startup only, never hot path.

        This deliberately uses more memory than redistribution. It checks the
        actual HCCL result against a distinct reference construction and never
        changes the shared target tensor. Host equality reads happen before
        profiling, graph capture, warmup and generation timers.
        """
        import torch_npu

        def dense(value):
            return torch_npu.npu_format_cast(value, 2) if torch_npu.get_npu_format(value) != 2 else value

        version = original_weight._version
        full = self.group.all_gather(dense(original_weight), dim=0)
        reference_rows = full.shape[0]
        reference = full.index_select(0, self.ids)
        actual = dense(self.weight)
        if not torch.equal(actual, reference):
            raise RuntimeError('Balanced draft HCCL rows differ from full target-head gather')
        if original_bias is not None:
            reference_bias = self.group.all_gather(original_bias, dim=0).index_select(0, self.ids)
            if not torch.equal(self.bias, reference_bias):
                raise RuntimeError('Balanced draft bias rows differ')
        if original_weight._version != version:
            raise RuntimeError('Balanced draft validation changed shared target weights')
        print(f'[q35-balanced-row-parity] pid={os.getpid()} rank={rank} '
              f'rows={len(self.ids)} full_reference_rows={reference_rows} bf16_exact=True target_unchanged=True', flush=True)


def install_q35_active_proposal_head():
    from ..config import q35_replicated_inputs

    replica_variant = q35_replicated_inputs()
    size = int(os.getenv('HUST_ISSUE18_Q35_MTP_DRAFT_ACTIVE_SIZE', '0'))
    # Private proposal-only TP2 redistribution; default off; non-sensitive.
    balanced = os.getenv('HUST_ISSUE18_Q35_MTP_DRAFT_BALANCED', '0') == '1'
    if size == 0:
        if replica_variant != 'off':
            raise ValueError('Draft input replicas require the active proposal head')
        if balanced:
            raise ValueError('Balanced proposal head requires an active vocabulary')
        return False
    if size not in (16384, 32768, 65536, 98304):
        raise ValueError('Q35 proposal active size must be16384,32768,65536,98304 or0')
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

    original_load = AscendEagleProposer.load_model
    if getattr(original_load, '_vspec_q35_active_proposal', False):
        return False

    @wraps(original_load)
    def load_model(self, *args, **kwargs):
        target_model = args[0] if args else kwargs.get('model')
        result = original_load(self, *args, **kwargs)
        cfg = self.vllm_config
        model_type = getattr(cfg.model_config.hf_text_config, 'model_type', '')
        if (type(self) is not AscendEagleProposer or self.method != 'mtp'
                or not str(model_type).startswith('qwen3_5')
                or cfg.parallel_config.tensor_parallel_size != 2
                or cfg.quant_config is not None or cfg.lora_config is not None
                or not self.use_local_argmax_reduction):
            raise RuntimeError('Compact Q35 proposal head requires unquantized TP2 greedy MTP/local argmax')
        model = self.model
        if model is target_model:
            raise RuntimeError('Proposal projection must never replace target model methods')
        head, processor = model.lm_head, model.logits_processor
        weight = head.weight
        original_weight_version = weight._version
        if (processor.org_vocab_size != 248320 or head.num_org_embeddings_per_partition != 124160
                or head.num_embeddings_per_partition != 124160 or head.tp_size != 2
                or weight.dtype != torch.bfloat16):
            raise RuntimeError('Unexpected Q35 BF16 proposal vocabulary sharding')
        start, end = head.shard_indices.org_vocab_start_index, head.shard_indices.org_vocab_end_index
        ids = (balanced_active_ids(processor.org_vocab_size, size, head.tp_rank) if balanced
               else active_ids_for_shard(processor.org_vocab_size, size, start, end))
        if balanced:
            compact = BalancedCompactProposalHead(weight, getattr(head, 'bias', None),
                processor.org_vocab_size, size, head.tp_rank, get_tp_group(),
                scale=processor.scale, soft_cap=processor.soft_cap)
            # Independent full-head reference at load time in BOTH modes.
            # The sync and temporary gather are outside all benchmark timers;
            # no target tensor or hot inference collective is changed.
            compact.validate_against_full_target_rows(weight, getattr(head, 'bias', None), head.tp_rank)
        else:
            compact = CompactProposalHead(weight, getattr(head, 'bias', None), ids, start, get_tp_group(),
                                          scale=processor.scale, soft_cap=processor.soft_cap)
        import torch_npu
        original_format = torch_npu.get_npu_format(weight)
        if torch_npu.get_npu_format(compact.weight) != original_format:
            compact.weight = torch_npu.npu_format_cast(compact.weight, original_format)
        assert weight._version == original_weight_version, 'Shared LM-head weight changed'
        if replica_variant != 'off':
            from vllm_ascend import envs as ascend_envs
            from .q35_draft_inputs import ReplicatedDraftInputs

            if not balanced or size != 98304 or ascend_envs.VLLM_ASCEND_Q35_GDN_INPUT_FUSION not in ('on', 'validate'):
                raise RuntimeError('Draft replicas require the private balanced96K dynamic4/6 candidate')
            bank = ReplicatedDraftInputs(self, target_model, get_tp_group(), replica_variant,
                                         ascend_envs.VLLM_ASCEND_Q35_GDN_INPUT_FUSION == 'validate')
            bank.install(self.runner._q35_gdn_input_manager)
            self._vspec_replicated_draft_inputs = bank
        self._vspec_active_proposal_head = compact
        self._vspec_original_proposal_get_top_tokens = model.get_top_tokens

        def top_tokens(_model, hidden_states):
            return compact.project(hidden_states)

        model.get_top_tokens = MethodType(top_tokens, model)
        print(f'[q35-active-proposal] pid={os.getpid()} rank={head.tp_rank} '
              f'global_size={size} local_size={len(ids)} balanced={balanced} target_unchanged=True', flush=True)
        return result

    load_model._vspec_q35_active_proposal = True
    AscendEagleProposer.load_model = load_model
    return True
