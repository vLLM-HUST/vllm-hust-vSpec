"""Default-off immutable draft input replicas. Never patch shared embedding.

The embedding module may alias the target. We override only the distinct
draft predictor's lookup method and its private FC method, not that shared
module, target model, native verifier, EOS policy, or sampling metadata.
"""

import os
from types import MethodType

import torch
import torch_npu

from .q35_draft_input_kernel import embedding, full_fc, split_fc
from .q35_draft_input_shadow import record_validation


class ReplicatedDraftInputs:
    def __init__(self, proposer, target_model, group, variant, validate):
        if variant not in ('embedding', 'split', 'full') or group.world_size != 2:
            raise ValueError('Draft input replicas require an admitted TP2 variant')
        self.variant, self.validate = variant, validate
        self.predictor = proposer.model.model
        target_ids = {id(module) for module in target_model.modules()}
        if id(self.predictor) in target_ids or id(self.predictor.fc) in target_ids:
            raise RuntimeError('Draft predictor and FC must not alias target modules')
        self.original_embedding = self.predictor.embed_input_ids
        self.original_fc = self.predictor.fc.forward
        embed, fc = self.predictor.embed_tokens, self.predictor.fc
        if (embed.forward_type is not None or embed.tp_size != 2 or embed.org_vocab_size != 248320
                or embed.num_embeddings != 248320 or embed.num_embeddings_per_partition != 124160
                or embed.num_added_embeddings != 0 or embed.embedding_dim != 2048
                or tuple(embed.weight.shape) != (124160, 2048)
                or tuple(fc.weight.shape) != (1024, 4096) or fc.bias is not None
                or fc.return_bias or not fc.gather_output or fc.tp_size != 2
                or embed.weight.dtype != torch.bfloat16 or fc.weight.dtype != torch.bfloat16):
            raise RuntimeError('Unexpected unquantized Q35 draft input geometry')
        self.embedding_source, self.fc_source = embed.weight, fc.weight
        self.versions = (embed.weight._version, fc.weight._version)
        rank = group.rank_in_group

        def dense(value):
            return torch_npu.npu_format_cast(value, 2) if torch_npu.get_npu_format(value) != 2 else value

        self.full_embedding = group.all_gather(dense(embed.weight), dim=0).contiguous()
        if tuple(self.full_embedding.shape) != (248320, 2048):
            raise RuntimeError('Embedding replica did not preserve the full vocabulary')
        if not torch.equal(self.full_embedding[rank * 124160:(rank + 1) * 124160], dense(embed.weight)):
            raise RuntimeError('Replicated embedding changed original BF16 rows')
        self.full_fc = self.fc_parts = None
        if variant != 'embedding':
            full = group.all_gather(dense(fc.weight), dim=0).contiguous()
            if tuple(full.shape) != (2048, 4096) or not torch.equal(full[rank * 1024:(rank + 1) * 1024], dense(fc.weight)):
                raise RuntimeError('Replicated draft FC changed native BF16 rows')
            layout = torch_npu.get_npu_format(fc.weight)
            if variant == 'split':
                self.fc_parts = tuple(torch_npu.npu_format_cast(part.contiguous(), layout)
                                      for part in full.chunk(2, 0))
            else:
                self.full_fc = torch_npu.npu_format_cast(full, layout)
        self.counts = torch.zeros((8193, 2), dtype=torch.int64, device=embed.weight.device)
        self.errors = torch.zeros((8193, 2), dtype=torch.int32, device=embed.weight.device)
        self.captured_embedding, self.captured_fc = set(), set()
        # Distinct reference construction: execute the ORIGINAL TP lookup
        # for EVERY immutable model token, chunked outside all timers. This
        # checks remote as well as local rows and native BF16 zero semantics.
        for start in range(0, 248320, 4096):
            ids = torch.arange(start, min(start + 4096, 248320), dtype=torch.int64, device=embed.weight.device)
            actual = embedding(ids, self.full_embedding)
            expected = self.original_embedding(ids)
            if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                raise RuntimeError('Draft input lookup differs from original native TP for full vocabulary')
        if variant != 'embedding':
            for rows in (1, 14, 70, 98):
                hidden = ((torch.arange(rows * 4096, dtype=torch.float32, device=fc.weight.device)
                           .reshape(rows, 4096) % 127 - 63) / 128).to(torch.bfloat16)
                actual = split_fc(hidden, self.fc_parts) if variant == 'split' else full_fc(hidden, self.full_fc)
                expected = self.original_fc(hidden)
                if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    raise RuntimeError('Replicated draft FC differs from original native TP startup reference')
        self.check_sources()
        print(f'[q35-draft-input-replica] pid={os.getpid()} rank={rank} variant={variant} '
              f'validate={validate} full_vocab=248320 original_rows_exact=True all_vocab_native_lookup_bits_exact=True '
              'target_methods_unchanged=True shared_embedding_forward_unchanged=True', flush=True)

    def check_sources(self):
        if (self.predictor.embed_tokens.weight is not self.embedding_source
                or self.predictor.fc.weight is not self.fc_source):
            raise RuntimeError('Immutable draft input source changed after load')
        # _version becomes an unbacked symbolic integer inside Dynamo.
        # Constructor, eager calls, and post-timer metrics all enforce it;
        # graph replay is admitted only with the immutable original tensors.
        if (not torch.compiler.is_compiling()
                and (self.embedding_source._version, self.fc_source._version) != self.versions):
            raise RuntimeError('Immutable draft input source changed after load')

    def compare(self, result, expected, rows, feature):
        record_validation(result, expected, self.counts, self.errors, feature)

    def lookup(self, input_ids):
        self.check_sources()
        rows = input_ids.numel()
        if not 0 < rows <= 8192:
            return self.original_embedding(input_ids)
        result = embedding(input_ids, self.full_embedding)
        if self.validate:
            self.compare(result, self.original_embedding(input_ids), rows, 0)
        if not torch.compiler.is_compiling():
            self.captured_embedding.add(rows)
        return result

    def project(self, hidden):
        self.check_sources()
        rows = hidden.shape[0]
        if hidden.ndim != 2 or hidden.shape[1] != 4096 or not 0 < rows <= 8192 or hidden.dtype != torch.bfloat16:
            return self.original_fc(hidden)
        result = split_fc(hidden, self.fc_parts) if self.variant == 'split' else full_fc(hidden, self.full_fc)
        if self.validate:
            self.compare(result, self.original_fc(hidden), rows, 1)
        if not torch.compiler.is_compiling():
            self.captured_fc.add(rows)
        return result

    def install(self, manager):
        bank = self
        def lookup(_predictor, input_ids):
            return bank.lookup(input_ids)
        def project(_fc, hidden):
            return bank.project(hidden)
        self.predictor.embed_input_ids = MethodType(lookup, self.predictor)
        if self.variant != 'embedding':
            self.predictor.fc.forward = MethodType(project, self.predictor.fc)
        original_metrics, original_reset = manager.metrics, manager.reset_stats
        def metrics(_manager):
            records = original_metrics()
            if records:
                records[0]['replicated_draft_inputs'] = bank.metrics()
            return records
        def reset(_manager):
            result = original_reset()
            bank.counts.zero_()
            bank.errors.zero_()
            return result
        manager.metrics, manager.reset_stats = MethodType(metrics, manager), MethodType(reset, manager)

    def metrics(self):
        self.check_sources()
        counts, errors = self.counts.cpu().tolist(), self.errors.cpu().tolist()
        return dict(variant=self.variant, validate=self.validate, full_vocab=248320,
                    target_and_shared_embedding_methods_unchanged=True,
                    captured_embedding=sorted(self.captured_embedding), captured_fc=sorted(self.captured_fc),
                    python_shape_sets_incomplete_for_compiled_paths=True,
                    actual_validation_counts={str(rows): pair for rows, pair in enumerate(counts) if any(pair)},
                    native_bitwise_error_flags={str(rows): errors[rows] for rows, pair in enumerate(counts) if any(pair)},
                    on_mode_replay_counts_not_instrumented=not self.validate)
