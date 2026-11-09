"""Default-off load-only private draft-head NZ; ON inference is unchanged.

Unlike the earlier dual-layout experiment, ON does not retain the original
ND proposal copy. The shared full target head is never transformed or edited.
Validation alone retains an independent ND oracle and injects a GPU shadow
after the exact original projection, with all remaining head code untouched.
"""
import ast
import copy
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
from types import MethodType

import torch
import torch.nn.functional as F

from .experimental_paths import artifact_dir, require_current_npu_proof

PRIMITIVE = artifact_dir('VSPEC_INPLACE_HEAD_NZ_DIR')
PROOF = PRIMITIVE / 'real-head-runtime-true-r1-primitive-audit.json'


def verify_primitive():
    data = json.loads(PROOF.read_text())
    require_current_npu_proof(data)
    assert data['completed'] and data['actual_public_checkpoint_weights']
    assert data['eligible_shapes']['draft'] == list(range(1, 17))
    for proof in (data, *data['ranks']):
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == d for p, d in proof['sha256'].items())
    return data


def shadow_logits(compact, hidden, actual):
    expected = F.linear(hidden, compact._q35_head_native_ND, compact.bias)
    flag = (((actual.view(torch.int16) != expected.view(torch.int16))
             | ~torch.isfinite(actual) | ~torch.isfinite(expected)).any()).to(torch.int32)
    rows = hidden.shape[0]
    compact._q35_head_checks[rows].add_(1)
    torch.maximum(compact._q35_head_errors[rows], flag, out=compact._q35_head_errors[rows])


def validation_project(original):
    method = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    assert isinstance(method, ast.FunctionDef) and method.name == 'project'
    assert ast.unparse(method.body[0]) == 'logits = F.linear(hidden, self.weight, self.bias)'
    result = copy.deepcopy(method)
    result.body.insert(1, ast.parse('_q35_shadow_head_logits(self, hidden, logits)').body[0])
    module = ast.fix_missing_locations(ast.Module(body=[result], type_ignores=[]))
    namespace = dict(original.__globals__, _q35_shadow_head_logits=shadow_logits)
    exec(compile(module, '<private-head-native-shadow>', 'exec'), namespace)
    return wraps(original)(namespace['project'])


def validate_geometry(compact, target_weight, original_format):
    original = compact.weight
    if (original.shape != (49152, 2048) or original.dtype != torch.bfloat16
            or not original.is_contiguous() or original_format not in (0, 2)
            or compact.bias is not None or compact.scale != 1 or compact.soft_cap is not None
            or compact.group.world_size != 2 or compact.ids.numel() != 49152
            or original.untyped_storage().data_ptr() == target_weight.untyped_storage().data_ptr()):
        raise RuntimeError('Require private balanced96K original BF16 ND proposal geometry')


def verify_conversion(original, packed, recovered, original_version, target, target_version, actual_format):
    if (not torch.equal(original.view(torch.int16), recovered.view(torch.int16))
            or original._version != original_version or target._version != target_version
            or packed.shape != original.shape or packed.dtype != original.dtype
            or actual_format != 29):
        raise RuntimeError('Head NZ cast changed original values, precision or target')


def convert_head(compact, target_weight, mode, rank):
    import torch_npu

    if mode not in ('on', 'validate') or rank not in (0, 1):
        raise ValueError('Private head replacement requires explicit on/validate TP2')
    if hasattr(compact, '_q35_inplace_head_receipt'):
        raise RuntimeError('Private head replacement installed twice')
    original = compact.weight
    validate_geometry(compact, target_weight, torch_npu.get_npu_format(original))
    proof = verify_primitive()
    cpu = original.detach().cpu().contiguous()
    digest = hashlib.sha256(memoryview(cpu.view(torch.int16).numpy()).cast('B')).hexdigest()
    rank_proof = next(r for r in proof['ranks'] if r['rank'] == rank)
    if digest != rank_proof['provenance']['balanced_draft_sha256']:
        raise RuntimeError('Actual private proposal head differs from proved real checkpoint')
    target_version = target_weight._version
    original_version = original._version
    packed = torch_npu.npu_format_cast(original, 29)
    recovered = torch_npu.npu_format_cast(packed, 2)
    verify_conversion(original, packed, recovered, original_version, target_weight,
                      target_version, torch_npu.get_npu_format(packed))
    project = type(compact).project
    if mode == 'validate':
        compact._q35_head_native_ND = original
        compact._q35_head_checks = torch.zeros(17, dtype=torch.int64, device=original.device)
        compact._q35_head_errors = torch.zeros(17, dtype=torch.int32, device=original.device)
        compact.project = MethodType(validation_project(project), compact)
    compact.weight = packed
    if mode == 'on' and ('project' in compact.__dict__ or hasattr(compact, '_q35_head_native_ND')):
        raise RuntimeError('ON must use original project without retaining ND oracle')
    receipt = dict(pid=os.getpid(), rank=rank, mode=mode, actual_format=29,
                   shape=[49152, 2048], complete_original_BF16_weight_bits_exact=True,
                   actual_real_weight_sha256=digest, shared_target_weight_version=target_version,
                   shared_target_weight_unchanged=True,
                   original_project_unchanged_ON=mode == 'on' and not getattr(project, '_q35_fused_proposal_vote', False),
                   original_linear_projection_unchanged=True,
                   postprojection_vote_fused=getattr(project, '_q35_fused_proposal_vote', False),
                   original_ND_weight_retained_only_in_validation=mode == 'validate',
                   extra_persistent_weight_bytes_ON=0,
                   private_draft_weight_values_and_dtype_unchanged=True)
    compact._q35_inplace_head_receipt = receipt
    print('[q35-inplace-head-nz-load] ' + json.dumps(receipt), flush=True)


def install_inplace_head_nz():
    from vllm_ascend import envs
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
    from vllm_ascend.worker.worker import NPUWorker

    verify_primitive()
    original_load = AscendEagleProposer.load_model
    if getattr(original_load, '_q35_inplace_head_nz', False):
        return

    @wraps(original_load)
    def load_model(self, target_model):
        result = original_load(self, target_model)
        compact = self._vspec_active_proposal_head
        convert_head(compact, target_model.language_model.lm_head.weight,
                     envs.VLLM_ASCEND_Q35_GDN_INPUT_FUSION, compact.group.rank_in_group)
        return result

    load_model._q35_inplace_head_nz = True
    AscendEagleProposer.load_model = load_model
    original_metrics = NPUWorker.get_private_q35_gdn_input_metrics
    original_reset = NPUWorker.reset_private_q35_gdn_input_stats

    @wraps(original_metrics)
    def metrics(self):
        import torch_npu

        records = original_metrics(self)
        head = self.model_runner.drafter._vspec_active_proposal_head
        receipt = head._q35_inplace_head_receipt
        if (torch_npu.get_npu_format(head.weight) != 29 or not records
                or self.model_runner.model.language_model.lm_head.weight._version
                != receipt['shared_target_weight_version']):
            raise RuntimeError('Actual private head NZ layout missing')
        if receipt['mode'] == 'validate':
            counts, errors = head._q35_head_checks.cpu().tolist(), head._q35_head_errors.cpu().tolist()
        else:
            if 'project' in head.__dict__ or hasattr(head, '_q35_head_native_ND'):
                raise RuntimeError('ON unexpectedly retains a runtime wrapper or ND source')
            counts, errors = None, None
        records[0]['inplace_draft_head_nz'] = dict(receipt, validation_counts=counts,
                                                output_logit_bitwise_error_flags=errors,
                                                stats_read_after_timers_only=True)
        return records

    @wraps(original_reset)
    def reset(self):
        result = original_reset(self)
        head = self.model_runner.drafter._vspec_active_proposal_head
        if head._q35_inplace_head_receipt['mode'] == 'validate':
            head._q35_head_checks.zero_()
            head._q35_head_errors.zero_()
        return result

    NPUWorker.get_private_q35_gdn_input_metrics = metrics
    NPUWorker.reset_private_q35_gdn_input_stats = reset
